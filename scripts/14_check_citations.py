"""Verify BibTeX metadata, DOI resolution, and all report citation keys.

Default: fresh registry + DOI checks. --offline reuses the last evidence cache and
recompares it against the CURRENT bibliography; it never treats a changed entry as
verified just because its key was seen before. --metadata-only is for research
before chapters exist, and does not satisfy the final report gate.
Set CITATION_CONTACT_EMAIL to add your actual contact email to the polite User-Agent.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from difflib import SequenceMatcher
import hashlib
import html
import json
import os
from pathlib import Path
import re
import time
import unicodedata
from urllib.parse import quote, urlparse

import requests

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / 'docs' / 'research' / 'citation_evidence.json'
# Crossref lacks the date for the original PSO proceedings paper. Do not infer a
# year from DOI digits or the deposit date. This publisher record was inspected.
PUBLISHER_YEARS = {
    '10.1109/icnn.1995.488968': {
        'year': 1995,
        'url': 'https://ieeexplore.ieee.org/document/488968/',
        'evidence': 'Date of Conference: 27 November 1995 - 01 December 1995',
        'checked_on': '2026-11-01',
    }
}
OFFICIAL_URLS = {
    'pib2026tomato': 'https://www.pib.gov.in/PressReleasePage.aspx?PRID=2241379&lang=1&reg=3'
}


def parse_bibtex(text: str) -> dict[str, dict[str, str]]:
    """Parse literal braced/quoted BibTeX fields, including nested LaTeX braces.

    Fail closed for unsupported concatenation/macros instead of verifying a
    truncated or guessed field. No external command or LaTeX is executed.
    """
    entries = {}
    cursor = 0
    while found := re.search(r'@([A-Za-z]+)\s*\{\s*([^,\s]+)\s*,', text[cursor:]):
        kind, key = found.groups()
        cursor += found.end()
        if key in entries:
            raise ValueError(f'Duplicate BibTeX key: {key}')
        fields = {'entry_type': kind.lower()}
        while cursor < len(text):
            while cursor < len(text) and (text[cursor].isspace() or text[cursor] == ','):
                cursor += 1
            if cursor < len(text) and text[cursor] == '}':
                cursor += 1
                break
            match = re.match(r'([A-Za-z][\w-]*)\s*=\s*', text[cursor:])
            if not match:
                raise ValueError(f'Invalid field in {key} near {text[cursor:cursor+40]!r}')
            name = match[1].lower()
            cursor += match.end()
            start = cursor
            if text[cursor] == '{':
                cursor += 1
                depth = 1
                start = cursor
                while cursor < len(text) and depth:
                    if text[cursor] == '\\':
                        cursor += 2
                        continue
                    depth += (text[cursor] == '{') - (text[cursor] == '}')
                    cursor += 1
                if depth:
                    raise ValueError(f'Unclosed field {key}.{name}')
                value = text[start:cursor-1]
            elif text[cursor] == '"':
                cursor += 1
                start = cursor
                while cursor < len(text) and text[cursor] != '"':
                    cursor += 2 if text[cursor] == '\\' else 1
                if cursor == len(text):
                    raise ValueError(f'Unclosed quoted field {key}.{name}')
                value = text[start:cursor]
                cursor += 1
            else:
                while cursor < len(text) and text[cursor] not in ',}':
                    cursor += 1
                value = text[start:cursor].strip()
                if not value.isdigit():
                    raise ValueError(f'Unsupported BibTeX macro in {key}.{name}')
            if name in fields:
                raise ValueError(f'Duplicate field {key}.{name}')
            fields[name] = value
        else:
            raise ValueError(f'Unclosed entry {key}')
        entries[key] = fields
    if not entries:
        raise ValueError('Bibliography has no entries')
    return entries


def normalize(value: str) -> str:
    value = html.unescape(re.sub(r'<[^>]+>', '', value))
    value = re.sub(r'\\[\"\'`^~=.uvHckbdtr]\s*\{?([A-Za-z])\}?', r'\1', value)
    value = value.replace('\\&', '&').replace('\\_', '_')
    value = unicodedata.normalize('NFKD', value)
    return ''.join(c.lower() for c in value if c.isalnum() and not unicodedata.combining(c))


def metadata_fields(registry: str, data: dict) -> tuple[list[str], str, set[int]]:
    if registry == 'datacite':
        titles = [x['title'] for x in data.get('titles', [])]
        authors = data.get('creators') or [{}]
        family = authors[0].get('familyName') or authors[0].get('name', '').split(',')[0]
        years = {int(data['publicationYear'])} if data.get('publicationYear') else set()
    else:
        titles = list(data.get('title', []))
        if data.get('subtitle') and titles:
            titles.append(titles[0] + ': ' + ': '.join(data['subtitle']))
        family = (data.get('author') or [{}])[0].get('family', '')
        years = set()
        for field in ('published', 'published-print', 'published-online', 'issued'):
            for date in (data.get(field) or {}).get('date-parts', []):
                if date and date[0]:
                    years.add(int(date[0]))
    return titles, family, years


def compare_entry(entry: dict, evidence: dict) -> dict:
    titles, family, years = metadata_fields(evidence['registry'], evidence['metadata'])
    title_similarity = max((SequenceMatcher(None, normalize(entry.get('title', '')), normalize(t)).ratio() for t in titles), default=0)
    first_author = entry.get('author', '').split(' and ')[0]
    first_author = first_author.split(',')[0] if ',' in first_author else first_author.split()[-1] if first_author else ''
    author_matches = bool(family) and normalize(first_author) == normalize(family)
    year = int(entry['year']) if entry.get('year', '').isdigit() else None
    year_matches = year in years
    doi = entry.get('doi', '').lower()
    supplement = PUBLISHER_YEARS.get(doi) if not years else None
    publisher_year_matches = bool(supplement and year == supplement['year'])
    returned_doi = evidence['metadata'].get('DOI', evidence['metadata'].get('doi', '')).lower()
    doi_matches = doi == returned_doi
    resolution = evidence.get('doi_resolution', {})
    resolved = resolution.get('resolved', False)
    ok = title_similarity >= .9 and author_matches and (year_matches or publisher_year_matches) and doi_matches and resolved
    return {
        'status': ('verified_with_publisher_year' if publisher_year_matches else 'verified') if ok else 'failed',
        'doi': doi, 'registry': evidence['registry'],
        'registry_url': evidence.get('registry_url'),
        'registered_titles': titles, 'title_similarity': round(title_similarity, 6),
        'registered_first_author': family, 'first_author_matches': author_matches,
        'registered_years': sorted(years), 'year_matches_registry': year_matches,
        'publisher_year_evidence': supplement, 'doi_matches_registry': doi_matches,
        'doi_resolution': resolution, 'evidence_checked_at': evidence.get('checked_at'),
    }


def request(session: requests.Session, url: str, **kwargs):
    last = None
    for attempt in range(4):
        try:
            response = session.get(url, timeout=30, **kwargs)
            if response.status_code == 429 or response.status_code >= 500:
                last = RuntimeError(f'{response.status_code} {url}')
                time.sleep(min(12, 2 ** (attempt + 1)))
                continue
            response.raise_for_status()
            return response
        except requests.RequestException as exc:
            last = exc
            if attempt < 3:
                time.sleep(2 ** attempt)
    raise RuntimeError(str(last))


def fetch_evidence(session: requests.Session, doi: str) -> dict:
    encoded = quote(doi, safe='')
    registry = 'datacite' if doi.lower().startswith('10.48550/') else 'crossref'
    endpoint = ('https://api.datacite.org/dois/' if registry == 'datacite' else 'https://api.crossref.org/works/') + encoded
    data = request(session, endpoint).json()
    metadata = data['data']['attributes'] if registry == 'datacite' else data['message']
    time.sleep(.5)
    # A resolver redirect verifies the DOI exists. A publisher paywall/robot
    # check at the redirect destination is not falsely reported as an invalid DOI.
    resolution = request(session, 'https://doi.org/' + quote(doi, safe='/:'), allow_redirects=False)
    destination = resolution.headers.get('Location', '')
    resolved = (resolution.status_code in (301, 302, 303, 307, 308) and urlparse(destination).scheme in ('http', 'https')) or resolution.status_code == 200
    return {'registry': registry, 'registry_url': endpoint, 'metadata': metadata,
            'checked_at': datetime.now(timezone.utc).isoformat(),
            'doi_resolution': {'resolved': bool(resolved), 'status_code': resolution.status_code, 'destination': destination}}


def citation_usage(tex_files: list[Path], entries: dict) -> dict:
    used = set()
    by_file = {}
    pattern = re.compile(r'\\(?:cite|citep|citet|citeauthor|citeyear|parencite|textcite|autocite)\*?(?:\[[^\]]*\]){0,2}\{([^}]+)\}')
    for path in tex_files:
        text = re.sub(r'(?<!\\)%[^\n]*', '', path.read_text(encoding='utf-8'))
        keys = {key.strip() for value in pattern.findall(text) for key in value.split(',') if key.strip()}
        by_file[str(path.relative_to(ROOT))] = sorted(keys)
        used.update(keys)
    return {'used_keys': sorted(used), 'unknown_keys': sorted(used-set(entries)),
            'uncited_keys': sorted(set(entries)-used), 'by_file': by_file}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--offline', action='store_true', help='Recompare current bibliography with saved retrieval evidence')
    parser.add_argument('--metadata-only', action='store_true', help='Research-stage check; does not satisfy the final gate')
    args = parser.parse_args()
    bib_path = ROOT / 'report' / 'references.bib'
    entries = parse_bibtex(bib_path.read_text(encoding='utf-8'))
    evidence = json.loads(CACHE.read_text(encoding='utf-8')) if CACHE.exists() else {}
    session = requests.Session()
    contact = os.environ.get('CITATION_CONTACT_EMAIL', '').strip()
    session.headers['User-Agent'] = 'Team10-BIA-CitationChecker/1.0 (+https://github.com/Suraj-B12/BIO-INSPIRED-ALGO---AAT)' + (f' (mailto:{contact})' if contact else '')
    checked = {}
    for key, entry in entries.items():
        doi = entry.get('doi', '').strip().lower()
        print(f'Checking {key} ...', flush=True)
        try:
            if not doi:
                allowed = key in OFFICIAL_URLS and entry.get('url') == OFFICIAL_URLS[key] and bool(entry.get('urldate'))
                checked[key] = {'status': 'allowed_official_page' if allowed else 'failed', 'doi': None,
                                'url': entry.get('url'), 'access_date': entry.get('urldate'),
                                'exception_reason': 'Indispensable official government statistics source; no DOI assigned.' if allowed else 'Missing DOI without an approved official-source exception.'}
                continue
            if not args.offline:
                evidence[doi] = fetch_evidence(session, doi)
                CACHE.parent.mkdir(parents=True, exist_ok=True)
                CACHE.write_text(json.dumps(evidence, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
                time.sleep(1)
            if doi not in evidence:
                raise ValueError('No saved verification evidence; run the online checker first')
            checked[key] = compare_entry(entry, evidence[doi])
        except Exception as exc:
            checked[key] = {'status': 'failed', 'doi': doi, 'error': str(exc)}
    usage = citation_usage(list((ROOT/'report').rglob('*.tex')), entries)
    metadata_ok = all(x['status'] != 'failed' for x in checked.values())
    usage_ok = not usage['unknown_keys'] and not usage['uncited_keys']
    passed = metadata_ok and (args.metadata_only or usage_ok)
    result = {
        'checked_at': datetime.now(timezone.utc).isoformat(),
        'mode': 'offline_cached_evidence' if args.offline else 'live_registry_verification',
        'scope': 'metadata_only' if args.metadata_only else 'full_report_gate',
        'passed': passed, 'final_gate_passed': passed and not args.metadata_only,
        'bibliography_sha256': hashlib.sha256(bib_path.read_bytes()).hexdigest(),
        'contact_email_configured': bool(contact),
        'contact_note': 'Set CITATION_CONTACT_EMAIL for an actual email; the project contact URL is always sent. No email address is invented.',
        'entries': checked, 'citation_usage': usage,
        'counts': {'entries': len(entries), 'verified_dois': sum(x['status'].startswith('verified') for x in checked.values()),
                   'official_exceptions': sum(x['status']=='allowed_official_page' for x in checked.values()),
                   'failed': sum(x['status']=='failed' for x in checked.values())},
    }
    output = ROOT/'results'/'citation_check.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
    print(json.dumps({'passed': passed, 'scope': result['scope'], 'counts': result['counts'],
                      'unknown': usage['unknown_keys'], 'uncited': usage['uncited_keys']}, indent=2))
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())

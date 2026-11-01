"""The report gate must reject mismatched evidence rather than rubber-stamp it."""
import importlib.util
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('citation_checker', ROOT/'scripts'/'14_check_citations.py')
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)


def evidence():
    return {'registry': 'crossref', 'metadata': {'DOI': '10.1234/example', 'title': ['A useful method'], 'author': [{'family':'Smith'}], 'published': {'date-parts':[[2020]]}}, 'doi_resolution': {'resolved':True}}


def entry():
    return {'doi':'10.1234/example','title':'{A useful method}','author':'Smith, A. and Jones, B.','year':'2020'}


def test_nested_bibtex_and_escaped_characters():
    parsed=checker.parse_bibtex('@article{a, title={{A {CNN} method}}, author={Smith, A.}, year={2020}, doi={10.1234/x}}')
    assert parsed['a']['title']=='{A {CNN} method}'
    assert checker.normalize(r'S{\"o}rensen') == 'sorensen'


@pytest.mark.parametrize('text', ['@article{a,title={x}', '@article{a,title={x},title={y}}', '@article{a,title={x}}\n@book{a,title={y}}', '@article{a,title=unresolvedmacro}'])
def test_parser_rejects_ambiguous_or_malformed_input(text):
    with pytest.raises(ValueError): checker.parse_bibtex(text)


def test_exact_evidence_passes():
    assert checker.compare_entry(entry(),evidence())['status']=='verified'


@pytest.mark.parametrize(('field','value'), [('title','Entirely different research'),('author','Jones, B.'),('year','2019'),('doi','10.1234/unrelated')])
def test_current_entry_must_match_saved_evidence(field,value):
    changed=entry();changed[field]=value
    assert checker.compare_entry(changed,evidence())['status']=='failed'


def test_doi_must_actually_resolve():
    saved=evidence();saved['doi_resolution']['resolved']=False
    assert checker.compare_entry(entry(),saved)['status']=='failed'


def test_print_and_online_years_are_both_valid():
    saved=evidence();saved['metadata']['published-print']={'date-parts':[[2021]]}
    changed=entry();changed['year']='2021'
    assert checker.compare_entry(changed,saved)['status']=='verified'


def test_citation_usage_does_not_hide_unknown_or_uncited_keys(tmp_path,monkeypatch):
    monkeypatch.setattr(checker,'ROOT',tmp_path)
    tex=tmp_path/'chapter.tex';tex.write_text(r'\cite{known,missing} % \cite{unused}'+'\n'+r'\nocite{*}',encoding='utf-8')
    usage=checker.citation_usage([tex],{'known':{},'unused':{}})
    assert usage['unknown_keys']==['missing']
    assert usage['uncited_keys']==['unused']


@pytest.mark.parametrize('registry', ['crossref', 'datacite'])
def test_selected_metadata_preserves_verification_without_source_content(registry):
    if registry == 'crossref':
        raw = {
            **evidence()['metadata'], 'publisher': 'Example Press',
            'URL': 'https://doi.org/10.1234/example',
            'link': [{'URL': 'https://example.org/paper', 'content-type': 'text/html'}],
            'author': [{'family': 'Smith', 'given': 'A.', 'affiliation': [{'name': 'Unneeded'}]}],
            'published-online': {'date-parts': [[2019, 12]], 'timestamp': 123},
        }
    else:
        raw = {
            'doi': '10.1234/example', 'titles': [{'title': 'A useful method'}],
            'creators': [{'name': 'Smith, A.', 'familyName': 'Smith', 'givenName': 'A.', 'affiliation': ['Unneeded']}],
            'publicationYear': 2020, 'publisher': 'Example Press',
            'url': 'https://example.org/paper',
        }
    raw.update({'abstract': 'Full abstract must not ship',
                'descriptions': [{'description': 'Full text must not ship'}],
                'reference': [{'unstructured': 'Unneeded citation'}],
                'relatedIdentifiers': [{'relatedIdentifier': 'Unneeded reference'}],
                'xml': '<record>Full source text</record>'})
    expected_fields = checker.metadata_fields(registry, raw)
    selected = checker.select_metadata(registry, raw)
    assert checker.metadata_fields(registry, selected) == expected_fields
    assert selected['publisher'] == 'Example Press'
    assert (selected.get('URL') or selected.get('url')).startswith('https://')
    assert not {'abstract', 'descriptions', 'reference', 'relatedIdentifiers', 'xml'} & selected.keys()
    assert 'affiliation' not in (selected.get('author') or selected['creators'])[0]
    assert checker.select_metadata(registry, selected) == selected
    saved = {'registry': registry, 'metadata': selected, 'doi_resolution': {'resolved': True}}
    assert checker.compare_entry(entry(), saved)['status'] == 'verified'


@pytest.mark.parametrize('registry,doi', [('crossref', '10.1234/example'), ('datacite', '10.48550/arxiv.example')])
def test_live_fetch_caches_selected_fields_and_resolution(monkeypatch, registry, doi):
    class Response:
        status_code = 302
        headers = {'Location': 'https://example.org/paper'}

        def json(self):
            metadata = {'DOI': doi, 'doi': doi, 'abstract': 'Do not retain',
                        'descriptions': [{'description': 'Do not retain'}],
                        'reference': [{'unstructured': 'Do not retain'}]}
            return {'message': metadata, 'data': {'attributes': metadata}}

    monkeypatch.setattr(checker, 'request', lambda *args, **kwargs: Response())
    monkeypatch.setattr(checker.time, 'sleep', lambda *args: None)
    saved = checker.fetch_evidence(None, doi)
    assert saved['registry'] == registry
    assert saved['metadata'].get('DOI', saved['metadata'].get('doi')) == doi
    assert not {'abstract', 'descriptions', 'reference'} & saved['metadata'].keys()
    assert saved['registry_url'].startswith('https://api.')
    assert saved['checked_at']
    assert saved['doi_resolution'] == {'resolved': True, 'status_code': 302, 'destination': 'https://example.org/paper'}

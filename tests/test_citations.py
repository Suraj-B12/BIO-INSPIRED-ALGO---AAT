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

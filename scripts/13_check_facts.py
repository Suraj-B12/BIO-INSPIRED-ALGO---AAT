"""Fail closed on the department's locked names, course, year and deliverables."""
from pathlib import Path
import argparse
import json
import re
import sys
ROOT=Path(__file__).resolve().parents[1]
TEAM=[('Aryanraj S Singh','1BM23CD008'),('Richik Chaudhuri','1BM23CD050'),('Suraj B','1BM23CD061'),('Sushant Deo','1BM23CD062')]
BANNED=['23DS6PWPP1','Shambhavi','1BM22','Sudeep','Harshavardhan','AY 2024-2025','AY 2024-25','2025-26','2026-27','Stock Price','LSTM','Indian Banking']
TITLES=['Cover Slide/Main Slide','Introduction to Optimization Problems','What are Nature-Inspired Algorithms?','Classification of Nature-Inspired Algorithms','Problem Statement','Implementation and Results','Performance Metrics and Benchmarking','Advantages and Limitations','Future Scope','References']


def normalized(text):return re.sub(r'\s+',' ',text.replace('\\&','&').replace('~',' ').replace('--','–'))

def inspect_content(name,value,errors):
    value=normalized(value)
    for bad in BANNED:
        if bad.lower() in value.lower():errors.append(f'{name}: forbidden fact {bad!r}')
    if re.search(r'Manoj\s+Kumar(?!\s+S\b)',value):errors.append(f'{name}: guide name must include S')
    for placeholder in ('<<','??','[?]'):
        if placeholder in value:errors.append(f'{name}: unresolved placeholder {placeholder}')


def check_identity(name,text,errors,certificate=False):
    text=normalized(text)
    required=['23DS7BSBIO','Manoj Kumar S','Assistant Professor']
    if certificate:required+=['Dr. Shyamala G','Professor & HOD']
    else:required+=['2026–2027']
    for token in required:
        if token not in text:errors.append(f'{name}: missing {token}')
    positions=[]
    for student,usn in TEAM:
        if student not in text or usn not in text:errors.append(f'{name}: missing exact name/USN {student}/{usn}')
        positions.append(text.find(student))
    if positions!=sorted(positions):errors.append(f'{name}: team order differs from locked order')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--source-only',action='store_true');args=parser.parse_args();errors=[];checked=[]
    paths=[ROOT/'README.md',*sorted((ROOT/'report').rglob('*.tex'))]
    for path in paths:
        if not path.exists():errors.append(f'Missing {path.relative_to(ROOT)}');continue
        inspect_content(str(path.relative_to(ROOT)),path.read_text(encoding='utf-8'),errors);checked.append(str(path.relative_to(ROOT)))
    if not args.source_only:
        import pymupdf
        from pptx import Presentation
        report=ROOT/'report/BIA_AAT_Team10_Report.pdf';pptx=ROOT/'slides/BIA_AAT_Team10.pptx';slides_pdf=ROOT/'slides/BIA_AAT_Team10.pdf'
        for path in [report,pptx,slides_pdf]:
            if not path.exists():errors.append(f'Missing required deliverable {path.relative_to(ROOT)}')
        if report.exists():
            pdf=pymupdf.open(report);pages=[p.get_text() for p in pdf];checked.append(str(report.relative_to(ROOT)))
            inspect_content('report PDF','\n'.join(pages),errors)
            if len(pages)<2:errors.append('Report must include cover and certificate')
            else:
                check_identity('Report cover',pages[0],errors)
                check_identity('Report certificate',pages[1],errors,certificate=True)
            if not 30<=len(pages)<=45:errors.append(f'Report has {len(pages)} pages; expected30–45')
        if pptx.exists():
            prs=Presentation(pptx);texts=[];checked.append(str(pptx.relative_to(ROOT)))
            for i,slide in enumerate(prs.slides,1):
                value='\n'.join(shape.text for shape in slide.shapes if shape.has_text_frame)
                note=slide.notes_slide.notes_text_frame.text if slide.has_notes_slide else ''
                if not note.strip():errors.append(f'Slide{i}: missing speaker notes')
                inspect_content(f'Slide{i}',value+'\n'+note,errors);texts.append(value)
                for shape in slide.shapes:
                    if shape.left <0 or shape.top<0 or shape.left+shape.width>prs.slide_width+10000 or shape.top+shape.height>prs.slide_height+10000:errors.append(f'Slide{i}: shape outside canvas')
            check_identity('Title slide',texts[0],errors)
            order=[]
            for title in TITLES:
                found=next((i for i,t in enumerate(texts) if title in t),None)
                if found is None:errors.append(f'Slides missing required title: {title}')
                else:order.append(found)
            if order!=sorted(order):errors.append('Required slide titles out of order')
        if slides_pdf.exists():
            pdf=pymupdf.open(slides_pdf);text='\n'.join(p.get_text() for p in pdf);inspect_content('slide PDF',text,errors);checked.append(str(slides_pdf.relative_to(ROOT)))
            if pptx.exists() and len(pdf)!=len(prs.slides):errors.append('Slide PDF page count differs from deck')
    output={'status':'passed' if not errors else 'failed','scope':'source-only' if args.source_only else 'full','checked':checked,'errors':errors}
    (ROOT/'results/facts_check.json').write_text(json.dumps(output,indent=2)+'\n',encoding='utf-8')
    for error in errors:print('FAIL:',error)
    print(f'Locked facts: {output["status"]}; {len(checked)} artifacts checked')
    raise SystemExit(bool(errors))

if __name__=='__main__':main()

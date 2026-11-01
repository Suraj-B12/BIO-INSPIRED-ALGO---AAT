"""Build final report/slides PDFs, then run locked-fact and citation checks."""
from pathlib import Path
import argparse
import os
import shutil
import subprocess
import sys
import tempfile
ROOT=Path(__file__).resolve().parents[1]


def run(command,cwd=ROOT):
    print('Running',Path(command[0]).name,flush=True)
    result=subprocess.run(command,cwd=cwd,text=True,capture_output=True)
    (ROOT/'results').mkdir(exist_ok=True)
    log=ROOT/'results'/('export_'+Path(command[0]).stem+'.log');log.write_text(result.stdout+'\n'+result.stderr,encoding='utf-8')
    if result.returncode:
        print(result.stdout[-5000:]);print(result.stderr[-5000:]);raise SystemExit(result.returncode)
    return result


def export_slides(path):
    binary=shutil.which('soffice')
    if not binary:
        candidate=Path('/Applications/LibreOffice.app/Contents/MacOS/soffice')
        if candidate.exists():binary=str(candidate)
    if not binary:raise SystemExit('Install LibreOffice to export the presentation as PDF.')
    with tempfile.TemporaryDirectory(prefix='plantwise-office-') as profile:
        run([binary,f'-env:UserInstallation={Path(profile).as_uri()}','--headless','--convert-to','pdf','--outdir',str(path.parent),str(path)])
    if not path.with_suffix('.pdf').exists():raise SystemExit('LibreOffice did not create the expected PDF.')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--slides-only',action='store_true');parser.add_argument('--report-only',action='store_true');parser.add_argument('--offline-citations',action='store_true');args=parser.parse_args()
    if not args.slides_only:
        if shutil.which('tectonic'):run(['tectonic','--keep-logs','main.tex'],ROOT/'report')
        elif shutil.which('latexmk'):run(['latexmk','-pdf','-interaction=nonstopmode','main.tex'],ROOT/'report')
        else:raise SystemExit('Install Tectonic or a LaTeX distribution to compile the report.')
        source=ROOT/'report/main.pdf'
        if not source.exists():raise SystemExit('Report compiler did not produce main.pdf')
        shutil.copy2(source,ROOT/'report/BIA_AAT_Team10_Report.pdf')
    if not args.report_only:export_slides(ROOT/'slides/BIA_AAT_Team10.pptx')
    run([sys.executable,str(ROOT/'scripts/13_check_facts.py')])
    command=[sys.executable,str(ROOT/'scripts/14_check_citations.py')]
    if args.offline_citations:command.append('--offline')
    run(command)
    print('PDF exports and locked-fact/citation gates passed.')

if __name__=='__main__':main()

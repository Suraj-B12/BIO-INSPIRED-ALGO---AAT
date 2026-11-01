"""Cross-platform entry point; resumes completed runs without changing protocols."""
from pathlib import Path
import argparse
import subprocess
import sys
from bioalgo.common import ROOT,read_json,save_json,timestamp


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--from',dest='first',default='00');parser.add_argument('--to',dest='last',default='09')
    parser.add_argument('--smoke',action='store_true');parser.add_argument('--quick',action='store_true',help='Alias for the synthetic offline smoke check')
    args=parser.parse_args()
    if args.smoke or args.quick:
        from bioalgo.smoke import run_smoke
        result=run_smoke();save_json('results/smoke.json',result);print(f"Synthetic pipeline check passed in {result['seconds']:.1f}s; no dataset or pretrained download required.");return
    first,last=int(args.first),int(args.last)
    if not 0<=first<=last<=14:parser.error('Stage range must be between 00 and 14')
    for number in range(first,last+1):
        matches=sorted((ROOT/'scripts').glob(f'{number:02d}_*.py'))
        if len(matches)!=1:raise FileNotFoundError(f'Expected one script for stage {number:02d}')
        if number==0 and read_json('results/system_info.json'):
            print('Using saved hardware benchmark. Run scripts/00_system_check.py directly to remeasure.');continue
        command=[sys.executable,str(matches[0])]
        if number==1:command.append('--download')
        print(f'Running {matches[0].name}',flush=True)
        subprocess.run(command,cwd=ROOT,check=True)
        status=read_json('results/pipeline_status.json',{});status[f'{number:02d}']={'status':'complete','timestamp':timestamp()};save_json('results/pipeline_status.json',status)

if __name__=='__main__':main()

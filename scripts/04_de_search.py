"""Run the resumable seven-gene DE search using validation macro-F1 only."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from bioalgo.common import ROOT,load_config,save_json
from bioalgo.de import optimize
from bioalgo.search import setup_search,rerank,cnn_de_controls
from bioalgo.training import train_recipe
import argparse


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--fresh',action='store_true');parser.add_argument('--skip-rerank',action='store_true');args=parser.parse_args()
    config=load_config();controls=cnn_de_controls(config);objective,callback,tier=setup_search(config,'de',args.fresh)
    baseline_proxy=train_recipe(config,config['baseline'],'baseline_proxy',int(config.get('split_seed',42)),resolution=tier['resolution'],fraction=tier['fraction'],epochs=tier['epochs'],save_model=False)
    save_json('results/baseline_proxy.json',baseline_proxy)
    result=optimize(objective,dimensions=7,population_size=tier['population'],generations=tier['generations'],seed=config['de']['seed'],**controls,state_path=ROOT/'results'/'de_state.json',resume=not args.fresh,callback=callback,record_positions=True)
    for event in result.get('evaluation_events',[]): callback(event)
    save_json('results/de_history.json',result['history']);save_json('results/de_search.json',result)
    if not args.skip_rerank: rerank(config,'de')

if __name__=='__main__': main()

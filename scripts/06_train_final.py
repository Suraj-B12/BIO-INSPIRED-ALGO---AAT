"""Freeze both winning recipes, then retrain each over every preregistered seed."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from bioalgo.common import load_config,read_json,save_json,timestamp
from bioalgo.training import train_recipe,fingerprint


def main():
    config=load_config();recipes={'baseline':config['baseline']}
    for method in ('de','rs'):
        result=read_json(f'results/{method}_rerank.json')
        if result is None:raise FileNotFoundError(f'Complete {method.upper()} search and re-ranking before final training')
        recipes[method]=result['best_config']
    frozen={'recipes':recipes,'training':config['training'],'seeds':config['seeds'],'timestamp':timestamp()}
    frozen['signature']=fingerprint({k:v for k,v in frozen.items() if k!='timestamp'})
    previous=read_json('results/frozen_protocol.json')
    if previous and previous['signature']!=frozen['signature']:raise RuntimeError('Final protocol changed after freezing; archive prior experiment before continuing')
    if previous is None:save_json('results/frozen_protocol.json',frozen)
    for method in ('de','rs'):
        for seed in config['seeds']:train_recipe(config,recipes[method],method,seed)

if __name__=='__main__':main()

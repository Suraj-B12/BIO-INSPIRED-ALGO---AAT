"""Independent uniform search with exactly the same candidate budget as DE."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from bioalgo.common import ROOT,load_config,save_json,read_json
from bioalgo.search import setup_search,rerank
import argparse
import numpy as np


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--fresh',action='store_true');parser.add_argument('--skip-rerank',action='store_true');args=parser.parse_args()
    config=load_config();objective,callback,tier=setup_search(config,'rs',args.fresh)
    budget=tier['population']*(tier['generations']+1);rng=np.random.default_rng(config['de']['seed']+1)
    state=read_json('results/rs_state.json') if not args.fresh else None
    if state:
        if state['budget']!=budget: raise ValueError('Random search budget changed')
        rng.bit_generator.state=state['rng_state']
    else: state={'evaluations':[],'budget':budget,'best_fitness':-1.,'best_vector':None,'history':[]}
    for index in range(len(state['evaluations']),budget):
        genes=rng.random(7);fitness=objective(genes);event={'evaluation':index+1,'generation':index//tier['population'],'individual':index%tier['population'],'genes':genes.tolist(),'fitness':fitness}
        callback(event);state['evaluations'].append(event)
        if fitness>=state['best_fitness']:state['best_fitness']=fitness;state['best_vector']=genes.tolist()
        state['history'].append({'evaluation':index+1,'best':state['best_fitness']});state['rng_state']=rng.bit_generator.state
        save_json('results/rs_state.json',state)
    for event in state['evaluations']:callback(event)
    save_json('results/rs_history.json',state['history']);save_json('results/rs_search.json',state)
    if not args.skip_rerank:rerank(config,'rs')

if __name__=='__main__': main()

"""Train the immutable preregistered baseline over all declared seeds."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from bioalgo.common import load_config,save_json,read_json
from bioalgo.training import train_recipe


def main():
    config=load_config()
    expected={'learning_rate':.001,'weight_decay':0.,'dropout':.2,'head_units':256,'unfreeze_blocks':0,'label_smoothing':0.,'augmentation':0,'optimizer':'adam'}
    if config['baseline'] != expected: raise ValueError('Baseline differs from preregistration in AGENTS.md')
    registration={'config':expected,'training':config['training'],'seeds':config['seeds']}
    previous=read_json('results/baseline_preregistration.json')
    if previous is not None and previous!=registration:raise RuntimeError('Baseline preregistration is immutable; archive the experiment before changing settings')
    if previous is None:save_json('results/baseline_preregistration.json',registration)
    for seed in config['seeds']: train_recipe(config,expected,'baseline',seed)

if __name__=='__main__': main()

"""Measure image loading choices without changing the preregistered protocol."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from bioalgo.common import load_config,save_json,seed_everything
from bioalgo.data import load_rows,make_loader,image_path,evaluation_transform
from PIL import Image
import copy
import time
import numpy as np


def main():
    config=load_config();rows=load_rows('train',config)[:640];seed_everything(42);records=[]
    for workers in (0,2,4):
        settings=copy.deepcopy(config);settings['training']['num_workers']=workers
        started=time.perf_counter();loader=make_loader(rows,settings,128);count=0
        for images,labels in loader:count+=len(labels)
        records.append({'kind':'cached_jpeg_dataloader','num_workers':workers,'images':count,'seconds_including_startup':time.perf_counter()-started})
    subset=rows[:128];decoded=[];transform=evaluation_transform(128)
    for row in subset:
        with Image.open(image_path(row,config)) as image:decoded.append(np.asarray(image.convert('RGB')).copy())
    for source in ('raw_jpeg','cached_jpeg','decoded_uint8_in_memory'):
        started=time.perf_counter()
        for index,row in enumerate(subset):
            if source=='decoded_uint8_in_memory':image=Image.fromarray(decoded[index]);transform(image)
            else:
                with Image.open(image_path(row,config,cached=source=='cached_jpeg')) as image:transform(image.convert('RGB'))
        records.append({'kind':source,'images':len(subset),'seconds':time.perf_counter()-started})
    save_json('results/data_loading_benchmark.json',{'records':records,'chosen':'cached_jpeg_dataloader','num_workers':config['training']['num_workers'],'note':'Measured end-to-end loader startup and a small deterministic decode sample. Retained the preregistered zero-worker JPEG loader for portable, repeatable augmentation and bounded memory; short-run timings are not a full-epoch throughput claim.'})
    print(records)

if __name__=='__main__':main()

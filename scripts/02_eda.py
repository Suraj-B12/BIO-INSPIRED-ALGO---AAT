"""Reproducible descriptive figures using training images only."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from bioalgo.common import ROOT,load_config,read_json,seed_everything,save_json
from bioalgo.data import load_rows,image_path,training_transform,MEAN,STD,display_crop
from bioalgo.model import build_model,model_summary
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def save(fig,name):
    fig.tight_layout();fig.savefig(ROOT/'figures'/f'{name}.png',dpi=300,bbox_inches='tight');fig.savefig(ROOT/'figures'/f'{name}.pdf',bbox_inches='tight');plt.close(fig)


def main():
    config=load_config();summary=read_json('results/data_summary.json');classes=read_json('results/class_names.json');rows=load_rows('train',config);seed_everything(42)
    if summary is None:raise FileNotFoundError('Prepare the dataset first')
    style=config.get('style',{});plt.rcParams.update({'font.family':style.get('font','DejaVu Sans'),'axes.spines.top':False,'axes.spines.right':False,'font.size':11})
    fig,axis=plt.subplots(figsize=(11,6));names=[name.replace('Tomato___','').replace('_',' ') for name in classes];counts=[summary['original_counts'][name] for name in classes]
    bars=axis.barh(names,counts,color=style.get('de','#147D55'));axis.invert_yaxis();axis.bar_label(bars,padding=4);axis.set_xlim(0,max(counts)*1.16);axis.set_xlabel('Images in original color dataset');axis.set_title('Tomato classes are strongly imbalanced',loc='left',fontweight='bold');save(fig,'fig_class_distribution')
    fig,axes=plt.subplots(10,3,figsize=(9,24));sample_pixels=[]
    for label,name in enumerate(classes):
        subset=[row for row in rows if row['label']==label][:3]
        for col,row in enumerate(subset):
            with Image.open(image_path(row,config)) as opened:
                pixels=np.asarray(display_crop(opened,224));sample_pixels.append(pixels);axes[label,col].imshow(pixels)
            axes[label,col].axis('off')
            if col==0:axes[label,col].set_title(names[label],fontsize=9,loc='left')
    save(fig,'fig_dataset_samples')
    with Image.open(image_path(rows[0],config)) as opened:image=opened.convert('RGB')
    fig,axes=plt.subplots(1,4,figsize=(12,3.5));augmentation_pixels=[np.asarray(display_crop(image,224))];axes[0].imshow(augmentation_pixels[0]);axes[0].set_title('Original training image')
    for index,magnitude in enumerate((0,7,15),start=1):
        tensor=training_transform(224,magnitude)(image);array=tensor.permute(1,2,0).numpy()*np.array(STD)+np.array(MEAN);augmentation_pixels.append(np.round(np.clip(array,0,1)*255).astype(np.uint8));axes[index].imshow(augmentation_pixels[-1]);axes[index].set_title('Flips + crop' if magnitude==0 else f'RandAugment magnitude {magnitude}')
    for axis in axes:axis.axis('off')
    save(fig,'fig_augmentation_examples')
    np.savez_compressed(ROOT/'results'/'eda_assets.npz',sample_images=np.asarray(sample_pixels).reshape(10,3,224,224,3),augmentation_images=np.asarray(augmentation_pixels))
    model=build_model(config['baseline'],pretrained=False);save_json('results/model_summary.json',model_summary(model))
    print('Created training-only EDA and model-summary artifacts.')

if __name__=='__main__':main()

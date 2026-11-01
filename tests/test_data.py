import csv
import numpy as np
import pytest
from PIL import Image
from bioalgo.common import ROOT,load_config
from bioalgo.data import group_hashes,stratified_group_split,subset_rows,corrupt_image,CORRUPTIONS,evaluation_transform,training_transform,load_rows


def test_perceptual_groups_are_transitive():
    values=[0,15,255,2**63]
    groups=group_hashes(values,radius=4)
    assert groups[0]==groups[1]==groups[2]==groups[3]


@pytest.mark.parametrize('seed',range(30))
def test_bk_tree_matches_brute_force(seed):
    rng=np.random.default_rng(seed);values=[int(v) for v in rng.integers(0,2**20,80)]
    groups=group_hashes(values,radius=4)
    for i in range(len(values)):
        for j in range(i):
            if (values[i]^values[j]).bit_count()<=4:assert groups[i]==groups[j]


def test_grouped_stratification_keeps_groups_and_proportions():
    rows=[{'path':f'{label}/{i}.jpg','label':label,'group':f'{label}-{i//2}'} for label in range(10) for i in range(200)]
    splits=stratified_group_split(rows)
    group_sets=[{r['group'] for r in values} for values in splits.values()]
    assert not group_sets[0]&group_sets[1] and not group_sets[0]&group_sets[2] and not group_sets[1]&group_sets[2]
    for ratio,values in zip((.7,.15,.15),splits.values()):
        for label in range(10):assert abs(sum(r['label']==label for r in values)/200-ratio)<=.01
    assert splits==stratified_group_split(rows)


def test_committed_manifests_no_leakage():
    directory=ROOT/'data'/'splits'
    if not (directory/'train.csv').exists():pytest.skip('Real data preparation has not run')
    splits=[]
    for name in ('train','val','test'):
        with (directory/f'{name}.csv').open(encoding='utf-8',newline='') as stream:splits.append(list(csv.DictReader(stream)))
    for key in ('path','sha256','group','phash'):
        sets=[{r[key] for r in values} for values in splits]
        assert not sets[0]&sets[1] and not sets[0]&sets[2] and not sets[1]&sets[2],key
    for label in range(10):
        total=sum(sum(int(r['label'])==label for r in values) for values in splits)
        for ratio,values in zip((.7,.15,.15),splits):assert abs(sum(int(r['label'])==label for r in values)/total-ratio)<=.01


def test_test_split_is_guarded():
    with pytest.raises(ValueError,match='held-out'):load_rows('test',{})


@pytest.mark.parametrize('kind',CORRUPTIONS)
@pytest.mark.parametrize('severity',[1,2,3])
@pytest.mark.parametrize('size',[(1,1),(32,48),(256,256)])
def test_corruptions_are_bounded_repeatable(kind,severity,size):
    image=Image.fromarray(np.full((size[1],size[0],3),127,dtype=np.uint8))
    first=corrupt_image(image,kind,severity,seed=7);second=corrupt_image(image,kind,severity,seed=7)
    assert first.size==size and first.mode=='RGB';assert np.array_equal(first,second)
    assert np.asarray(first).min()>=0 and np.asarray(first).max()<=255


@pytest.mark.parametrize('resolution',[32,112,128,160,224])
def test_transforms_have_valid_shapes(resolution):
    image=Image.new('RGB',(320,256),(125,140,85))
    assert evaluation_transform(resolution)(image).shape==(3,resolution,resolution)
    assert training_transform(resolution,15)(image).shape==(3,resolution,resolution)


def test_stratified_subset_seeded():
    rows=[{'path':str(i),'label':i%10} for i in range(1000)]
    subset=subset_rows(rows,.1,42)
    assert len(subset)==100 and subset==subset_rows(rows,.1,42)
    assert all(sum(r['label']==label for r in subset)==10 for label in range(10))

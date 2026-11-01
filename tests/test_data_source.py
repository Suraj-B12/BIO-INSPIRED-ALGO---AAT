from pathlib import Path
import runpy
from unittest.mock import Mock

module=runpy.run_path(str(Path(__file__).resolve().parents[1]/'scripts'/'01_prepare_data.py'))


def test_download_checks_out_recorded_commit_before_listing_images(tmp_path,monkeypatch):
    repository=tmp_path/'PlantVillage';(repository/'.git').mkdir(parents=True)
    raw=repository/'raw'/'color';calls=[];revision=module['DATASET_REVISION'];heads=iter(['0'*40,revision])
    def output(command,**kwargs):
        calls.append(command)
        if command[1]=='rev-parse':return next(heads)+'\n'
        return '\n'.join(f'Tomato___class_{i}' for i in range(10))
    def run(command,**kwargs):calls.append(command)
    monkeypatch.setattr(module['subprocess'],'check_output',output)
    monkeypatch.setattr(module['subprocess'],'run',run)
    module['download'](raw)
    checkout=['git','checkout','--detach',revision]
    assert checkout in calls
    assert ['git','fetch','--depth','1','origin',revision] in calls
    assert calls.index(checkout)<calls.index(['git','ls-tree','--name-only','HEAD:raw/color'])
    assert calls[-1][0:3]==['git','sparse-checkout','set']


def test_pinned_existing_checkout_does_not_switch_revision(tmp_path,monkeypatch):
    repository=tmp_path/'PlantVillage';(repository/'.git').mkdir(parents=True)
    calls=[];revision=module['DATASET_REVISION']
    def output(command,**kwargs):
        if command[1]=='rev-parse':return revision+'\n'
        return '\n'.join(f'Tomato___class_{i}' for i in range(10))
    monkeypatch.setattr(module['subprocess'],'check_output',output)
    monkeypatch.setattr(module['subprocess'],'run',lambda command,**kwargs:calls.append(command))
    module['download'](repository/'raw'/'color')
    assert not any(command[1] in ('fetch','checkout') for command in calls)

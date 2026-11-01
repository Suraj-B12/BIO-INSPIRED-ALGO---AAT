import copy
from pathlib import Path
import runpy
import numpy as np
import pytest
from bioalgo.evaluation import classification_metrics

module=runpy.run_path(str(Path(__file__).resolve().parents[1]/'scripts'/'validate_results.py'))


def test_saved_prediction_validator_rejects_altered_metric():
    labels=np.tile(np.arange(10),3);probabilities=np.eye(10)[labels];classes=[str(i) for i in range(10)]
    metrics=classification_metrics(labels,probabilities,classes)
    module['compare_predictions'](metrics,labels,probabilities,classes)
    changed=copy.deepcopy(metrics);changed['macro_f1']=.999
    with pytest.raises(AssertionError,match='macro_f1'):module['compare_predictions'](changed,labels,probabilities,classes)


def test_saved_prediction_validator_rejects_nonfinite():
    with pytest.raises(AssertionError,match='Non-finite'):module['require_finite']({'models':{'accuracy':float('nan')}})

import numpy as np
import pytest
from bioalgo.evaluation import classification_metrics,calibration,bootstrap_ci


def test_perfect_predictions_and_calibration():
    labels=np.tile(np.arange(10),5);prob=np.eye(10)[labels]
    metrics=classification_metrics(labels,prob,[str(i) for i in range(10)])
    assert metrics['accuracy']==metrics['macro_f1']==metrics['roc_auc']==metrics['top3_accuracy']==1
    assert metrics['calibration']['ece']==0
    assert bootstrap_ci(labels,prob,samples=50)['accuracy']==[1,1]


def test_uniform_predictions_ece():
    labels=np.tile(np.arange(10),10);prob=np.full((100,10),.1)
    result=calibration(labels,prob)
    assert result['ece']==pytest.approx(0)
    assert sum(row['count'] for row in result['bins'])==100


def test_invalid_probability_guard():
    with pytest.raises(ValueError,match='Invalid probabilities'):classification_metrics(np.arange(10),np.ones((10,10)),[str(i) for i in range(10)])

from bioalgo.smoke import run_smoke


def test_end_to_end_offline():
    result=run_smoke()
    assert result['status']=='passed'
    assert result['seconds']<600
    assert result['checkpoint_round_trip_exact'] and result['finite_loss']

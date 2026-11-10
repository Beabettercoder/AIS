from anomaly_workbench.evidence_snapshot import evidence_signature,unavailable_diagnostics


def test_signature_checks_original_observations_and_neighbors():
    first=evidence_signature({'sog':[0,8],'xy':[[1,2],[3,4]]},[{'mmsi':'A','distance':1}])
    assert first==evidence_signature({'xy':[[1,2],[3,4]],'sog':[0,8]},[{'distance':1,'mmsi':'A'}])
    assert first!=evidence_signature({'sog':[0,9],'xy':[[1,2],[3,4]]},[{'mmsi':'A','distance':1}])
    assert first!=evidence_signature({'sog':[0,8],'xy':[[1,2],[3,4]]},[{'mmsi':'B','distance':1}])


def test_missing_or_nonfinite_snapshot_does_not_claim_verified_model():
    assert evidence_signature(None,[]) is None
    assert evidence_signature({'sog':[float('nan')]},[]) is None
    original={'gate_weights':[1.,.2],'time_seconds':[0,60]}
    output=unavailable_diagnostics(original,'unavailable')
    assert output['pg_probability']==[] and output['raw_probability']==[]
    assert output['gate_weights']==original['gate_weights']

from __future__ import annotations

from fastapi.testclient import TestClient

from app.config import CONTRACT_PATH, JOBLIB_PATH, METADATA_PATH
from app.model.contract import load_contract
from app.model.loader import load_runtime
from app.services.predictor import predict_record


def _valid_record(contract) -> dict:
    record: dict = {}
    for name in contract.feature_columns_in_order:
        if name == "protocol_type":
            record[name] = "tcp"
        elif name == "service":
            record[name] = "http"
        elif name == "flag":
            record[name] = "SF"
        else:
            record[name] = 0.0
    record["src_bytes"] = 215.0
    record["dst_bytes"] = 45076.0
    record["logged_in"] = 1.0
    record["count"] = 1.0
    record["srv_count"] = 1.0
    record["same_srv_rate"] = 1.0
    record["dst_host_count"] = 1.0
    record["dst_host_srv_count"] = 1.0
    record["dst_host_same_srv_rate"] = 1.0
    return record


def test_predictor_uses_locked_artifact():
    runtime = load_runtime(JOBLIB_PATH, METADATA_PATH, CONTRACT_PATH)
    result = predict_record(runtime.pipeline, runtime.contract, _valid_record(runtime.contract))

    assert result.predicted_class in {"Normal", "DoS", "Probe", "R2L", "U2R"}
    assert 0.0 <= result.confidence <= 1.0

    classifier = runtime.pipeline.named_steps["classifier"]
    mapped = result.class_probabilities[result.predicted_class]
    assert mapped == result.confidence
    assert set(result.class_probabilities) == set(map(str, classifier.classes_))


def test_predictor_does_not_require_non_features():
    contract = load_contract(CONTRACT_PATH)
    record = _valid_record(contract)
    assert "label" not in record
    assert "difficulty" not in record
    assert "true_label" not in record

    runtime = load_runtime(JOBLIB_PATH, METADATA_PATH, CONTRACT_PATH)
    result = predict_record(runtime.pipeline, runtime.contract, record)
    assert result.predicted_class in {"Normal", "DoS", "Probe", "R2L", "U2R"}


def test_predict_endpoint_rejects_label_fields():
    from app.main import app

    contract = load_contract(CONTRACT_PATH)
    payload = _valid_record(contract)
    payload["label"] = "normal"
    payload["difficulty"] = 20
    payload["true_label"] = "Normal"

    with TestClient(app) as client:
        response = client.post("/predict", json=payload)

    assert response.status_code == 422

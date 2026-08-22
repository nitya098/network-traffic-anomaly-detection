from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from app.config import CONTRACT_PATH, JOBLIB_PATH, MAX_BATCH_SIZE, METADATA_PATH
from app.model.contract import load_contract
from app.model.loader import load_runtime
from app.services.predictor import predict_batch, predict_record

ALLOWED_CLASSES = {"Normal", "DoS", "Probe", "R2L", "U2R"}


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


def _probe_like_record(contract) -> dict:
    record = _valid_record(contract)
    record["duration"] = 0.0
    record["protocol_type"] = "icmp"
    record["service"] = "eco_i"
    record["flag"] = "SF"
    record["src_bytes"] = 8.0
    record["dst_bytes"] = 0.0
    record["logged_in"] = 0.0
    record["count"] = 1.0
    record["srv_count"] = 12.0
    return record


@pytest.fixture(scope="module")
def runtime():
    return load_runtime(JOBLIB_PATH, METADATA_PATH, CONTRACT_PATH)


@pytest.fixture(scope="module")
def client():
    from app.main import app

    with TestClient(app) as test_client:
        yield test_client


def test_predictor_uses_locked_artifact(runtime):
    result = predict_record(runtime.pipeline, runtime.contract, _valid_record(runtime.contract))

    assert result.predicted_class in ALLOWED_CLASSES
    assert 0.0 <= result.confidence <= 1.0

    classifier = runtime.pipeline.named_steps["classifier"]
    mapped = result.class_probabilities[result.predicted_class]
    assert mapped == result.confidence
    assert set(result.class_probabilities) == set(map(str, classifier.classes_))


def test_predictor_does_not_require_non_features(runtime):
    contract = load_contract(CONTRACT_PATH)
    record = _valid_record(contract)
    assert "label" not in record
    assert "difficulty" not in record
    assert "true_label" not in record

    result = predict_record(runtime.pipeline, runtime.contract, record)
    assert result.predicted_class in ALLOWED_CLASSES


def test_predict_batch_one_record(runtime):
    records = [_valid_record(runtime.contract)]
    batch = predict_batch(runtime.pipeline, runtime.contract, records)
    assert len(batch.predictions) == 1
    assert batch.predictions[0].predicted_class in ALLOWED_CLASSES
    assert 0.0 <= batch.predictions[0].confidence <= 1.0


def test_predict_batch_multiple_records(runtime):
    records = [
        _valid_record(runtime.contract),
        _probe_like_record(runtime.contract),
        _valid_record(runtime.contract),
    ]
    batch = predict_batch(runtime.pipeline, runtime.contract, records)
    assert len(batch.predictions) == len(records)
    for item in batch.predictions:
        assert item.predicted_class in ALLOWED_CLASSES
        assert 0.0 <= item.confidence <= 1.0


def test_predict_batch_calls_pipeline_once(runtime):
    records = [
        _valid_record(runtime.contract),
        _probe_like_record(runtime.contract),
        _valid_record(runtime.contract),
    ]
    real_predict = runtime.pipeline.predict
    real_proba = runtime.pipeline.predict_proba
    runtime.pipeline.predict = MagicMock(side_effect=real_predict)
    runtime.pipeline.predict_proba = MagicMock(side_effect=real_proba)
    try:
        batch = predict_batch(runtime.pipeline, runtime.contract, records)
        assert runtime.pipeline.predict.call_count == 1
        assert runtime.pipeline.predict_proba.call_count == 1
        assert len(batch.predictions) == 3
    finally:
        runtime.pipeline.predict = real_predict
        runtime.pipeline.predict_proba = real_proba


def test_health_and_model_info(client):
    health = client.get("/health")
    assert health.status_code == 200
    body = health.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True

    info = client.get("/model/info")
    assert info.status_code == 200
    payload = info.json()
    assert payload["n_estimators"] == 500
    assert payload["random_state"] == 42
    assert payload["artifact_filename"] == "nsl_kdd_random_forest_500.joblib"
    assert payload["max_batch_size"] == MAX_BATCH_SIZE


def test_predict_endpoint_still_works(client, runtime):
    response = client.post("/predict", json=_valid_record(runtime.contract))
    assert response.status_code == 200
    body = response.json()
    assert body["predicted_class"] in ALLOWED_CLASSES
    assert 0.0 <= body["confidence"] <= 1.0
    assert body["model_version"] == "nsl_kdd_random_forest_500"
    assert "latency_ms" in body


def test_predict_endpoint_rejects_label_fields(client, runtime):
    payload = _valid_record(runtime.contract)
    payload["label"] = "normal"
    payload["difficulty"] = 20
    payload["true_label"] = "Normal"
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_batch_endpoint_one_and_many(client, runtime):
    one = client.post("/predict/batch", json={"records": [_valid_record(runtime.contract)]})
    assert one.status_code == 200
    one_body = one.json()
    assert one_body["n_records"] == 1
    assert len(one_body["predictions"]) == 1
    assert one_body["model_version"] == "nsl_kdd_random_forest_500"

    records = [_valid_record(runtime.contract), _probe_like_record(runtime.contract)]
    many = client.post("/predict/batch", json={"records": records})
    assert many.status_code == 200
    many_body = many.json()
    assert many_body["n_records"] == 2
    assert len(many_body["predictions"]) == 2
    for item in many_body["predictions"]:
        assert item["predicted_class"] in ALLOWED_CLASSES
        assert 0.0 <= item["confidence"] <= 1.0


def test_batch_empty_rejected(client):
    response = client.post("/predict/batch", json={"records": []})
    assert response.status_code in {400, 422}


def test_batch_over_max_rejected(client, runtime):
    records = [_valid_record(runtime.contract) for _ in range(MAX_BATCH_SIZE + 1)]
    response = client.post("/predict/batch", json={"records": records})
    assert response.status_code == 413


@pytest.mark.parametrize("field,value", [("label", "normal"), ("difficulty", 20), ("true_label", "Normal"), ("unexpected", 1)])
def test_batch_rejects_non_features(client, runtime, field, value):
    record = _valid_record(runtime.contract)
    record[field] = value
    response = client.post("/predict/batch", json={"records": [record]})
    assert response.status_code == 422

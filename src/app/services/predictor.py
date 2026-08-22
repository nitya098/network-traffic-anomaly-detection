from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import pandas as pd
from sklearn.pipeline import Pipeline

from app.model.contract import InputContract

RecordMapping = Mapping[str, Any]


@dataclass(frozen=True)
class PredictionResult:
    predicted_class: str
    confidence: float
    class_probabilities: dict[str, float]
    latency_ms: float


@dataclass(frozen=True)
class BatchInferenceResult:
    predictions: list[PredictionResult]
    latency_ms: float


def _classifier(pipeline: Pipeline):
    if "classifier" in pipeline.named_steps:
        return pipeline.named_steps["classifier"]
    return pipeline.steps[-1][1]


def _records_to_frame(
    contract: InputContract,
    records: Sequence[RecordMapping],
) -> pd.DataFrame:
    columns = list(contract.feature_columns_in_order)
    rows = [{name: record[name] for name in columns} for record in records]
    return pd.DataFrame(rows, columns=columns)


def _row_prediction(
    predicted_label: Any,
    probability_row: Any,
    class_names: Any,
    latency_ms: float,
) -> PredictionResult:
    predicted_class = str(predicted_label)
    # Map each probability to the fitted class name. Do not assume index 0 is Normal.
    # predict_proba() is treated as Random Forest model confidence, not a calibrated
    # probability of correctness.
    class_probabilities = {
        str(class_name): float(prob)
        for class_name, prob in zip(class_names, probability_row)
    }
    confidence = class_probabilities[predicted_class]
    return PredictionResult(
        predicted_class=predicted_class,
        confidence=confidence,
        class_probabilities=class_probabilities,
        latency_ms=latency_ms,
    )


def _run_pipeline_inference(pipeline: Pipeline, frame: pd.DataFrame):
    started = time.perf_counter()
    predicted = pipeline.predict(frame)
    probabilities = pipeline.predict_proba(frame)
    latency_ms = (time.perf_counter() - started) * 1000.0
    return predicted, probabilities, latency_ms


def predict_batch(
    pipeline: Pipeline,
    contract: InputContract,
    records: Sequence[RecordMapping],
) -> BatchInferenceResult:
    if len(records) == 0:
        raise ValueError("Batch must contain at least one record.")

    frame = _records_to_frame(contract, records)
    predicted, probabilities, latency_ms = _run_pipeline_inference(pipeline, frame)
    class_names = _classifier(pipeline).classes_

    predictions = [
        _row_prediction(label, proba_row, class_names, latency_ms)
        for label, proba_row in zip(predicted, probabilities)
    ]
    return BatchInferenceResult(predictions=predictions, latency_ms=latency_ms)


def predict_one(
    pipeline: Pipeline,
    contract: InputContract,
    record: RecordMapping,
) -> PredictionResult:
    batch = predict_batch(pipeline, contract, [record])
    return batch.predictions[0]


# Phase 1 name kept so existing imports continue to work.
predict_record = predict_one

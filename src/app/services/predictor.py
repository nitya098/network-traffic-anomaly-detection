from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Mapping

import pandas as pd
from sklearn.pipeline import Pipeline

from app.model.contract import InputContract


@dataclass(frozen=True)
class PredictionResult:
    predicted_class: str
    confidence: float
    class_probabilities: dict[str, float]
    latency_ms: float


def _classifier(pipeline: Pipeline):
    if "classifier" in pipeline.named_steps:
        return pipeline.named_steps["classifier"]
    return pipeline.steps[-1][1]


def predict_record(
    pipeline: Pipeline,
    contract: InputContract,
    record: Mapping[str, Any],
) -> PredictionResult:
    ordered = {name: record[name] for name in contract.feature_columns_in_order}
    frame = pd.DataFrame([ordered], columns=list(contract.feature_columns_in_order))

    started = time.perf_counter()
    predicted = pipeline.predict(frame)
    probabilities = pipeline.predict_proba(frame)
    latency_ms = (time.perf_counter() - started) * 1000.0

    predicted_class = str(predicted[0])
    classifier = _classifier(pipeline)
    # Map each probability to the fitted class name. Do not assume index 0 is Normal.
    # predict_proba() is treated as Random Forest model confidence, not a calibrated
    # probability of correctness.
    class_probabilities = {
        str(class_name): float(prob)
        for class_name, prob in zip(classifier.classes_, probabilities[0])
    }
    confidence = class_probabilities[predicted_class]

    return PredictionResult(
        predicted_class=predicted_class,
        confidence=confidence,
        class_probabilities=class_probabilities,
        latency_ms=latency_ms,
    )

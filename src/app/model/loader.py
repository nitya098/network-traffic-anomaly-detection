from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
from sklearn.pipeline import Pipeline

from app.model.contract import InputContract, load_contract


@dataclass
class LoadedModel:
    pipeline: Pipeline
    metadata: dict[str, Any]
    contract: InputContract


def load_metadata(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def load_locked_pipeline(joblib_path: Path) -> Pipeline:
    if not joblib_path.is_file():
        raise FileNotFoundError("Locked model artifact was not found.")
    pipeline = joblib.load(joblib_path)
    if not isinstance(pipeline, Pipeline):
        raise TypeError("Locked artifact is not a sklearn Pipeline.")
    return pipeline


def load_runtime(
    joblib_path: Path,
    metadata_path: Path,
    contract_path: Path,
) -> LoadedModel:
    pipeline = load_locked_pipeline(joblib_path)
    metadata = load_metadata(metadata_path)
    contract = load_contract(contract_path)
    return LoadedModel(pipeline=pipeline, metadata=metadata, contract=contract)

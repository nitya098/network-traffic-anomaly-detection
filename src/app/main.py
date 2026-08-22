from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, create_model

from app.config import CONTRACT_PATH, JOBLIB_PATH, METADATA_PATH
from app.model.contract import load_contract
from app.model.loader import LoadedModel, load_runtime
from app.services.predictor import predict_record

_contract = load_contract(CONTRACT_PATH)


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


_request_fields: dict[str, tuple[type, Any]] = {}
for _name in _contract.feature_columns_in_order:
    if _name in _contract.categorical_set:
        _request_fields[_name] = (str, Field(...))
    else:
        _request_fields[_name] = (float, Field(...))

PredictRequest = create_model(
    "PredictRequest",
    __base__=_StrictModel,
    **_request_fields,
)


class PredictResponse(BaseModel):
    predicted_class: str
    confidence: float
    model_version: str
    latency_ms: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.runtime = load_runtime(JOBLIB_PATH, METADATA_PATH, CONTRACT_PATH)
    yield
    app.state.runtime = None


app = FastAPI(
    title="Network Intrusion Detection Demonstrator",
    description="Phase 1 inference API around the locked NSL-KDD Random Forest pipeline.",
    version="0.1.0",
    lifespan=lifespan,
)


def _runtime(request: Request) -> LoadedModel:
    runtime = getattr(request.app.state, "runtime", None)
    if runtime is None:
        raise HTTPException(status_code=503, detail="Model is unavailable.")
    return runtime


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content={"detail": "Invalid request: missing or malformed fields."},
    )


@app.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    loaded = getattr(request.app.state, "runtime", None) is not None
    return HealthResponse(status="ok" if loaded else "error", model_loaded=loaded)


@app.get("/model/info")
def model_info(request: Request) -> dict[str, Any]:
    metadata = _runtime(request).metadata
    class_weight = metadata.get("class_weight")
    return {
        "model_type": metadata.get("model_type"),
        "algorithm": "Random Forest",
        "n_estimators": metadata.get("n_estimators"),
        "random_state": metadata.get("random_state"),
        "class_weight": class_weight,
        "validation_accuracy": metadata.get("validation_accuracy"),
        "validation_macro_f1": metadata.get("validation_macro_f1"),
        "training_partition": metadata.get("training_partition"),
        "artifact_filename": metadata.get("artifact_filename"),
    }


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest, request: Request) -> PredictResponse:
    runtime = _runtime(request)
    try:
        result = predict_record(
            runtime.pipeline,
            runtime.contract,
            payload.model_dump(),
        )
    except KeyError:
        raise HTTPException(status_code=422, detail="Invalid request: missing required fields.")
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=500, detail="Internal prediction error.")

    return PredictResponse(
        predicted_class=result.predicted_class,
        confidence=result.confidence,
        model_version=str(runtime.metadata.get("model_name", "nsl_kdd_random_forest_500")),
        latency_ms=round(result.latency_ms, 3),
    )

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, create_model
from starlette.concurrency import run_in_threadpool

from app.config import CONTRACT_PATH, JOBLIB_PATH, MAX_BATCH_SIZE, METADATA_PATH
from app.model.contract import load_contract
from app.model.loader import LoadedModel, load_runtime
from app.services.predictor import predict_batch, predict_one

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


class BatchPredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    records: list[PredictRequest]  # type: ignore[valid-type]


class BatchItemResponse(BaseModel):
    predicted_class: str
    confidence: float


class BatchPredictResponse(BaseModel):
    n_records: int
    predictions: list[BatchItemResponse]
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
    description="Inference API around the locked NSL-KDD Random Forest pipeline.",
    version="0.2.0",
    lifespan=lifespan,
)


def _runtime(request: Request) -> LoadedModel:
    runtime = getattr(request.app.state, "runtime", None)
    if runtime is None:
        raise HTTPException(status_code=503, detail="Model is unavailable.")
    return runtime


def _model_version(runtime: LoadedModel) -> str:
    return str(runtime.metadata.get("model_name", "nsl_kdd_random_forest_500"))


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    errors = []
    for err in exc.errors():
        errors.append(
            {
                "loc": [str(part) for part in err.get("loc", ())],
                "msg": err.get("msg"),
                "type": err.get("type"),
            }
        )
    return JSONResponse(
        status_code=422,
        content={
            "detail": "Invalid request: missing, extra, or malformed fields.",
            "errors": errors,
        },
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
        "max_batch_size": MAX_BATCH_SIZE,
    }


@app.post("/predict", response_model=PredictResponse)
async def predict(payload: PredictRequest, request: Request) -> PredictResponse:
    runtime = _runtime(request)
    try:
        result = await run_in_threadpool(
            predict_one,
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
        model_version=_model_version(runtime),
        latency_ms=round(result.latency_ms, 3),
    )


@app.post("/predict/batch", response_model=BatchPredictResponse)
async def predict_batch_endpoint(
    payload: BatchPredictRequest,
    request: Request,
) -> BatchPredictResponse:
    if len(payload.records) == 0:
        raise HTTPException(status_code=400, detail="Batch must contain at least one record.")
    if len(payload.records) > MAX_BATCH_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"Batch exceeds maximum size of {MAX_BATCH_SIZE} records.",
        )

    runtime = _runtime(request)
    try:
        batch = await run_in_threadpool(
            predict_batch,
            runtime.pipeline,
            runtime.contract,
            [row.model_dump() for row in payload.records],
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid batch request.") from exc
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=500, detail="Internal prediction error.")

    return BatchPredictResponse(
        n_records=len(batch.predictions),
        predictions=[
            BatchItemResponse(
                predicted_class=item.predicted_class,
                confidence=item.confidence,
            )
            for item in batch.predictions
        ],
        model_version=_model_version(runtime),
        latency_ms=round(batch.latency_ms, 3),
    )

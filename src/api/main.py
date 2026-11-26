import io
import logging
import time
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import RedirectResponse

from src.api.schemas import (BatchResponse, BatchRow, FeaturesRequest,
                             FeaturesResponse, RatingRequest, RatingResponse,
                             Recommendation, RecommendResponse)
from src.api.service import ModelService

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("mlops.api")

MAX_BATCH_ROWS = 10_000


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.service = ModelService()
    yield


app = FastAPI(title="MovieLens ML API", version="1.0", lifespan=lifespan)

@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    ms = (time.perf_counter() - start) * 1000
    log.info("%s %s -> %d (%.1f ms)", request.method, request.url.path,
             response.status_code, ms)
    return response


@app.get("/health")
def health():
    return {"status": "ok", "model_version": app.state.service.version}


@app.get("/model/info")
def model_info():
    svc = app.state.service
    return {"model_version": svc.version, "test_metrics": svc.metrics}


@app.post("/predict/rating", response_model=RatingResponse)
def predict_rating(req: RatingRequest):
    svc = app.state.service
    df = pd.DataFrame({"userId": [req.user_id], "movieId": [req.movie_id]})
    row = svc.predict_pairs(df).iloc[0]
    prob = float(row["liked_probability"])
    return RatingResponse(
        user_id=req.user_id,
        movie_id=req.movie_id,
        title=svc.titles.get(req.movie_id),
        predicted_rating=round(float(row["predicted_rating"]), 3),
        liked_probability=round(prob, 4),
        liked=prob >= 0.5,
        model_version=svc.version,
    )


@app.post("/predict/features", response_model=FeaturesResponse)
def predict_features(req: FeaturesRequest):
    return app.state.service.predict_features(req)


@app.get("/recommend/{user_id}", response_model=RecommendResponse)
def recommend(user_id: int, n: int = Query(10, ge=1, le=50)):
    svc = app.state.service
    known, items = svc.recommend(user_id, n)
    return RecommendResponse(
        user_id=user_id,
        known_user=known,
        recommendations=[Recommendation(movie_id=m, title=t) for m, t in items],
        model_version=svc.version,
    )


@app.post("/predict/batch", response_model=BatchResponse)
async def predict_batch(file: UploadFile = File(...)):
    if not (file.filename or "").lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Upload a .csv file")

    raw = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(raw))
    except Exception:
        raise HTTPException(status_code=400, detail="Could not parse the CSV file")

    if not {"userId", "movieId"}.issubset(df.columns):
        raise HTTPException(status_code=422,
                            detail="CSV must have the columns: userId, movieId")
    try:
        df = df[["userId", "movieId"]].dropna().astype(int).reset_index(drop=True)
    except (ValueError, TypeError):
        raise HTTPException(status_code=422,
                            detail="userId and movieId must be whole numbers")
    if not 0 < len(df) <= MAX_BATCH_ROWS:
        raise HTTPException(status_code=422,
                            detail=f"Send between 1 and {MAX_BATCH_ROWS} rows")

    svc = app.state.service
    out = await run_in_threadpool(svc.predict_pairs, df)
    rows = [
        BatchRow(user_id=int(r.userId), movie_id=int(r.movieId),
                 predicted_rating=round(float(r.predicted_rating), 3),
                 liked_probability=round(float(r.liked_probability), 4))
        for r in out.itertuples()
    ]
    return BatchResponse(n_rows=len(rows), model_version=svc.version, predictions=rows)
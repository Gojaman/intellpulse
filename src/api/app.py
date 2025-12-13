from __future__ import annotations
from mangum import Mangum

import os
from typing import Literal, Optional

from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.utils.data_loader import load_price_data
from src.features.price_features import build_price_feature_set
from src.models.signal_engine import generate_rule_based_signal, generate_combined_signal
from src.ingestion.sentiment_ingestion import load_sentiment_csv
from src.features.sentiment_features import (
    apply_sentiment_scorer,
    aggregate_sentiment_to_prices,
    get_sentiment_scorer,
)

app = FastAPI(title="Intellpulse API", version="0.2.0")

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # for dev. Later you can lock this to your Lovable domain.
    allow_credentials=False,       # must be False when allow_origins=["*"]
    allow_methods=["*"],
    allow_headers=["*"],
)



# -------------------------
# Helpers
# -------------------------
def _signal_to_text(sig: int) -> str:
    if sig > 0:
        return "BUY"
    if sig < 0:
        return "SELL"
    return "HOLD"


def _load_price_pipeline(asset: str):
    symbol_filter = asset.replace("-", "_")  # BTC-USD -> BTC_USD
    price = load_price_data(symbol_filter=symbol_filter)
    feat = build_price_feature_set(price)
    price_sig = generate_rule_based_signal(feat)
    return price_sig


def _load_aligned_sentiment(asset: str, price_df):
    path = os.getenv("SENTIMENT_CSV_PATH", "data/sentiment_sample.csv")
    sent_raw = load_sentiment_csv(path, asset_filter=asset)
    scorer = get_sentiment_scorer()  # naive or claude (env-controlled)
    sent_scored = apply_sentiment_scorer(sent_raw, scorer=scorer)
    sent_aligned = aggregate_sentiment_to_prices(sent_scored, price_df)
    return sent_aligned


# -------------------------
# Schemas
# -------------------------
class SentimentScoreRequest(BaseModel):
    text: str = Field(..., min_length=1)
    asset: Optional[str] = None


class SentimentScoreResponse(BaseModel):
    score: float
    engine: str


class SignalResponse(BaseModel):
    asset: str
    mode: str
    latest_timestamp: str
    latest_signal: int
    latest_signal_text: Literal["BUY", "HOLD", "SELL"]
    latest_sentiment: Optional[float] = None


class ExplainRequest(BaseModel):
    asset: str = "BTC-USD"
    mode: Literal["price_only", "combined"] = "combined"


class ExplainResponse(BaseModel):
    explanation: str


# -------------------------
# Routes
# -------------------------
@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/sentiment/score", response_model=SentimentScoreResponse)
def sentiment_score(req: SentimentScoreRequest):
    scorer = get_sentiment_scorer()
    score = float(scorer(req.text))
    score = max(0.0, min(1.0, score))
    return SentimentScoreResponse(score=score, engine=os.getenv("SENTIMENT_ENGINE", "naive"))


@app.get("/signal", response_model=SignalResponse)
def get_signal(
    asset: str = "BTC-USD",
    mode: Literal["price_only", "combined"] = "combined",
):
    price_sig = _load_price_pipeline(asset)

    latest_ts = price_sig.index[-1]
    latest_signal = int(price_sig["signal"].iloc[-1])
    latest_sentiment: Optional[float] = None

    if mode == "combined":
        sent_aligned = _load_aligned_sentiment(asset, price_sig)
        combined = generate_combined_signal(price_sig, sent_aligned)
        latest_signal = int(combined["signal_combined"].iloc[-1])
        latest_sentiment = float(combined["sentiment_score"].iloc[-1])

    return SignalResponse(
        asset=asset,
        mode=mode,
        latest_timestamp=latest_ts.isoformat(),
        latest_signal=latest_signal,
        latest_signal_text=_signal_to_text(latest_signal),
        latest_sentiment=latest_sentiment,
    )


@app.post("/signal/explain", response_model=ExplainResponse)
def explain_signal(req: ExplainRequest):
    # Pull the same “latest” values used by /signal
    price_sig = _load_price_pipeline(req.asset)
    latest_ts = price_sig.index[-1]
    price_signal = int(price_sig["signal"].iloc[-1])

    explanation_parts = [
        f"Asset: {req.asset}",
        f"Timestamp (latest bar): {latest_ts.isoformat()}",
        f"Price-model signal: {_signal_to_text(price_signal)} ({price_signal})",
    ]

    if req.mode == "combined":
        sent_aligned = _load_aligned_sentiment(req.asset, price_sig)
        combined = generate_combined_signal(price_sig, sent_aligned)
        combined_signal = int(combined["signal_combined"].iloc[-1])
        sentiment = float(combined["sentiment_score"].iloc[-1])

        explanation_parts += [
            f"Sentiment score (aligned): {sentiment:.2f} (0..1)",
            f"Combined signal: {_signal_to_text(combined_signal)} ({combined_signal})",
            "Logic: combined signal adjusts the price-model signal using recent sentiment strength.",
        ]
    else:
        explanation_parts += [
            "Mode: price_only",
            "Logic: signal is derived strictly from price features (no sentiment adjustment).",
        ] 

    return ExplainResponse(explanation="\n".join(explanation_parts))

handler = Mangum(app)


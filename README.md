# Intellpulse MVP — Signal API (AWS Lambda + FastAPI)
This API is designed for quantitative traders, fintech teams, and developers who need programmatic market signals with transparent decision logic.

Serverless quantitative signal engine that returns **trading signals + explainability** over HTTPS.

Intellpulse is a lightweight API that generates **BUY / HOLD / SELL** signals using price action and optional sentiment inputs.  
Built as a production-ready MVP with **API key authentication**, **quotas**, and **CI/CD**.

---

## What this MVP does

### Endpoints

- `GET /health` — public health check (no auth)
- `GET /signal` — latest BUY / HOLD / SELL signal  
  - modes: `price_only`, `combined`
- `GET /signal/explain` — signal + transparent explanation block
- `GET /backtest` — quick strategy backtest (metrics + equity curve tail)
- `GET /usage` — daily quota & plan status

---

## Key features

- API key authentication (**production + demo keys**)
- Rate limiting & daily quotas (DynamoDB)
- FastAPI packaged as a Docker image
- Deployed to AWS Lambda via container images
- CI/CD with AWS CodeBuild
- Digest-pinned production deploys (`@sha256:`)

---

## Live API (Production)

```bash
BASE_URL="https://5vjql4veoxz4tnvdr462x7rdoa0bvfuo.lambda-url.us-east-1.on.aws"
```
🔐 Access requires an API key
For demo access, request a DEMO_API_KEY (see Demo access below).

30-second demo
Replace <REQUEST_DEMO_KEY> with a real demo key.

1) Health check (no auth)


curl "$BASE_URL/health"
Expected:



{"status":"ok"}
2) Signal (authenticated)


curl -H "x-api-key: <REQUEST_DEMO_KEY>" \
  "$BASE_URL/signal?asset=BTC-USD&mode=combined"
Example response:



{
  "asset": "BTC-USD",
  "mode": "combined",
  "latest_signal_text": "HOLD",
  "latest_sentiment": 0.24,
  "cached_at_utc": "2025-12-23T14:55:55Z"
}
3) Explainability


curl -H "x-api-key: <REQUEST_DEMO_KEY>" \
  "$BASE_URL/signal/explain?asset=BTC-USD&mode=combined"
Returns:

signal decision

indicator contributions

sentiment impact

decision rationale

## Authentication behavior (by design)

| Scenario         | Result           |
|------------------|------------------|
| No API key       | 401 Unauthorized |
| Wrong API key    | 401 Unauthorized |
| Valid API key    | 200 OK           |
| Valid demo key   | 200 OK           |


## Architecture (high level)
FastAPI app packaged as a Docker image

AWS Lambda Function URL for HTTPS access

Amazon ECR for image storage

DynamoDB for rate limits, quotas, and plans

AWS CodeBuild for CI/CD (digest-pinned deploys)

## Why this MVP exists
This MVP demonstrates:

Real-world API design (auth, quotas, observability)

Cloud-native deployment (Lambda + containers)

Quant-style signal generation with explainability

A strong foundation for a trading intelligence SaaS

## What’s next (post-MVP)
Self-serve user API keys

Strategy customization

Web dashboard (signals + charts)

Additional assets & markets

Paid plans (quota-based tiers)

## Demo access
If you’re evaluating this project:

Request a DEMO_API_KEY directly from the project owner.
Demo keys are manually issued and rate-limited.


LinkedIn DM, or

GitHub issue titled “Demo key request”

Or review the code + CI/CD pipeline in this repo.




---
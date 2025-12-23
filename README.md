Intellpulse MVP — Signal API

Serverless quantitative signal engine (AWS Lambda + FastAPI)

Intellpulse is a lightweight API that generates trading signals using price action and optional sentiment inputs.
Built as a production-ready MVP with authentication, quotas, and CI/CD.

What this MVP does

Endpoints

GET /health — public health check

GET /signal — latest BUY / HOLD / SELL signal

modes: price_only, combined

GET /signal/explain — signal + transparent explanation

GET /backtest — quick strategy backtest (metrics + equity curve tail)

GET /usage — daily quota & plan status

Key features

API key authentication (prod + demo keys)

Rate limiting & daily quotas (DynamoDB)

Docker image deployed to AWS Lambda

CI/CD via AWS CodeBuild

Digest-pinned production deploys

Live API (Production)
BASE_URL="https://5vjql4veoxz4tnvdr462x7rdoa0bvfuo.lambda-url.us-east-1.on.aws"


🔐 Access requires an API key
For demo access, request a DEMO_API_KEY

30-second demo (how anyone can test it)
1️⃣ Health check (no auth)
curl "$BASE_URL/health"


Expected:

{"status":"ok"}

2️⃣ Signal (authenticated)
curl -H "x-api-key: <REQUEST_DEMO_KEY>" \
  "$BASE_URL/signal?asset=BTC-USD&mode=combined"


Expected (example):

{
  "asset": "BTC-USD",
  "latest_signal_text": "HOLD",
  "latest_sentiment": "neutral",
  "cached_at_utc": "2025-12-23T12:01:00Z"
}

3️⃣ Explainability
curl -H "x-api-key: <REQUEST_DEMO_KEY>" \
  "$BASE_URL/signal/explain?asset=BTC-USD&mode=combined"


Returns:

signal

indicator contributions

sentiment impact

decision rationale

Authentication behavior (by design)
Scenario	Result
No API key	401 Unauthorized
Wrong API key	401 Unauthorized
Valid API key	200 OK
Valid demo key	200 OK
Architecture (high level)

FastAPI app packaged as Docker image

AWS Lambda (Function URL) for serverless HTTPS access

Amazon ECR for image storage

DynamoDB for rate limits, quotas, and plans

AWS CodeBuild for CI/CD (digest-pinned deploys)

Why this MVP exists

This MVP demonstrates:

Real-world API design (auth, quotas, observability)

Cloud-native deployment (Lambda + containers)

Quant-style signal generation with explainability

A foundation for a full trading intelligence SaaS

What’s next (post-MVP)

User-scoped API keys (self-service)

Strategy customization

Web dashboard

Additional assets & markets

Paid plans

Contact / Demo access

If you’re evaluating this project:

Request a DEMO_API_KEY

Or review the code + CI/CD pipeline in this repo
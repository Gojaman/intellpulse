# Intellpulse MVP — Signal API (AWS Lambda + FastAPI)

A minimal signal API that returns:
- `/signal` — latest BUY/HOLD/SELL decision (price_only or combined sentiment)
- `/signal/explain` — same + transparent explain block
- `/backtest` — quick strategy backtest metrics + equity curve tail
- `/usage` — daily quota status (plan + remaining)
- `/health` — public health check

## Live endpoints (staging)
Set these locally:

```bash
export BASE_URL="https://5vjql4veoxz4tnvdr462x7rdoa0bvfuo.lambda-url.us-east-1.on.aws"
export API_KEY="REPLACE_ME"


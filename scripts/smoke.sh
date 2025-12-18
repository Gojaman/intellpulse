#!/usr/bin/env bash
set -euo pipefail

API_KEY="${API_KEY:?API_KEY not set}"
BASE_URL="${BASE_URL:?BASE_URL not set}"
BASE_URL="${BASE_URL%/}"

need() { command -v "$1" >/dev/null 2>&1 || { echo "Missing dependency: $1"; exit 2; }; }
need curl
need jq

fail() { echo "❌ $*"; exit 1; }

# Usage: req NAME PATH [JQ_FILTER]
req() {
  local name="$1"
  local path="$2"
  local filter="${3:-.}"
  local url="${BASE_URL}${path}"

  echo "▶ ${name}"
  local http body
  http="$(curl -sS -o /tmp/smoke_body.json -w "%{http_code}" \
    -H "x-api-key: ${API_KEY}" \
    "$url" || true)"
  body="$(cat /tmp/smoke_body.json || true)"

  # If not JSON, show raw body
  if ! echo "$body" | jq -e . >/dev/null 2>&1; then
    echo "$body"
    fail "${name} returned non-JSON (HTTP ${http})"
  fi

  if [[ "$http" != "200" ]]; then
    echo "$body" | jq .
    fail "${name} failed (HTTP ${http})"
  fi

  # Print only selected fields
  echo "$body" | jq -c "$filter" | jq .
}

echo "▶ Health"
health_http="$(curl -sS -o /tmp/smoke_health.json -w "%{http_code}" "${BASE_URL}/health" || true)"
health_body="$(cat /tmp/smoke_health.json || true)"
echo "$health_body" | jq . >/dev/null 2>&1 || { echo "$health_body"; fail "Health returned non-JSON (HTTP ${health_http})"; }
[[ "$health_http" == "200" ]] || { echo "$health_body" | jq .; fail "Health failed (HTTP ${health_http})"; }
echo "$health_body" | jq .

req "Signal"  "/signal?asset=BTC-USD&mode=combined" \
  '{asset, mode, latest_timestamp, latest_signal_text, latest_sentiment, cached_at_utc}'

req "Explain" "/signal/explain?asset=BTC-USD&mode=combined" \
  '{asset, mode, latest_timestamp, latest_signal_text, latest_sentiment, explain: (.explain.summary // null)}'

req "Usage"   "/usage" \
  '{plan, daily_limit, used_today, remaining, resets_at, date_utc}'

req "Backtest" "/backtest?asset=BTC-USD&mode=combined" \
  '{asset, mode, rows, total_return, sharpe, max_drawdown, trades, win_rate, equity_end, buy_hold_total_return, buy_hold_max_drawdown, buy_hold_equity_end}'

echo "✅ Smoke test passed"

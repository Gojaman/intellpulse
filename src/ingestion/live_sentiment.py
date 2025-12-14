import httpx

FNG_URL = "https://api.alternative.me/fng/?limit=1&format=json"

def get_fear_greed_score() -> float:
    """
    Returns a 0..1 sentiment score derived from Fear & Greed Index (0..100).
    """
    r = httpx.get(FNG_URL, timeout=5.0)
    r.raise_for_status()
    data = r.json()
    value_0_100 = float(data["data"][0]["value"])
    return max(0.0, min(1.0, value_0_100 / 100.0))

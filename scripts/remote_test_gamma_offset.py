import requests

base = "https://gamma-api.polymarket.com/markets"
tests = [
    {"closed": "true", "limit": "5", "order": "id", "ascending": "false"},
    {"closed": "true", "limit": "5", "order": "id", "ascending": "false", "offset": "1000"},
    {"closed": "true", "limit": "5", "order": "id", "ascending": "false", "offset": "10000"},
    {"closed": "true", "limit": "5", "order": "id", "ascending": "false", "offset": "30000"},
    {
        "closed": "true",
        "limit": "5",
        "order": "createdAt",
        "ascending": "false",
        "offset": "10000",
    },
]

for params in tests:
    response = requests.get(base, params=params, timeout=30)
    print("status", response.status_code, response.url)
    print(response.text[:250].replace("\n", " "))
    try:
        payload = response.json()
        rows = payload if isinstance(payload, list) else payload.get("markets") or []
        print("ids", [row.get("id") for row in rows[:5]])
    except Exception as exc:
        print("err", exc)
    print("---")

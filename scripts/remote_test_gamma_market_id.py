import requests

ids = ["2319668", "2320601", "2366960"]
for mid in ids:
    for params in ({"id": mid}, {"ids": mid}, {"market_id": mid}):
        params = {**params, "limit": "5"}
        response = requests.get("https://gamma-api.polymarket.com/markets", params=params, timeout=30)
        print("mid", mid, "params", params, "status", response.status_code, response.url)
        print(response.text[:250].replace("\n", " "))
        try:
            payload = response.json()
            rows = payload if isinstance(payload, list) else payload.get("markets") or []
            print("rows", len(rows), [row.get("id") for row in rows[:5]])
        except Exception as exc:
            print("err", exc)
        print("---")

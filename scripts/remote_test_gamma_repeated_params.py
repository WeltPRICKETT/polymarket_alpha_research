import requests

ids = [
    "10000044597233142009580369736770708957400032922928702133891118997446655093345",
    "10000056965700826009606607154170556213627789129748171579354099238814817437276",
    "100000633886833654986985145796184165075140247574884583752071124901969408436026",
]

params = [("clob_token_ids", token_id) for token_id in ids]
params.extend([("closed", "true"), ("limit", "10")])

response = requests.get("https://gamma-api.polymarket.com/markets", params=params, timeout=30)
print(response.status_code, response.url)
print(response.text[:1000])
try:
    payload = response.json()
    rows = payload if isinstance(payload, list) else payload.get("markets") or []
    print("rows", len(rows), [row.get("id") for row in rows[:10]])
except Exception as exc:
    print("json err", exc)

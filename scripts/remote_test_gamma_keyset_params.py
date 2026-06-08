import requests


tests = [
    ("default_closed", {"closed": "true", "limit": 5}),
    ("order_desc", {"closed": "true", "limit": 5, "order": "id", "ascending": "false"}),
    ("sort_desc", {"closed": "true", "limit": 5, "sort": "id", "order": "desc"}),
    ("id_desc", {"closed": "true", "limit": 5, "order": "id", "direction": "desc"}),
    ("created_desc", {"closed": "true", "limit": 5, "order": "createdAt", "ascending": "false"}),
    ("updated_desc", {"closed": "true", "limit": 5, "order": "updatedAt", "ascending": "false"}),
    ("id_gt_2200000", {"closed": "true", "limit": 5, "id_gt": "2200000"}),
    ("id_min_2200000", {"closed": "true", "limit": 5, "id_min": "2200000"}),
    ("created_after", {"closed": "true", "limit": 5, "created_after": "2026-05-04T00:00:00Z"}),
    ("end_after", {"closed": "true", "limit": 5, "end_date_min": "2026-05-04T00:00:00Z"}),
]

session = requests.Session()
url = "https://gamma-api.polymarket.com/markets/keyset"
for name, params in tests:
    try:
        response = session.get(url, params=params, timeout=30)
        print("TEST", name, "status", response.status_code, "url", response.url)
        print(response.text[:300].replace("\n", " "))
        payload = response.json()
        rows = payload.get("markets", [])
        print("ids", [row.get("id") for row in rows[:5]])
        print("slugs", [row.get("slug") for row in rows[:3]])
    except Exception as exc:
        print("ERR", name, exc)
    print("---")

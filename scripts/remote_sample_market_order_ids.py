import csv
import itertools
import json


print("MARKETS SAMPLE")
with open("data/markets.csv", newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    print("header_first20", reader.fieldnames[:20] if reader.fieldnames else None)
    print("ncols", len(reader.fieldnames or []))
    for row in itertools.islice(reader, 3):
        print("id", row.get("id"))
        raw = row.get("clobTokenIds")
        print("clobTokenIds raw", raw)
        try:
            parsed = json.loads(raw or "[]")
            print("parsed_first2", parsed[:2])
        except Exception as exc:
            print("parse_err", exc)

print("ORDER SAMPLE")
with open("data/orderFilled.csv", newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    print("order_header", reader.fieldnames)
    for row in itertools.islice(reader, 5):
        print(row.get("makerAssetId"), row.get("takerAssetId"))

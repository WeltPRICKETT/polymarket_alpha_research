import csv
import json
import shutil


src = "data/missing_markets.csv"
bak = "data/missing_markets.double_encoded.bak.csv"
shutil.copy2(src, bak)

rows = []
with open(src, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        raw = row.get("clobTokenIds") or ""
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, str):
                raw = parsed
        except Exception:
            pass
        rows.append((row.get("id", ""), raw))

with open(src, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["id", "clobTokenIds"])
    writer.writerows(rows)

print("rewrote", len(rows), "backup", bak)

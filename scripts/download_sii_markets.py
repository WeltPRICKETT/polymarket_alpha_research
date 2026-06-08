#!/usr/bin/env python
"""Download the small SII Polymarket markets metadata parquet."""

from pathlib import Path

from huggingface_hub import hf_hub_download


def main() -> None:
    target_dir = Path("data/external/sii_polymarket")
    target_dir.mkdir(parents=True, exist_ok=True)
    path = hf_hub_download(
        repo_id="SII-WANGZJ/Polymarket_data",
        repo_type="dataset",
        filename="markets.parquet",
        local_dir=str(target_dir),
    )
    print(path)


if __name__ == "__main__":
    main()

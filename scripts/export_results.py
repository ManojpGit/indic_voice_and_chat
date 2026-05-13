"""Export the latest in-memory benchmark suite results to CSVs.

Usage (typically wired to the FastAPI app's runner):
    python scripts/export_results.py --runner <pickled state> --out-dir data/results
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export benchmark run results")
    parser.add_argument(
        "--results-json",
        required=True,
        type=Path,
        help="Path to a JSON file produced by SuiteRunner.record (one record per file)",
    )
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    payload = json.loads(args.results_json.read_text(encoding="utf-8"))
    results = payload.get("results") or {}

    # We can't reconstruct full dataclasses generically — write the slices as
    # CSV by hand. Each block in ``results`` is already a flat JSON map.
    csv_files: list[Path] = []
    for kind, contents in results.items():
        if contents is None:
            continue
        path = args.out_dir / f"{kind}.json"
        path.write_text(json.dumps(contents, indent=2, ensure_ascii=False), encoding="utf-8")
        csv_files.append(path)
    print(f"wrote {len(csv_files)} files to {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

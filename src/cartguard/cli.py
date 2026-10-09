"""Training and batch inference for complete retail pipelines."""

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from .data import CATEGORIES, build_features


def main():
    p = argparse.ArgumentParser(description="CartGuard retail feedback pipelines")
    p.add_argument("command", choices=["run", "predict"])
    p.add_argument("--root", type=Path, default=Path.cwd())
    p.add_argument("--input", type=Path)
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    if args.command == "run":
        from .pipeline import run

        print(run(args.root).to_string(index=False))
    else:
        if args.input is None or args.output is None:
            p.error("predict requires --input and --output")
        frame = pd.read_csv(args.input)
        if set(["Review Text", "Age", *CATEGORIES]) - set(frame):
            raise ValueError("Provide Review Text, Age and product category columns")
        x = build_features(frame)
        if x.text.eq("").any():
            raise ValueError("Reviews must contain words")
        model = joblib.load(args.root / "models/review_triage.joblib")
        meta = json.loads((args.root / "models/inference.json").read_text())
        frame["non_recommendation_probability"] = model.predict_proba(x)[:, 1]
        frame["needs_review"] = (
            frame.non_recommendation_probability >= meta["threshold"]
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()

"""Command line interface for Intelligent Infrastructure Monitor."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .model import DEFAULT_MODEL_PATH
from .pipeline import analyze_logs, read_logs, train_from_csv


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="iim",
        description="Train and run the Intelligent Infrastructure Monitor anomaly detector.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    train = sub.add_parser("train", help="Tune and train the anomaly detection model.")
    train.add_argument("--input", default="data/sample_infrastructure_logs.csv", help="Input infrastructure log CSV.")
    train.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH), help="Where to save the trained model.")

    analyze = sub.add_parser("analyze", help="Score logs and write anomaly reports.")
    analyze.add_argument("--input", default="data/sample_infrastructure_logs.csv", help="Input infrastructure log CSV.")
    analyze.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH), help="Trained model bundle.")
    analyze.add_argument("--output-dir", default="reports", help="Output folder for CSV/JSON reports.")
    analyze.add_argument("--explain", action="store_true", help="Generate explanations for the highest-scoring anomalies.")
    analyze.add_argument("--max-explanations", type=int, default=10, help="Maximum anomaly explanations to request.")

    demo = sub.add_parser("demo", help="Train on the sample data and analyze it end-to-end.")
    demo.add_argument("--input", default="data/sample_infrastructure_logs.csv", help="Sample or labelled log CSV.")
    demo.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH), help="Where to save the model.")
    demo.add_argument("--output-dir", default="reports", help="Output folder.")
    demo.add_argument("--explain", action="store_true", help="Generate explanations (LLM key optional).")
    demo.add_argument("--max-explanations", type=int, default=10)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "train":
        result = train_from_csv(args.input, args.model_path)
        print(json.dumps(result, indent=2, default=str))
    elif args.command == "analyze":
        scored, result = analyze_logs(
            read_logs(args.input), model_path=args.model_path, output_dir=args.output_dir,
            explain=args.explain, max_explanations=args.max_explanations,
        )
        print(json.dumps(result, indent=2, default=str))
        print(f"\nSaved {len(scored)} scored rows to {Path(args.output_dir) / 'anomaly_results.csv'}")
    elif args.command == "demo":
        training = train_from_csv(args.input, args.model_path)
        scored, result = analyze_logs(
            read_logs(args.input), model_path=args.model_path, output_dir=args.output_dir,
            explain=args.explain, max_explanations=args.max_explanations,
        )
        print("TRAINING SUMMARY")
        print(json.dumps(training, indent=2, default=str))
        print("\nANALYSIS SUMMARY")
        print(json.dumps(result, indent=2, default=str))
        print(f"\nDemo complete: {len(scored)} rows scored.")


if __name__ == "__main__":
    main()

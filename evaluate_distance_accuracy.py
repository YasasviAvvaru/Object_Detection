import argparse
import csv
import math
from collections import defaultdict
from typing import Dict, List


def read_rows(path: str) -> List[Dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def summarize(rows: List[Dict[str, str]]) -> Dict[str, float]:
    errors = []
    abs_pct_errors = []

    for row in rows:
        predicted = float(row["predicted_m"])
        actual = float(row["actual_m"])
        if actual <= 0:
            continue
        err = predicted - actual
        errors.append(err)
        abs_pct_errors.append(abs(err) / actual * 100.0)

    if not errors:
        raise ValueError("No valid rows found. Need predicted_m and actual_m columns.")

    mae = sum(abs(e) for e in errors) / len(errors)
    rmse = math.sqrt(sum(e * e for e in errors) / len(errors))
    mape = sum(abs_pct_errors) / len(abs_pct_errors)
    accuracy = max(0.0, 100.0 - mape)

    return {
        "samples": float(len(errors)),
        "mae_m": mae,
        "rmse_m": rmse,
        "mape_percent": mape,
        "distance_accuracy_percent": accuracy,
    }


def print_summary(title: str, metrics: Dict[str, float]) -> None:
    print(title)
    print(f"  samples: {int(metrics['samples'])}")
    print(f"  MAE: {metrics['mae_m']:.3f} m")
    print(f"  RMSE: {metrics['rmse_m']:.3f} m")
    print(f"  MAPE: {metrics['mape_percent']:.2f}%")
    print(f"  distance accuracy: {metrics['distance_accuracy_percent']:.2f}%")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate distance accuracy from a CSV containing method,predicted_m,actual_m columns."
    )
    parser.add_argument("csv_path", help="Evaluation CSV path.")
    args = parser.parse_args()

    rows = read_rows(args.csv_path)
    print_summary("Overall", summarize(rows))

    by_method = defaultdict(list)
    for row in rows:
        by_method[row.get("method", "unknown")].append(row)

    if len(by_method) > 1:
        print()
        for method, method_rows in sorted(by_method.items()):
            print_summary(method, summarize(method_rows))


if __name__ == "__main__":
    main()

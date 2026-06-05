#!/usr/bin/env python3
"""
Evaluate POLLEN results.

Reads responses.json (and per_query.csv) from a results directory
and computes comprehensive metrics:

  RESTGPT METRICS:
    - S%   (Success Rate):        % queries where all tasks executed successfully
    - CP%  (Correct Path Rate):   % queries whose task sequence matches oracle
    - ΔSL  (Delta Solution Length): avg |planned| - |oracle|

  PLAN-LEVEL METRICS (per-query and aggregate):
    - Precision, Recall, F1
    - Accuracy, Jaccard
    - Coverage, Overprediction/Underprediction rates
    - Micro & Macro averages

  LATENCY STATISTICS:
    - Mean, P50, P95 planning & total latency

Usage:
    python benchmark/post_evaluation.py
    python benchmark/post_evaluation.py --results-dir path/to/results
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from urllib.parse import urlparse
import re

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results" / "qwen" / "spotify-dist4"


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate POLLEN results")
    p.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR),
                   help="Results directory containing responses.json and per_query.csv")
    return p.parse_args()


def load_adjusted_responses(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_adjusted_csv(path):
    rows = []
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["query_index"] = int(row["query_index"])
            row["oracle_count"] = int(row["oracle_count"])
            row["planned_count"] = int(row["planned_count"])
            row["delta_sl"] = int(row["delta_sl"])
            row["is_correct_path"] = row["is_correct_path"].strip() == "True"
            row["tasks_successful"] = row["tasks_successful"].strip() == "True"
            row["planning_latency_s"] = float(row["planning_latency_s"])
            row["total_latency_s"] = float(row["total_latency_s"])
            rows.append(row)
    return rows


# ── Plan-level metrics (precision, recall, f1, etc.) ──

def compute_plan_metrics(response_entry):
    """Compute per-query metrics from a response entry."""
    tasks = response_entry.get("execution_plan", {}).get("tasks", [])
    oracle_steps = response_entry.get("oracle", [])
    matches = response_entry.get("matches", [])
    mismatches = response_entry.get("mismatches", [])
    oracle_missed = response_entry.get("oracle_missed", [])

    tp = len(matches)
    fp = len(mismatches)
    fn = len(oracle_missed)
    total = tp + fp + fn

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    accuracy = tp / total if total > 0 else 0.0
    jaccard = tp / (tp + fp + fn) if (tp + fp + fn) else 0.0
    coverage = tp / len(oracle_steps) if oracle_steps else 0.0
    overpred_rate = fp / len(tasks) if tasks else 0.0
    underpred_rate = fn / len(oracle_steps) if oracle_steps else 0.0

    return {
        "tp": tp, "fp": fp, "fn": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "accuracy": round(accuracy, 4),
        "jaccard": round(jaccard, 4),
        "coverage": round(coverage, 4),
        "overpred_rate": round(overpred_rate, 4),
        "underpred_rate": round(underpred_rate, 4),
    }


# ── Aggregate metrics ──

def compute_aggregate_metrics(results):
    total = len(results)
    if total == 0:
        return {"total_queries": 0}

    successful = sum(1 for r in results if r.get("tasks_successful"))
    correct_path = sum(1 for r in results if r.get("is_correct_path"))
    deltas = [r.get("delta_sl", 0) for r in results]
    planning_latencies = [r.get("planning_latency_s", 0) for r in results]
    total_latencies = [r.get("total_latency_s", 0) for r in results]

    # Per-query plan metrics
    plan_metrics_list = [compute_plan_metrics(r) for r in results]

    total_tp = sum(m["tp"] for m in plan_metrics_list)
    total_fp = sum(m["fp"] for m in plan_metrics_list)
    total_fn = sum(m["fn"] for m in plan_metrics_list)

    # Micro averages (global)
    micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0.0
    micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0.0
    micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall) if (micro_precision + micro_recall) else 0.0

    # Macro averages (per-query averaged)
    macro_precision = sum(m["precision"] for m in plan_metrics_list) / total
    macro_recall = sum(m["recall"] for m in plan_metrics_list) / total
    macro_f1 = sum(m["f1"] for m in plan_metrics_list) / total

    # Global accuracy & jaccard
    global_accuracy = total_tp / (total_tp + total_fp + total_fn) if (total_tp + total_fp + total_fn) else 0.0
    global_jaccard = total_tp / (total_tp + total_fp + total_fn) if (total_tp + total_fp + total_fn) else 0.0

    # Latency stats
    sorted_planning = sorted(planning_latencies)
    n = len(sorted_planning)
    p50_planning = sorted_planning[n // 2]
    p95_idx = min(int(n * 0.95), n - 1)
    p95_planning = sorted_planning[p95_idx]

    avg_delta = sum(deltas) / total if total else 0.0
    cp_percentage = correct_path / total * 100
    s_percentage = successful / total * 100

    status_counts = {}
    for r in results:
        s = r.get("http_status", 0)
        status_counts[s] = status_counts.get(s, 0) + 1

    return {
        "total_queries": total,
        "successful_queries": successful,
        "correct_path_queries": correct_path,
        "success_rate_s_pct": round(s_percentage, 2),
        "correct_path_rate_cp_pct": round(cp_percentage, 2),
        "avg_delta_sl": round(avg_delta, 2),
        "avg_planning_latency_s": round(sum(planning_latencies) / total, 3),
        "avg_total_latency_s": round(sum(total_latencies) / total, 3),
        "p50_planning_latency_s": round(p50_planning, 3),
        "p95_planning_latency_s": round(p95_planning, 3),
        "http_status_distribution": {str(k): v for k, v in sorted(status_counts.items())},

        "total_tp": total_tp,
        "total_fp": total_fp,
        "total_fn": total_fn,
        "micro_precision": round(micro_precision, 4),
        "micro_recall": round(micro_recall, 4),
        "micro_f1": round(micro_f1, 4),
        "macro_precision": round(macro_precision, 4),
        "macro_recall": round(macro_recall, 4),
        "macro_f1": round(macro_f1, 4),
        "global_accuracy": round(global_accuracy, 4),
        "global_jaccard": round(global_jaccard, 4),
    }


def write_metrics_json(metrics, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"  Metrics:  {path}")


def write_per_query_csv(results, path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "query_index", "query", "oracle_count", "planned_count",
            "delta_sl", "is_correct_path", "tasks_successful",
            "precision", "recall", "f1", "jaccard", "coverage",
            "planning_latency_s", "total_latency_s", "http_status",
        ])
        for r in results:
            pm = compute_plan_metrics(r)
            writer.writerow([
                r.get("query_index"), r.get("query"),
                r.get("oracle_count", 0), r.get("planned_count", 0),
                r.get("delta_sl", 0), r.get("is_correct_path", False),
                r.get("tasks_successful", False),
                pm["precision"], pm["recall"], pm["f1"],
                pm["jaccard"], pm["coverage"],
                r.get("planning_latency_s", 0), r.get("total_latency_s", 0),
                r.get("http_status", ""),
            ])
    print(f"  Per-query: {path}")


def main():
    args = parse_args()

    results_dir = Path(args.results_dir)
    resp_path = results_dir / "responses.json"
    csv_path = results_dir / "per_query.csv"

    if not resp_path.exists():
        print(f"ERROR: responses file not found: {resp_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading responses from: {resp_path}")
    results = load_adjusted_responses(resp_path)
    print(f"Loaded {len(results)} entries")

    output_dir = results_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Compute aggregate metrics
    metrics = compute_aggregate_metrics(results)
    metrics["source"] = str(resp_path)

    print(f"\n── Results ──")
    print(f"  Total:    {metrics['total_queries']}")
    print(f"  CP%:      {metrics['correct_path_rate_cp_pct']:.2f}%")
    print(f"  S%:       {metrics['success_rate_s_pct']:.2f}%")
    print(f"  ΔSL:      {metrics['avg_delta_sl']:.2f}")
    print(f"  Micro F1: {metrics['micro_f1']:.4f}")
    print(f"  Macro F1: {metrics['macro_f1']:.4f}")
    print(f"  Avg Plan: {metrics['avg_planning_latency_s']:.3f}s")

    # Write outputs
    write_metrics_json(metrics, output_dir / "metrics.json")
    write_per_query_csv(results, output_dir / "per_query.csv")

    print(f"\nOutput directory: {output_dir}")
    print("Done.")


if __name__ == "__main__":
    main()

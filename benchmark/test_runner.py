#!/usr/bin/env python3
"""
POLLEN Test Runner — Dataset-driven evaluation with restGpt metrics.

Loads benchmark datasets from datasets/, submits queries to the Control Unit,
and computes the metrics defined in the restGpt paper:
  - Success Rate (S%)         : percentage of queries where all tasks succeed
  - Correct Path rate (CP%)   : percentage of queries whose task sequence matches the oracle
  - Delta Solution Length (ΔSL): average absolute difference |planned| - |oracle|
  - Latency (sec)             : model-related execution time (planning)

Directory layout:
  results/
    test_runs/
      <dataset>_<model>_<run_id>/
        responses.json        — full raw responses per query
        metrics.json          — aggregated metrics (machine-readable)
        summary.txt           — human-readable report
        per_query.csv         — per-query detailed breakdown

Usage:
    # Run against the currently running control unit (default http://localhost:5500):
    python benchmark/test_runner.py

    # Specify a dataset and control-unit URL:
    python benchmark/test_runner.py --datasets tmdb spotify --control-url http://localhost:5500

    # Custom results directory:
    python benchmark/test_runner.py --output-dir results/my_run
"""

import argparse
import csv
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

import requests


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASETS_DIR = PROJECT_ROOT / "datasets"
RESULTS_DIR  = PROJECT_ROOT / "results" / "test_runs"

BACKEND_TYPES = {
    "ollama": "Ollama",
    "llamacpp": "llama.cpp",
}


def parse_args():
    p = argparse.ArgumentParser(description="POLLEN dataset-driven test runner")
    p.add_argument(
        "--datasets", nargs="+", default=None,
        help="Datasets to evaluate (default: all JSON files in datasets/)"
    )
    p.add_argument(
        "--control-url", default="http://localhost:5500",
        help="Control Unit base URL (default: http://localhost:5500)"
    )
    p.add_argument(
        "--output-dir", default=None,
        help="Output directory (default: results/test_runs/<dataset>_<model>_<timestamp>)"
    )
    p.add_argument(
        "--model", default=None,
        help="Model identifier for directory naming (default: from env LLM_MODEL or 'unknown')"
    )
    p.add_argument(
        "--run-id", default=None,
        help="Custom run identifier (default: auto-generated timestamp)"
    )
    p.add_argument(
        "--max-queries", type=int, default=None,
        help="Limit queries per dataset (for quick smoke tests)"
    )
    return p.parse_args()


def discover_datasets(selected=None):
    datasets = {}
    for fpath in sorted(DATASETS_DIR.glob("*.json")):
        stem = fpath.stem
        if selected and stem not in selected:
            continue
        with open(fpath, "r", encoding="utf-8") as f:
            datasets[stem] = json.load(f)
    return datasets


def invoke_control_unit(query, control_url):
    payload = {"input": query}
    t0 = time.perf_counter()
    resp = requests.post(
        f"{control_url}/api/control/invoke",
        json=payload,
        timeout=None,
    )
    elapsed = time.perf_counter() - t0
    http_status = resp.status_code
    try:
        data = resp.json()
    except Exception:
        data = {"error": f"non-json response (HTTP {http_status})", "raw_text": resp.text[:500]}
    return data, elapsed, http_status


def extract_operation_path(task):
    operation = task.get("operation", "GET").upper()
    url = task.get("url") or task.get("endpoint") or ""
    parsed = urlparse(url)
    path = parsed.path.rstrip("/")
    if not path:
        path = url.split(" ")[-1] if " " in url else url
    return operation, path


def normalize_oracle_path(path):
    path = path.strip().rstrip("/")
    return path


def _segments_match(planned_seg, oracle_seg):
    """Compare two URL path segments, treating oracle placeholders as wildcards."""
    if re.fullmatch(r'\{[^}]+\}', oracle_seg):
        return True
    if re.fullmatch(r'\{[^}]+\}', planned_seg):
        return True
    return planned_seg == oracle_seg


def compare_plan_with_oracle(tasks, oracle_steps, backend_mode="MOCK"):
    planned_steps = []
    for task in tasks:
        operation = task.get("operation", "").upper()
        url = task.get("url", "")
        if not url:
            planned_steps.append((operation, ""))
            continue
        parsed = urlparse(url)
        path = parsed.path.rstrip("/")
        if not path and " " in url:
            path = url.split(" ")[-1].rstrip("/")
        planned_steps.append((operation, path))

    oracle_normalized = []
    for step in oracle_steps:
        parts = step.strip().split(maxsplit=1)
        if len(parts) == 2:
            op = parts[0].strip().upper()
            path = normalize_oracle_path(parts[1])
            oracle_normalized.append((op, path))

    matches = []
    mismatches = []
    matched_indices = set()

    for p_op, p_path in planned_steps:
        found = False
        for i, (o_op, o_path) in enumerate(oracle_normalized):
            if i in matched_indices:
                continue
            if p_op != o_op:
                continue
            p_segments = [s for s in p_path.split("/") if s]
            o_segments = [s for s in o_path.split("/") if s]
            if len(p_segments) != len(o_segments):
                if p_path.endswith(o_path):
                    matches.append({"planned": (p_op, p_path), "oracle": (o_op, o_path)})
                    matched_indices.add(i)
                    found = True
                    break
                continue
            if all(_segments_match(ps, os) for ps, os in zip(p_segments, o_segments)):
                matches.append({"planned": (p_op, p_path), "oracle": (o_op, o_path)})
                matched_indices.add(i)
                found = True
                break
        if not found:
            mismatches.append({"planned": (p_op, p_path), "oracle": None})

    oracle_missed = []
    for i, (o_op, o_path) in enumerate(oracle_normalized):
        if i not in matched_indices:
            oracle_missed.append({"oracle": (o_op, o_path), "planned": None})

    is_correct_path = len(mismatches) == 0 and len(oracle_missed) == 0

    return {
        "planned_steps": planned_steps,
        "oracle_steps": oracle_normalized,
        "matches": matches,
        "mismatches": mismatches,
        "oracle_missed": oracle_missed,
        "is_correct_path": is_correct_path,
        "num_planned": len(planned_steps),
        "num_oracle": len(oracle_normalized),
    }


def is_all_successful(execution_results):
    for r in execution_results:
        if r.get("status") != "SUCCESS":
            return False
    return True


def compute_delta_sl(planned_count, oracle_count):
    return abs(planned_count - oracle_count)


def evaluate_dataset(dataset_name, queries, control_url, max_queries=None, output_dir=None):
    if max_queries:
        queries = queries[:max_queries]

    total = len(queries)
    results = []

    for idx, entry in enumerate(queries):
        query = entry["query"]
        oracle = entry.get("solution", [])

        print(f"[{idx+1}/{total}] {dataset_name}: {query[:80]}{'...' if len(query) > 80 else ''}")

        try:
            response_data, total_latency, http_status = invoke_control_unit(query, control_url)
        except requests.exceptions.RequestException as e:
            print(f"  ERROR: HTTP request failed — {e}")
            response_data = {"error": str(e)}
            total_latency = 0.0
            http_status = 0

        execution_plan = response_data.get("execution_plan", {})
        execution_results = response_data.get("execution_results", [])
        planning_latency = response_data.get("planning_latency_s", 0.0) or 0.0
        error = response_data.get("error", "")

        tasks = execution_plan.get("tasks", [])
        comparison = compare_plan_with_oracle(tasks, oracle)

        tasks_successful = is_all_successful(execution_results) if execution_results else False
        delta_sl = compute_delta_sl(comparison["num_planned"], comparison["num_oracle"])

        per_query = {
            "query_index": idx + 1,
            "query": query,
            "oracle": oracle,
            "error": error,
            "http_status": http_status,
            "planning_latency_s": round(planning_latency, 3),
            "total_latency_s": round(total_latency, 3),
            "planned_count": comparison["num_planned"],
            "oracle_count": comparison["num_oracle"],
            "delta_sl": delta_sl,
            "is_correct_path": comparison["is_correct_path"],
            "tasks_successful": tasks_successful,
            "matches": comparison["matches"],
            "mismatches": comparison["mismatches"],
            "oracle_missed": comparison["oracle_missed"],
            "execution_plan": execution_plan,
            "execution_results": [
                {
                    "task_name": r.get("task_name"),
                    "operation": r.get("operation"),
                    "status": r.get("status"),
                    "status_code": r.get("status_code"),
                }
                for r in (execution_results or [])
            ],
        }
        results.append(per_query)

        status = "OK" if comparison["is_correct_path"] and tasks_successful else "MISMATCH"
        print(f"  → {status}  planned={comparison['num_planned']} oracle={comparison['num_oracle']} "
              f"ΔSL={delta_sl} latency={planning_latency:.2f}s")

        if output_dir:
            _save_incremental(results, dataset_name, output_dir)

    return results


def _save_incremental(results, dataset_name, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)

    tmp = output_dir / ".responses.tmp"
    final = output_dir / "responses.json"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    tmp.replace(final)

    tmp = output_dir / ".per_query.tmp"
    final = output_dir / "per_query.csv"
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "query_index", "query", "oracle_count", "planned_count",
            "delta_sl", "is_correct_path", "tasks_successful",
            "planning_latency_s", "total_latency_s", "http_status", "error",
        ])
        for r in results:
            writer.writerow([
                r["query_index"], r["query"], r["oracle_count"],
                r["planned_count"], r["delta_sl"], r["is_correct_path"],
                r["tasks_successful"], r["planning_latency_s"],
                r["total_latency_s"], r.get("http_status", ""), r.get("error", ""),
            ])
    tmp.replace(final)

    metrics = compute_aggregate_metrics(results, dataset_name)
    tmp = output_dir / ".metrics.tmp"
    final = output_dir / "metrics.json"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    tmp.replace(final)


def compute_aggregate_metrics(results, dataset_name):
    total = len(results)
    if total == 0:
        return {
            "dataset": dataset_name,
            "total_queries": 0,
            "success_rate": 0.0,
            "correct_path_rate": 0.0,
            "avg_delta_sl": 0.0,
            "avg_planning_latency_s": 0.0,
            "avg_total_latency_s": 0.0,
            "p50_planning_latency_s": 0.0,
            "p95_planning_latency_s": 0.0,
        }

    successful = sum(1 for r in results if r["tasks_successful"])
    correct_path = sum(1 for r in results if r["is_correct_path"])
    total_delta_sl = sum(r["delta_sl"] for r in results)
    planning_latencies = [r["planning_latency_s"] for r in results]
    total_latencies = [r["total_latency_s"] for r in results]
    sorted_planning = sorted(planning_latencies)
    sorted_total = sorted(total_latencies)

    n = len(sorted_planning)
    p50_planning = sorted_planning[n // 2]
    p95_idx = min(int(n * 0.95), n - 1)
    p95_planning = sorted_planning[p95_idx]

    # HTTP status distribution
    status_counts = {}
    http_errors = 0
    for r in results:
        s = r.get("http_status", 0)
        status_counts[s] = status_counts.get(s, 0) + 1
        if s and s >= 400:
            http_errors += 1

    return {
        "dataset": dataset_name,
        "total_queries": total,
        "success_rate_s": round(successful / total * 100, 2),
        "correct_path_rate_cp": round(correct_path / total * 100, 2),
        "avg_delta_sl": round(total_delta_sl / total, 2),
        "avg_planning_latency_s": round(sum(planning_latencies) / total, 3),
        "avg_total_latency_s": round(sum(total_latencies) / total, 3),
        "p50_planning_latency_s": round(p50_planning, 3),
        "p95_planning_latency_s": round(p95_planning, 3),
        "successful_queries": successful,
        "correct_path_queries": correct_path,
        "http_status_distribution": {str(k): v for k, v in sorted(status_counts.items())},
        "http_error_count": http_errors,
    }


def write_responses(results, output_dir):
    path = output_dir / "responses.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"  Responses: {path}")


def write_metrics(metrics, output_dir):
    path = output_dir / "metrics.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"  Metrics:   {path}")


def write_per_query_csv(results, output_dir):
    path = output_dir / "per_query.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "query_index", "query", "oracle_count", "planned_count",
            "delta_sl", "is_correct_path", "tasks_successful",
            "planning_latency_s", "total_latency_s", "http_status", "error",
        ])
        for r in results:
            writer.writerow([
                r["query_index"], r["query"], r["oracle_count"],
                r["planned_count"], r["delta_sl"], r["is_correct_path"],
                r["tasks_successful"], r["planning_latency_s"],
                r["total_latency_s"], r.get("http_status", ""), r.get("error", ""),
            ])
    print(f"  Per-query: {path}")


def write_summary(metrics, output_dir, run_label):
    path = output_dir / "summary.txt"
    lines = [
        "=" * 64,
        f"  POLLEN Test Runner — Summary",
        f"  Run: {run_label}",
        f"  Dataset: {metrics['dataset']}",
        "=" * 64,
        "",
        f"  Total queries:       {metrics['total_queries']}",
        f"  Successful (S):      {metrics['successful_queries']}",
        f"  Correct path (CP):   {metrics['correct_path_queries']}",
        f"  HTTP errors (4xx/5xx): {metrics.get('http_error_count', 0)}",
        "",
        f"  Success Rate (S%):         {metrics['success_rate_s']:.2f}%",
        f"  Correct Path Rate (CP%):   {metrics['correct_path_rate_cp']:.2f}%",
        f"  Avg Δ Solution Length:     {metrics['avg_delta_sl']:.2f}",
        f"  HTTP Status Distribution:  {metrics.get('http_status_distribution', {})}",
        "",
        "  Latency (planning):",
        f"    Average:  {metrics['avg_planning_latency_s']:.3f} s",
        f"    P50:      {metrics['p50_planning_latency_s']:.3f} s",
        f"    P95:      {metrics['p95_planning_latency_s']:.3f} s",
        "",
        f"  Latency (total):",
        f"    Average:  {metrics['avg_total_latency_s']:.3f} s",
        "",
    ]
    text = "\n".join(lines)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"  Summary:  {path}")


def main():
    args = parse_args()

    datasets = discover_datasets(args.datasets)
    if not datasets:
        print("No datasets found.")
        sys.exit(1)

    print(f"Discovered datasets: {list(datasets.keys())}")
    model = args.model or os.environ.get("LLM_MODEL", "unknown")
    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")

    for dataset_name, queries in datasets.items():
        print(f"\n{'=' * 64}")
        print(f"  Dataset: {dataset_name} ({len(queries)} queries)")
        print(f"{'=' * 64}")

        run_label = f"{dataset_name}_{model}_{run_id}"
        if args.output_dir:
            output_dir = Path(args.output_dir)
        else:
            output_dir = RESULTS_DIR / run_label

        results = evaluate_dataset(
            dataset_name, queries, args.control_url, args.max_queries, output_dir
        )

        metrics = compute_aggregate_metrics(results, dataset_name)
        metrics["model"] = model
        metrics["run_id"] = run_id
        metrics["control_url"] = args.control_url

        write_responses(results, output_dir)
        write_metrics(metrics, output_dir)
        write_per_query_csv(results, output_dir)
        write_summary(metrics, output_dir, run_label)

        print(f"\n  Results saved to: {output_dir}")

    print("\nDone.")


if __name__ == "__main__":
    main()

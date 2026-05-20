#!/usr/bin/env python3
"""
POLLEN Benchmark — Manual Workflow Helper

Prepares configuration and processes results for RestBench runs.
You manually run the tests, this script handles the rest.

Usage:
    # 1. Configure .env.pi for a model (do this for each model):
    python benchmark/pipeline.py --config dist-2 --model llama2 --configure

    # 2. Manually restart the swarm and run tests:
    #    docker compose -f docker-compose.pi.yml up -d
    #    python test_validation/testRunner.py \\
    #        --export-latency-csv results/llama2/dist-2/baseline/latencies.csv

    # 3. Process results and generate plots:
    python benchmark/pipeline.py --process
"""

import argparse
import csv
import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV_PI_PATH = os.path.join(PROJECT_ROOT, ".env.pi")

MODELS = {
    "llama2": {
        "id": "meta-llama/Llama-2-7b-chat-hf",
        "quant": "nf4",
        "hf_token": True,
    },
    "vicuna": {
        "id": "lmsys/vicuna-7b-v1.5",
        "quant": "nf4",
        "hf_token": False,
    },
}


def parse_args():
    p = argparse.ArgumentParser(description="POLLEN benchmark helper")
    p.add_argument("--config", choices=["dist-2", "dist-4", "dist-8"],
                   help="Pi configuration")
    p.add_argument("--model", choices=list(MODELS.keys()),
                   help="Model to configure")
    p.add_argument("--configure", action="store_true",
                   help="Update .env.pi for the given model and config, then print instructions")
    p.add_argument("--process", action="store_true",
                   help="Scan results/, compute metrics, and print summary table")
    p.add_argument("--add-model", nargs=2, metavar=("NAME", "HF_ID"),
                   help="Register a custom model (e.g. --add-model phi3 microsoft/Phi-3-mini-4k-instruct)")
    return p.parse_args()


def read_env_pi():
    if not os.path.exists(ENV_PI_PATH):
        return {}, []
    env = {}
    lines = []
    with open(ENV_PI_PATH) as f:
        for line in f:
            lines.append(line)
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if "=" in stripped:
                k, v = stripped.split("=", 1)
                env[k.strip()] = v.strip()
    return env, lines


def write_env_pi(updates):
    _, lines = read_env_pi()
    new_lines = []
    written_keys = set()
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            new_lines.append(line)
            continue
        if "=" in stripped:
            k = stripped.split("=", 1)[0].strip()
            if k in updates:
                new_lines.append(f"{k}={updates[k]}\n")
                written_keys.add(k)
            else:
                new_lines.append(line)
    for k, v in updates.items():
        if k not in written_keys:
            new_lines.append(f"{k}={v}\n")
    with open(ENV_PI_PATH, "w") as f:
        f.writelines(new_lines)


def do_configure(args):
    if not args.config or not args.model:
        print("ERROR: --configure requires --config and --model")
        sys.exit(1)

    model = MODELS[args.model]
    write_env_pi({"PETALS_MODEL": model["id"], "QUANT_TYPE": model["quant"]})

    config_num = args.config.split("-")[1]
    results_dir = f"results/{args.model}/{args.config}/baseline"

    print(f"\n  .env.pi updated for {args.model} ({args.config})")
    print(f"     PETALS_MODEL={model['id']}")
    print(f"     QUANT_TYPE={model['quant']}")
    print()
    print(f"  Now run these steps manually:")
    print()
    print(f"  1. Restart the swarm:")
    print(f"     docker compose -f docker-compose.pi.yml up -d")
    print()
    print(f"  2. Wait for the proxy to be healthy, then run RestBench:")
    print(f"     python test_validation/testRunner.py \\")
    print(f"         --export-latency-csv {results_dir}/latencies.csv")
    print()
    print(f"  3. Repeat for other models/configs, then process:")
    print(f"     python benchmark/pipeline.py --process")
    print()


def compute_metrics(csv_path):
    """Compute S%, CP%, DeltaSL from a latencies CSV file.

    Expected columns: verdict, l1_verdict, tasks_count, oracle_count, latency_s
    """
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))

    total = len(rows)
    if total == 0:
        return {"S": 0, "CP": 0, "DeltaSL": 0, "total": 0, "successful": 0, "latencies": []}

    error_verdicts = {"NETWORK_ERROR", "APP_ERROR"}
    latencies = [float(r["latency_s"]) for r in rows if r.get("latency_s")]
    successful = [r for r in rows if r["verdict"] not in error_verdicts]
    s_rate = len(successful) / total * 100

    cp_rate = 0
    if successful:
        cp = sum(1 for r in successful if r.get("l1_verdict") == "PASS")
        cp_rate = cp / len(successful) * 100

    delta_sl = 0
    if successful:
        deltas = []
        for r in successful:
            t = int(r.get("tasks_count", 0))
            o = int(r.get("oracle_count", 0))
            deltas.append(abs(t - o))
        delta_sl = sum(deltas) / len(deltas)

    return {
        "S": round(s_rate, 1),
        "CP": round(cp_rate, 1),
        "DeltaSL": round(delta_sl, 1),
        "total": total,
        "successful": len(successful),
        "latencies": latencies,
    }


def do_process():
    """Scan results/ directory and produce a metrics summary table."""
    results_base = os.path.join(PROJECT_ROOT, "results")
    if not os.path.exists(results_base):
        print("No results/ directory found.")
        return

    entries = []
    for model in sorted(os.listdir(results_base)):
        model_dir = os.path.join(results_base, model)
        if not os.path.isdir(model_dir):
            continue
        for config in sorted(os.listdir(model_dir)):
            csv_path = os.path.join(model_dir, config, "baseline", "latencies.csv")
            if os.path.exists(csv_path):
                metrics = compute_metrics(csv_path)
                metrics_path = os.path.join(model_dir, config, "metrics.json")
                with open(metrics_path, "w") as f:
                    json.dump(metrics, f, indent=2)

                lat = metrics.pop("latencies", [])
                entries.append((model, config, metrics, csv_path, metrics_path))

    if not entries:
        print("No latencies.csv files found under results/.")
        return

    print(f"\n{'='*80}")
    print(f"  POLLEN Benchmark \u2014 RestBench Metrics Summary")
    print(f"{'='*80}")
    print(f"  {'Model':<12} {'Config':<10} {'Queries':>8} {'OK':>5} {'S%':>7} {'CP%':>7} {'\u0394SL':>7} {'Avg(s)':>8} {'p50(s)':>8} {'p95(s)':>8}")
    print(f"  {'-'*80}")

    for model, config, metrics, csv_path, m_path in entries:
        lat = metrics.get("latencies", [])
        avg = f"{sum(lat)/len(lat):.2f}" if lat else "\u2014"
        p50 = f"{sorted(lat)[len(lat)//2]:.2f}" if lat else "\u2014"
        p95_val = sorted(lat)[int(len(lat)*0.95)] if lat else "\u2014"
        p95 = f"{p95_val:.2f}" if isinstance(p95_val, float) else "\u2014"

        print(f"  {model:<12} {config:<10} {metrics['total']:>8} {metrics['successful']:>5} "
              f"{metrics['S']:>6}% {metrics['CP']:>6}% {metrics['DeltaSL']:>6.1f} "
              f"{avg:>8} {p50:>8} {p95:>8}")
        print(f"  {'CSV:':<8} {csv_path}")
        print(f"  {'JSON:':<8} {m_path}")
        print()

    print(f"  Generate plots: python benchmark/plots.py")
    print()


def main():
    args = parse_args()

    if args.add_model:
        name, hf_id = args.add_model
        MODELS[name] = {"id": hf_id, "quant": "nf4", "hf_token": False}
        print(f"  Registered custom model: {name} -> {hf_id}")
        print(f"  Use: python benchmark/pipeline.py --config dist-4 --model {name} --configure")
        return

    if args.configure:
        do_configure(args)
    elif args.process:
        do_process()
    else:
        print("Specify --configure or --process")
        print("See: python benchmark/pipeline.py --help")


if __name__ == "__main__":
    main()

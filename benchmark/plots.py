#!/usr/bin/env python3
"""
Generate plots from POLLEN benchmark results.

Reads results/<model>/<config>/baseline/latencies.csv for each
model/config combination found and produces comparison plots.

Usage:
    python benchmark/plots.py                          # all available data
    python benchmark/plots.py --models llama2 vicuna   # specific models
    python benchmark/plots.py --configs dist-2 dist-4  # specific configs
"""

import argparse
import json
import os
import sys
import glob
from statistics import mean

import numpy as np
import pandas as pd
import plotly.graph_objects as go

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")


def parse_args():
    p = argparse.ArgumentParser(description="Generate POLLEN benchmark plots")
    p.add_argument("--models", nargs="+", default=None,
                   help="Filter to specific models")
    p.add_argument("--configs", nargs="+", default=None,
                   help="Filter to specific configs (dist-2, dist-4, dist-8)")
    p.add_argument("--output", default=os.path.join(RESULTS_DIR, "plots"),
                   help="Output directory for plots")
    return p.parse_args()


def discover_results():
    """Scan results/ and return {model: {config: latencies_list}}."""
    data = {}
    pattern = os.path.join(RESULTS_DIR, "*", "dist-*", "baseline", "latencies.csv")
    for csv_path in glob.glob(pattern):
        rel = os.path.relpath(csv_path, RESULTS_DIR)
        parts = rel.split(os.sep)
        if len(parts) >= 3:
            model, config = parts[0], parts[1]
            df = pd.read_csv(csv_path)
            if "latency_s" in df.columns:
                if model not in data:
                    data[model] = {}
                data[model][config] = df["latency_s"].tolist()
    return data


def calculate_qos_metrics(latencies, label):
    """Compute QoS statistics for a latency series."""
    lat = np.array(latencies)
    n = len(lat)
    total_time = np.sum(lat)
    avg_time = np.mean(lat)
    std_time = np.std(lat)
    p50 = np.percentile(lat, 50)
    p95 = np.percentile(lat, 95)
    p99 = np.percentile(lat, 99)
    min_time = np.min(lat)
    max_time = np.max(lat)
    outliers_p95 = np.sum(lat > p95) / n * 100
    throughput = n / total_time if total_time > 0 else float('inf')

    print(f"  Total workflow time: {total_time:.3f} s")
    print(f"  Average step time: {avg_time:.3f} s")
    print(f"  Std deviation: {std_time:.3f} s")
    print(f"  Min latency: {min_time:.3f} s")
    print(f"  Max latency: {max_time:.3f} s")
    print(f"  p50 latency: {p50:.3f} s")
    print(f"  p95 latency: {p95:.3f} s")
    print(f"  p99 latency: {p99:.3f} s")
    print(f"  % steps above p95: {outliers_p95:.2f}%")
    print(f"  Throughput: {throughput:.3f} workflows/sec")
    return {
        "total_time": total_time, "avg": avg_time, "std": std_time,
        "min": min_time, "max": max_time, "p50": p50, "p95": p95, "p99": p99,
        "outliers_p95": outliers_p95, "throughput": throughput,
    }


def print_qos(data):
    """Print QoS metrics for all model/config combinations."""
    print(f"\n{'='*72}")
    print("QoS Metrics")
    print(f"{'='*72}")
    for model in sorted(data):
        print(f"\n  Model: {model}")
        for config in sorted(data[model]):
            print(f"  Config: {config}")
            calculate_qos_metrics(data[model][config], f"{model} {config}")
            print()


def plot_latency_curves(data, output_dir, log_y=True):
    """Scatter plot with IQR band, median, p95, min/max per model."""
    for config in sorted({c for m in data for c in data[m]}):
        fig = go.Figure()
        colors = [
            ("rgb(31,119,180)", "rgba(31,119,180,0.15)"),
            ("rgb(214,39,40)",  "rgba(214,39,40,0.15)"),
            ("rgb(44,160,44)",  "rgba(44,160,44,0.15)"),
            ("rgb(255,127,14)", "rgba(255,127,14,0.15)"),
        ]

        matching = [(m, data[m][config]) for m in data if config in data[m]]
        for i, (model, latencies) in enumerate(matching):
            lat = np.array(latencies)
            x = np.arange(1, len(lat) + 1)
            median = np.array([np.median(lat[:k]) for k in x])
            q25 = np.array([np.percentile(lat[:k], 25) for k in x])
            q75 = np.array([np.percentile(lat[:k], 75) for k in x])
            p95 = np.array([np.percentile(lat[:k], 95) for k in x])
            minv = np.array([np.min(lat[:k]) for k in x])
            maxv = np.array([np.max(lat[:k]) for k in x])

            base_color, band_color = colors[i % len(colors)]
            lg = model

            fig.add_trace(go.Scatter(x=x, y=q25, line=dict(width=0), hoverinfo="skip", showlegend=False, legendgroup=lg))
            fig.add_trace(go.Scatter(x=x, y=q75, fill="tonexty", fillcolor=band_color, line=dict(width=0), hoverinfo="skip", showlegend=False, legendgroup=lg))
            fig.add_trace(go.Scatter(x=x, y=median, mode="lines", line=dict(color=base_color, width=3), name=model, legendgroup=lg, showlegend=True))
            fig.add_trace(go.Scatter(x=x, y=p95, mode="lines", line=dict(color=base_color, width=1.5, dash="dash"), showlegend=False, legendgroup=lg))
            fig.add_trace(go.Scatter(x=x, y=minv, mode="lines", line=dict(color=base_color, width=1, dash="dot"), showlegend=False, legendgroup=lg, hoverinfo="skip"))
            fig.add_trace(go.Scatter(x=x, y=maxv, mode="lines", line=dict(color=base_color, width=1, dash="dot"), showlegend=False, legendgroup=lg, hoverinfo="skip"))

        fig.update_layout(
            title=f"Latency Curves \u2014 {config}",
            xaxis_title="Question index",
            yaxis_title="Latency (seconds)",
            template="simple_white",
            font=dict(size=18),
            legend=dict(orientation="h", yanchor="top", y=-0.25, xanchor="center", x=0.5, title=None),
            xaxis=dict(showgrid=True, gridcolor="rgba(0,0,0,0.05)"),
            yaxis=dict(showgrid=True, gridcolor="rgba(0,0,0,0.05)", type="log" if log_y else "linear"),
        )

        out = os.path.join(output_dir, f"latency_curves_{config}.pdf")
        os.makedirs(output_dir, exist_ok=True)
        fig.write_image(out)
        print(f"  Saved: {out}")


def plot_boxplot(data, output_dir):
    """Grouped boxplots comparing models across configs."""
    configs = sorted({c for m in data for c in data[m]})
    models = sorted(data.keys())

    fig = go.Figure()
    colors = ["rgba(31,119,180,0.6)", "rgba(214,39,40,0.6)", "rgba(44,160,44,0.6)", "rgba(255,127,14,0.6)"]

    for i, model in enumerate(models):
        color = colors[i % len(colors)]
        for config in configs:
            if config in data[model]:
                label = f"{model} ({config})"
                fig.add_trace(go.Box(
                    y=data[model][config],
                    name=label,
                    marker_color=color,
                    line=dict(width=1.5),
                    showlegend=False,
                ))

    fig.update_layout(
        title="Latency Distribution by Model and Config",
        yaxis_title="Latency (seconds)",
        template="plotly_white",
        font=dict(size=18),
        margin=dict(b=120),
        xaxis=dict(tickangle=45),
    )

    out = os.path.join(output_dir, "boxplot_all.pdf")
    os.makedirs(output_dir, exist_ok=True)
    fig.write_image(out)
    print(f"  Saved: {out}")


def plot_bar_chart(data, output_dir):
    """Per-config bar chart comparing model latencies."""
    configs = sorted({c for m in data for c in data[m]})
    models = sorted(data.keys())
    palette = ["rgba(31, 119, 180, 0.7)", "rgba(255, 127, 14, 0.7)",
               "rgba(44, 160, 44, 0.7)", "rgba(214, 39, 40, 0.7)"]

    for config in configs:
        matching = {m: data[m][config] for m in models if config in data[m]}
        if not matching:
            continue

        max_len = max(len(v) for v in matching.values())
        x_vals = np.arange(1, max_len + 1)

        sorted_per_response = []
        for idx in range(max_len):
            response_vals = []
            for model in matching:
                if idx < len(matching[model]):
                    response_vals.append((model, matching[model][idx]))
            response_vals.sort(key=lambda x: x[1], reverse=True)
            sorted_per_response.append(response_vals)

        series_dict = {m: [] for m in matching}
        for idx in range(max_len):
            for order, (model, value) in enumerate(sorted_per_response[idx]):
                series_dict[model].append(value)

        sorted_models = sorted(matching.keys(), key=lambda m: np.median(matching[m]), reverse=True)

        fig = go.Figure()
        for i, model in enumerate(sorted_models):
            color = palette[i % len(palette)]
            fig.add_trace(go.Bar(
                x=x_vals[:len(series_dict[model])],
                y=series_dict[model],
                name=model,
                marker_color=color,
            ))

        fig.update_layout(
            title=f"Latency per Response \u2014 {config}",
            xaxis_title="Response index",
            yaxis_title="Latency (seconds)",
            template="plotly_white",
            font=dict(size=18),
            barmode="overlay",
            legend=dict(orientation="h", yanchor="top", y=-0.25, xanchor="center", x=0.5, title=""),
            margin=dict(b=120),
        )

        out = os.path.join(output_dir, f"bar_chart_{config}.pdf")
        fig.write_image(out)
        print(f"  Saved: {out}")


def load_metrics(models, configs):
    """Load RestBench metrics from results/<model>/<config>/metrics.json."""
    metrics = {}
    for model in models:
        for config in configs:
            path = os.path.join(RESULTS_DIR, model, config, "metrics.json")
            if os.path.exists(path):
                with open(path) as f:
                    metrics[(model, config)] = json.load(f)
    return metrics


def print_summary_table(data, output_dir, restbench_metrics=None):
    """Print a markdown summary table and save as text."""
    models = sorted(data.keys())
    configs = sorted({c for m in data for c in data[m]})

    lines = []
    lines.append(f"{'Model':<12} {'Config':<10} {'Count':>6} {'Avg(s)':>8} {'p50(s)':>8} {'p95(s)':>8} {'S%':>6} {'CP%':>6} {'\u0394SL':>6}")
    lines.append("-" * 74)

    for model in models:
        for config in configs:
            if config in data[model]:
                lat = np.array(data[model][config])
                n = len(lat)
                avg = np.mean(lat)
                p50 = np.percentile(lat, 50)
                p95 = np.percentile(lat, 95)

                s_val = cp_val = dsl_val = "\u2014"
                if restbench_metrics and (model, config) in restbench_metrics:
                    m = restbench_metrics[(model, config)]
                    s_val = f"{m['S']}%"
                    cp_val = f"{m['CP']}%"
                    dsl_val = f"{m['DeltaSL']}"

                lines.append(f"{model:<12} {config:<10} {n:>6} {avg:>8.3f} {p50:>8.3f} {p95:>8.3f} {s_val:>6} {cp_val:>6} {dsl_val:>6}")

    table = "\n".join(lines)
    print(f"\n{'='*74}")
    print("  Benchmark Summary \u2014 Latency + RestBench Metrics")
    print(f"{'='*74}")
    print(table)

    out = os.path.join(output_dir, "summary_table.txt")
    with open(out, "w") as f:
        f.write(table + "\n")
    print(f"  Saved: {out}")


def filter_data(data, models_filter, configs_filter):
    if models_filter:
        data = {m: data[m] for m in models_filter if m in data}
    if configs_filter:
        data = {m: {c: v for c, v in configs.items() if c in configs_filter}
                for m, configs in data.items()}
        data = {m: v for m, v in data.items() if v}
    return data


def main():
    args = parse_args()

    data = discover_results()
    if not data:
        print(f"No results found in {RESULTS_DIR}")
        print("Expected: results/<model>/dist-<N>/baseline/latencies.csv")
        sys.exit(1)

    print(f"Discovered models: {list(data.keys())}")
    print(f"Discovered configs: {sorted({c for m in data for c in data[m]})}")

    data = filter_data(data, args.models, args.configs)

    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)

    print_qos(data)

    restbench_metrics = load_metrics(list(data.keys()), sorted({c for m in data for c in data[m]}))

    plot_latency_curves(data, output_dir)
    plot_boxplot(data, output_dir)
    plot_bar_chart(data, output_dir)
    print_summary_table(data, output_dir, restbench_metrics)

    print(f"\nAll plots saved to {output_dir}/")


if __name__ == "__main__":
    main()

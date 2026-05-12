import json
import csv
import re
from urllib.parse import urlparse
import unicodedata

def extract_method_and_paths(oracle_string):
    """Extract method and path from a multiline Oracle string."""
    if not oracle_string:
        return []
    
    cleaned = oracle_string.replace('"', '').replace('\\n', '\n').replace('\\r', '\r')
    lines = cleaned.strip().splitlines()
    
    result = []
    for line in lines:
        parts = line.strip().split(maxsplit=1)
        if len(parts) == 2:
            method = parts[0].strip().upper()
            path = parts[1].strip().rstrip('/')
            result.append((method, path))
    return result

def extract_method_and_path_from_task(task):
    method = task.get("operation")
    endpoint = task.get("endpoint", "")
    parsed = urlparse(endpoint)
    path = parsed.path.rstrip('/')
    return method, path

def normalize_question(q):
    q = q.strip().lower()
    q = unicodedata.normalize("NFKD", q)
    q = re.sub(r"[‘’´`]", "'", q)
    q = re.sub(r"[“”]", '"', q)
    q = re.sub(r"\s+", " ", q)
    return q

def load_json_questions(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_csv_oracles(csv_path):
    oracle_map = {}
    with open(csv_path, 'r', newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            question_key = next((k for k in row if 'question' in k.lower()), None)
            oracle_key = next((k for k in row if 'oracle' in k.lower()), None)
            if question_key and oracle_key:
                question = normalize_question(row[question_key])
                oracle_raw = row.get(oracle_key)
                oracle_map[question] = extract_method_and_paths(oracle_raw)
    return oracle_map

def compare(json_data, oracle_map):
    results = []

    for item in json_data:
        question = normalize_question(item["question"])
        tasks = item.get("execution_plan", {}).get("tasks", [])
        oracle_steps = oracle_map.get(question, [])

        matches = []
        mismatches = []
        matched_indices = set()

        # Matching logic
        for task in tasks:
            task_method, task_path = extract_method_and_path_from_task(task)

            match_found = False
            for idx, (oracle_method, oracle_path) in enumerate(oracle_steps):
                if idx in matched_indices:
                    continue
                if task_method == oracle_method and task_path.endswith(oracle_path):
                    matches.append({"task": (task_method, task_path), "oracle": (oracle_method, oracle_path)})
                    matched_indices.add(idx)
                    match_found = True
                    break

            if not match_found:
                mismatches.append({"task": (task_method, task_path), "oracle": None})

        # Status assignment
        status = "correct"
        if matches and mismatches:
            status = "partial"
        elif not matches:
            status = "incorrect"

        # === METRICHE ===
        tp = len(matches)
        fp = len(mismatches)
        fn = max(0, len(oracle_steps) - tp)
        total = tp + fp + fn

        precision = tp / (tp + fp) if (tp + fp) else 0
        recall = tp / (tp + fn) if (tp + fn) else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

        accuracy = tp / total if total > 0 else 0
        overprediction_rate = fp / len(tasks) if len(tasks) else 0
        underprediction_rate = fn / len(oracle_steps) if len(oracle_steps) else 0
        jaccard = tp / (tp + fp + fn) if (tp + fp + fn) else 0
        coverage = tp / len(oracle_steps) if len(oracle_steps) else 0
        error_ratio = (fp + fn)

        comparison = {
            "question_index": item["question_index"],
            "question": item["question"],
            "matches": matches,
            "mismatches": mismatches,
            "missing_in_oracle": len(tasks) > len(oracle_steps),
            "extra_in_oracle": len(oracle_steps) > len(tasks),
            "status": status,
            "total_tasks": len(tasks),
            "matched_tasks": len(matches),

            # New metrics
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "accuracy": accuracy,
            "jaccard": jaccard,
            "coverage": coverage,
            "overpred": overprediction_rate,
            "underpred": underprediction_rate,
            "error_ratio": error_ratio,

            "tp": tp,
            "fp": fp,
            "fn": fn
        }

        results.append(comparison)

    return results

def write_detailed_output(results, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(f"--- Question #{result['question_index']} ---\n")
            f.write(f"Q: {result['question']}\n")

            if result["matches"]:
                f.write("Matches:\n")
                for match in result["matches"]:
                    task = match["task"]
                    oracle = match["oracle"]
                    f.write(f"  - Task:   {task[0]} {task[1]}\n")
                    f.write(f"    Oracle: {oracle[0]} {oracle[1]}\n")

            if result["mismatches"]:
                f.write("Mismatches:\n")
                for mm in result["mismatches"]:
                    task = mm["task"]
                    f.write(f"  - Task:   {task[0]} {task[1]}\n")
                    f.write("    Oracle: None\n")

            # Metrics
            f.write("\nMetrics:\n")
            f.write(f"Precision: {result['precision']:.3f}\n")
            f.write(f"Recall:    {result['recall']:.3f}\n")
            f.write(f"F1-score:  {result['f1']:.3f}\n")
            f.write(f"Accuracy:  {result['accuracy']:.3f}\n")
            f.write(f"Jaccard:   {result['jaccard']:.3f}\n")
            f.write(f"Coverage:  {result['coverage']:.3f}\n")
            f.write(f"Overprediction rate: {result['overpred']:.3f}\n")
            f.write(f"Underprediction rate: {result['underpred']:.3f}\n")
            f.write(f"Errors (FP+FN): {result['error_ratio']}\n")
            f.write(f"TP: {result['tp']}, FP: {result['fp']}, FN: {result['fn']}\n")

            f.write(f"Status: {result['status'].upper()}\n\n")

def write_summary_output(results, filename):
    total = len(results)
    correct = sum(1 for r in results if r['status'] == 'correct')
    partial = sum(1 for r in results if r['status'] == 'partial')
    incorrect = sum(1 for r in results if r['status'] == 'incorrect')

    total_endpoints = sum(r['total_tasks'] for r in results)
    matched_endpoints = sum(r['matched_tasks'] for r in results)

    # Global confusion values
    total_tp = sum(r["tp"] for r in results)
    total_fp = sum(r["fp"] for r in results)
    total_fn = sum(r["fn"] for r in results)

    # Micro averaging
    micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0
    micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0
    micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall) if (micro_precision + micro_recall) else 0

    # Macro averaging
    macro_precision = sum(r["precision"] for r in results) / total
    macro_recall = sum(r["recall"] for r in results) / total
    macro_f1 = sum(r["f1"] for r in results) / total

    # Global metrics
    global_accuracy = total_tp / (total_tp + total_fp + total_fn) if (total_tp + total_fp + total_fn) else 0
    global_jaccard = total_tp / (total_tp + total_fp + total_fn) if (total_tp + total_fp + total_fn) else 0

    with open(filename, 'w', encoding='utf-8') as f:

        f.write("Summary Report\n")
        f.write("=================\n")
        f.write(f"Total execution plans: {total}\n")
        f.write(f"Correct plans: {correct}\n")
        f.write(f"Partially correct plans: {partial}\n")
        f.write(f"Incorrect plans: {incorrect}\n")

        f.write(f"\nTotal endpoints in all plans: {total_endpoints}\n")
        f.write(f"Matching endpoints: {matched_endpoints}\n")

        f.write(f"\nPercentages:\n")
        f.write(f"- Correct: {correct / total * 100:.2f}%\n")
        f.write(f"- Partial: {partial / total * 100:.2f}%\n")
        f.write(f"- Incorrect: {incorrect / total * 100:.2f}%\n")
        f.write(f"- Endpoint match rate: {matched_endpoints / total_endpoints * 100:.2f}%\n")

        # --- NEW METRICS ---
        f.write("\nGlobal Metrics:\n")
        f.write(f"Total TP: {total_tp}\n")
        f.write(f"Total FP: {total_fp}\n")
        f.write(f"Total FN: {total_fn}\n\n")

        f.write(f"Micro Precision: {micro_precision:.3f}\n")
        f.write(f"Micro Recall:    {micro_recall:.3f}\n")
        f.write(f"Micro F1:        {micro_f1:.3f}\n\n")

        f.write(f"Macro Precision: {macro_precision:.3f}\n")
        f.write(f"Macro Recall:    {macro_recall:.3f}\n")
        f.write(f"Macro F1:        {macro_f1:.3f}\n\n")

        f.write(f"Global Accuracy: {global_accuracy:.3f}\n")
        f.write(f"Global Jaccard:  {global_jaccard:.3f}\n")

# === CONFIGURATION ===
BASE_PATH = "smart-city-results/test-with-reranker-roles"
json_file = f"{BASE_PATH}/execution_plans.json"
csv_file = "smart-city-requests/requests_roles.csv"
detailed_output = f"{BASE_PATH}/request_oracle_details.txt"
summary_output = f"{BASE_PATH}/request_oracle_summary.txt"

# === EXECUTION ===
json_data = load_json_questions(json_file)
oracle_map = load_csv_oracles(csv_file)
results = compare(json_data, oracle_map)
write_detailed_output(results, detailed_output)
write_summary_output(results, summary_output)

print("Analysis completed. Output saved to:")
print(f"  - {detailed_output}")
print(f"  - {summary_output}")

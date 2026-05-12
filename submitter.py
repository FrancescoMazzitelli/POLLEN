import sys
import requests
import pandas as pd
from collections import Counter

# =================== LOGGER SETUP =================== #
class Logger:
    def __init__(self, filename):
        self.terminal_stdout = sys.stdout
        self.terminal_stderr = sys.stderr
        self.log = open(filename, "w", encoding="utf-8")

    def write(self, message):
        self.terminal_stdout.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal_stdout.flush()
        self.log.flush()

class ErrorLogger:
    def __init__(self, logger):
        self.terminal_stderr = sys.stderr
        self.logger = logger

    def write(self, message):
        self.terminal_stderr.write(message)
        self.logger.log.write(message)

    def flush(self):
        self.terminal_stderr.flush()
        self.logger.log.flush()



logger = Logger("test.txt")
sys.stdout = logger
sys.stderr = ErrorLogger(logger)

# =================== SCRIPT START =================== #
try:
    http_response = requests.post("http://localhost:7500/api/importer/import")
except Exception as e:
    print(f"Error importing external servces: {e}")

# Load questions
df = pd.read_csv(r"C:\Universita\Tesi\LADARAG-Extended-Rita\test_validation\requests_cross_service_test.csv")
questions = df["Questions"].tolist()

# Initialize counters
total = len(questions)
success_count = 0
failure_count = 0
responses = []

# Task-level counters
task_status_counter = Counter()
task_name_counter = Counter()
task_name_failure_counter = Counter()

# Detailed failure log
failed_tasks_details = []
execution_plans_log = []

for idx, question in enumerate(questions):
    try:
        response = requests.post("http://localhost:5500/api/control/invoke", json={"input": question})
        data = response.json()
        responses.append(data)

        execution_plan = data.get("execution_plan", {})
        tasks = data.get("execution_results", [])

        print(f"\n[{idx+1}] Question: {question}")
        print("[EXECUTION PLAN]")
        for task in execution_plan.get("tasks", []):
            print(f"  - Task: {task.get('task_name')}")
            print(f"    - Service ID: {task.get('service_id')}")
            print(f"    - Endpoint:   {task.get('endpoint')}")
            print(f"    - Operation:  {task.get('operation')}")
            print(f"    - Input:      {task.get('input')}")

        # Store execution plan
        execution_plans_log.append({
            "question_index": idx + 1,
            "question": question,
            "execution_plan": execution_plan
        })

        # Task status check
        task_statuses = [task.get("status") for task in tasks]

        if all(status == "SUCCESS" for status in task_statuses):
            success_count += 1
            print(f"[{idx+1}] SUCCESS")
            print(tasks)
        else:
            failure_count += 1
            print(f"[{idx+1}] FAILED - At least one task was not successful")
            print(tasks)

        # Per-task statistics
        for task in tasks:
            status = task.get("status", "UNKNOWN")
            name = task.get("task_name", "UNKNOWN")

            task_status_counter[status] += 1
            task_name_counter[name] += 1

            if status != "SUCCESS":
                task_name_failure_counter[name] += 1
                failed_tasks_details.append({
                    "question_index": idx + 1,
                    "question": question,
                    "task_name": name,
                    "status": status,
                    "status_code": task.get("status_code"),
                    "result": task.get("result")
                })

        print("=" * 100)

    except Exception as e:
        print(f"[{idx+1}] ERROR parsing response: {e}")
        failure_count += 1
        responses.append(None)
        print("=" * 100)

    except Exception as e:
        print(f"[{idx+1}] ERROR parsing response: {e}")
        failure_count += 1
        responses.append(None)
        print("=" * 100)

# Final summary
print("\nFINAL STATISTICS")
print("=" * 60)
print(f"Total questions: {total}")
print(f"Questions fully successful: {success_count}")
print(f"Questions with at least one failed/exception task: {failure_count}")

print("\nTask status counts:")
for status, count in task_status_counter.items():
    print(f"  {status}: {count}")

print("\nMost common tasks:")
for task, count in task_name_counter.most_common(10):
    print(f"  {task}: {count} total")

print("\nMost failing tasks:")
for task, count in task_name_failure_counter.most_common(10):
    print(f"  {task}: {count} failures")

# Percentages
total_tasks = sum(task_status_counter.values())
print("\nPERCENTAGES")
print("=" * 60)
print(f"Total tasks: {total_tasks}")
for status, count in task_status_counter.items():
    percentage = (count / total_tasks) * 100
    print(f"  {status}: {percentage:.2f}%")

# Export logs
pd.DataFrame(failed_tasks_details).to_csv("failed_tasks_log.csv", index=False)
pd.DataFrame(execution_plans_log).to_json("execution_plans.json", orient="records", indent=2)

print("\nFailed task details saved to 'failed_tasks_log.csv'")
print("Execution plans saved to 'execution_plans.json'")
print("=" * 60)

import os
import json
import mlflow
from mlflow.tracking import MlflowClient
from google import genai

MLFLOW_PATH = "/tmp/fed_mlruns"

def load_env():
    env_path = os.path.join(os.path.dirname(__file__), "../.env")
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, val = line.split("=", 1)
                    os.environ[key.strip()] = val.strip()

def fetch_mlflow_metrics():
    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
    mlflow.set_tracking_uri(f"file://{MLFLOW_PATH}")
    client = MlflowClient()

    experiments = [e for e in client.search_experiments() if e.name != "Default"]
    if not experiments:
        raise ValueError("No MLflow experiments found.")

    summary = {}
    for exp in experiments:
        runs = client.search_runs(experiment_ids=[exp.experiment_id])
        for run in runs:
            site = run.info.run_name
            if not site or site in summary:
                continue
            metrics = run.data.metrics
            history = {}
            for metric_name in ["accuracy", "loss"]:
                history[metric_name] = [
                    m.value for m in client.get_metric_history(run.info.run_id, metric_name)
                ]
            summary[site] = {
                "final_accuracy": metrics.get("accuracy", 0),
                "final_loss": metrics.get("loss", 0),
                "accuracy_history": history.get("accuracy", []),
                "loss_history": history.get("loss", []),
            }

    return summary

def build_prompt(metrics_summary: dict) -> str:
    metrics_json = json.dumps(metrics_summary, indent=2)
    return f"""You are an expert QC analyst at a manufacturing company running a federated learning system.

Three factories have just completed a federated defect detection training job.
Each factory trained locally on its own data and shared only model weights.

Here are the training metrics per factory across 5 rounds:

{metrics_json}

Please do the following:
1. Summarize overall federated training performance
2. Identify any factory that underperformed or showed unusual patterns
3. Flag any QC concerns (e.g. slow convergence, high final loss, accuracy gaps)
4. Give 2-3 concrete operational recommendations for the QC team
5. Rate overall system health: GREEN / YELLOW / RED

Be concise and practical. This report will be read by a factory operations manager.
"""

def run_qc_agent():
    load_env()
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in .env")

    print("Fetching MLflow metrics...")
    metrics = fetch_mlflow_metrics()
    print(f"Metrics loaded for sites: {list(metrics.keys())}")

    if not metrics:
        raise ValueError("No metrics found in MLflow.")

    print("\nSending to Gemini for QC analysis...\n")
    print("=" * 60)

    client = genai.Client(api_key=api_key)
    prompt = build_prompt(metrics)

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
    )

    print(response.text)
    print("=" * 60)

    report_path = os.path.join(os.path.dirname(__file__), "qc_report.txt")
    with open(report_path, "w") as f:
        f.write(response.text)
    print(f"\nReport saved to {report_path}")

if __name__ == "__main__":
    run_qc_agent()

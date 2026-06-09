import mlflow

def start_run(run_name: str, experiment_name: str = "federated-qc"):
    mlflow.set_experiment(experiment_name)
    mlflow.start_run(run_name=run_name)

def log_metrics(metrics: dict, step: int):
    mlflow.log_metrics(metrics, step=step)

def log_params(params: dict):
    mlflow.log_params(params)

def end_run():
    mlflow.end_run()

import sys
import os

# Absolute path to trainer
TRAINER_SCRIPT = os.path.abspath(os.path.join(os.path.dirname(__file__), "trainer.py"))

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from nvflare.app_opt.pt.job_config.fed_sag_mlflow import SAGMLFlowJob
from nvflare.app_opt.pt.in_process_client_api_executor import PTInProcessClientAPIExecutor
from model.cnn import DefectCNN

job = SAGMLFlowJob(
    name="defect_detection",
    initial_model=DefectCNN(),
    n_clients=3,
    num_rounds=5,
    key_metric="accuracy",
)

for i in range(1, 4):
    executor = PTInProcessClientAPIExecutor(
        task_script_path=TRAINER_SCRIPT,
        params_exchange_format="pytorch",
        server_expected_format="numpy",
    )
    job.to(executor, target=f"site-{i}", id="executor")

if __name__ == "__main__":
    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
    job.simulator_run(
        workspace="/tmp/flare_workspace",
    )

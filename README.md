# NVIDIA-Flare-Federated-Learning

# Federated Defect Detection with Agentic QC Insights

A federated learning project simulating 3 factories collaboratively training a defect detection model — without sharing raw data.

---

## Tech Stack

| Layer | Tool |
|---|---|
| Federated Learning | NVIDIA FLARE (simulator) |
| ML Model | PyTorch (CNN) |
| MLOps | MLflow |
| GenAI Agent | Gemini API |
| Cloud Storage | AWS S3 |

---

## Project Structure
federated-qc/
├── data/
│   ├── factory_1/          # 80 good, 40 defect images
│   ├── factory_2/          # 80 good, 40 defect images
│   ├── factory_3/          # 80 good, 40 defect images
│   └── generate_data.py    # Synthetic dataset generator
├── flare_job/
│   ├── app_server/         # FLARE server-side config
│   └── app_client/         # FLARE client-side config
├── model/
│   └── cnn.py              # Simple CNN definition
├── mlops/
│   └── mlflow_logger.py    # MLflow tracking helper
├── agent/
│   └── qc_agent.py         # Gemini agentic QC layer
├── cloud/
│   └── s3_upload.py        # AWS S3 upload script
└── requirements.txt

---

## Phases

- [x] Phase 1 — Environment setup (WSL2, packages, venv)
- [x] Phase 2 — Synthetic dataset generation + CNN model
- [ ] Phase 3 — NVIDIA FLARE federated training + MLflow tracking
- [ ] Phase 4 — Gemini agentic QC layer
- [ ] Phase 5 — AWS S3 global model upload

---

## Setup

### 1. Clone the repo
```bash
git clone https://github.com/mrityunjay2627/NVIDIA-Flare-Federated-Learning.git
cd NVIDIA-Flare-Federated-Learning
```

### 2. Create virtual environment (WSL/Linux only)
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install nvflare torch torchvision mlflow google-genai boto3 Pillow scikit-learn
```

### 4. Generate synthetic dataset
```bash
python data/generate_data.py
```

### 5. Verify CNN
```bash
python model/cnn.py
```

---

## Environment Variables

Create a `.env` file in the root (never commit this):
GEMINI_API_KEY=your_gemini_api_key_here
AWS_ACCESS_KEY_ID=your_aws_key_here
AWS_SECRET_ACCESS_KEY=your_aws_secret_here
AWS_BUCKET_NAME=your_bucket_name_here
AWS_REGION=us-east-1

---

## Notes

- NVIDIA FLARE does not support Windows natively — use WSL2
- Dataset is synthetically generated to simulate real manufacturing defects (scratches, cracks, blobs)
- All 3 factories train locally; only model weights are shared with the FLARE server

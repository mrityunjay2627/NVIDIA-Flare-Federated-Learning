# Federated Defect Detection with Agentic QC Insights

**A federated learning system simulating 3 factories collaboratively training a defect detection model — without sharing raw data — with a Gemini-powered agentic QC layer.**

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![NVIDIA FLARE](https://img.shields.io/badge/NVIDIA%20FLARE-2.7.2-green.svg)](https://github.com/NVIDIA/NVFlare)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Understanding the Phases](#understanding-the-phases)
- [Project Structure](#project-structure)
- [Results](#results)
- [Contact](#contact)

---

## Overview

This project demonstrates a full federated learning pipeline for manufacturing defect detection across 3 simulated factories. Each factory trains a CNN locally on its own defect image data and shares only model weights with a central FLARE server — never raw data. After training, a Gemini-powered agentic QC layer reads MLflow metrics and generates operational recommendations for factory managers. The final global model is uploaded to AWS S3.

**Core idea**: Privacy-preserving collaborative AI for industrial quality control.

---

## Key Features

- **Federated Learning**: 3 factory clients train locally, FLARE server aggregates via FedAvg
- **No Raw Data Sharing**: Only model weights leave each factory
- **MLOps Tracking**: Per-round, per-factory metrics logged to MLflow
- **Agentic QC Layer**: Gemini 2.5 Flash reads real metrics and generates QC reports
- **Cloud Integration**: Global model uploaded to AWS S3 after training
- **Synthetic Dataset**: Realistic defect images (scratches, cracks, blobs) generated programmatically

---

## Architecture

```
┌─────────────────────────────────────────────────┐
│              NVIDIA FLARE Simulator              │
│                                                  │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐       │
│  │Factory 1 │  │Factory 2 │  │Factory 3 │       │
│  │ CNN Train│  │ CNN Train│  │ CNN Train│       │
│  │ (local)  │  │ (local)  │  │ (local)  │       │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘       │
│       │  weights    │  weights    │              │
│       └─────────────┴─────────────┘              │
│                     │                            │
│              ┌──────▼──────┐                     │
│              │  FL Server  │  FedAvg             │
│              │ Aggregation │  (global model)     │
│              └──────┬──────┘                     │
└─────────────────────┼───────────────────────────┘
                      │
          ┌───────────┼───────────┐
          ▼           ▼           ▼
    ┌──────────┐ ┌─────────┐ ┌────────┐
    │  MLflow  │ │ Gemini  │ │ AWS S3 │
    │ Tracking │ │QC Agent │ │ Model  │
    │(metrics) │ │(report) │ │Storage │
    └──────────┘ └─────────┘ └────────┘
```

**Three-Layer Design:**
- **Layer 1 (Federation)**: NVIDIA FLARE simulator runs 3 factory clients + 1 server locally
- **Layer 2 (MLOps)**: MLflow tracks loss and accuracy per round per factory
- **Layer 3 (Agentic AI)**: Gemini reads metrics and outputs operational QC recommendations

---

## Tech Stack

| Layer | Tool |
|---|---|
| Federated Learning | NVIDIA FLARE 2.7.2 |
| ML Model | PyTorch CNN |
| MLOps | MLflow |
| GenAI Agent | Gemini 2.5 Flash |
| Cloud Storage | AWS S3 (boto3) |
| Environment | WSL2 + Python venv |

---

## Installation

### Prerequisites

- Windows 10/11 with WSL2 (Ubuntu)
- Python 3.11+
- Google Gemini API key
- AWS account with S3 bucket

### Setup Steps

```bash
# 1. Clone the repository
git clone https://github.com/mrityunjay2627/NVIDIA-Flare-Federated-Learning.git
cd NVIDIA-Flare-Federated-Learning

# 2. Create virtual environment (WSL only)
python3 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install nvflare torch torchvision mlflow google-genai boto3 Pillow scikit-learn

# 4. Generate synthetic dataset
python data/generate_data.py

# 5. Verify CNN
python model/cnn.py
```

### Environment Variables

Create a `.env` file in the project root (never commit this):

```
GEMINI_API_KEY=your_gemini_api_key_here
AWS_ACCESS_KEY_ID=your_aws_key_here
AWS_SECRET_ACCESS_KEY=your_aws_secret_here
AWS_BUCKET_NAME=your_bucket_name_here
AWS_REGION=us-east-1
```

---

## Quick Start

### Step 1 — Run Federated Training

```bash
rm -rf /tmp/flare_workspace2 /tmp/fed_mlruns
python flare_job/job.py
```

### Step 2 — View MLflow Metrics

```bash
MLFLOW_ALLOW_FILE_STORE=true mlflow ui --host 0.0.0.0 --port 5000 \
  --backend-store-uri /tmp/fed_mlruns
```

Open [http://localhost:5000](http://localhost:5000) in your browser.

### Step 3 — Run Gemini QC Agent

```bash
python agent/qc_agent.py
```

### Step 4 — Upload Global Model to S3

```bash
python cloud/s3_upload.py
```

---

## Understanding the Phases

### Phase 1: Environment Setup ✅

WSL2, Python virtual environment, all dependencies installed and verified.

---

### Phase 2: Dataset + CNN Model ✅

**What it is**: Synthetic defect image dataset generated across 3 factory folders. A simple 3-layer CNN trained to classify images as `good` or `defect`.

**Dataset per factory**:
- 80 good images (clean textured surface)
- 40 defect images (scratches, cracks, blobs)

**CNN Architecture**:
```
Input (3×224×224)
→ Conv2d(3→16) + ReLU + MaxPool
→ Conv2d(16→32) + ReLU + MaxPool
→ Conv2d(32→64) + ReLU + MaxPool
→ Flatten → Linear(64×28×28 → 128) → ReLU → Dropout(0.3)
→ Linear(128 → 2)  [good / defect]
```

---

### Phase 3: FLARE Federated Training + MLflow ✅

**What it is**: NVIDIA FLARE simulator runs all 3 factory clients locally. Each client loads its factory's data, trains the CNN for one round, and sends updated weights back to the server. The server aggregates using FedAvg and broadcasts the improved global model for the next round.

**Training results (5 rounds)**:

| Round | Factory 1 | Factory 2 | Factory 3 |
|---|---|---|---|
| 0 | ~55% | ~50% | ~60% |
| 1 | ~66% | ~68% | ~68% |
| 2 | ~78% | ~68% | ~71% |
| 3 | ~93% | ~93% | ~95% |
| 4 | **99.2%** | **97.5%** | **97.5%** |

MLflow tracks loss and accuracy per round per factory.

---

### Phase 4: Gemini Agentic QC Layer ✅

**What it is**: After training, `agent/qc_agent.py` reads real MLflow metrics and sends them to Gemini 2.5 Flash with a QC analyst prompt. Gemini returns a structured operational report.

**Example output**:
```
QC Report: Federated Defect Detection Training

1. Overall Performance: All factories achieved >97% accuracy after 5 rounds.

2. Factory Analysis:
   - Factory 1: Best performer — 99.2% accuracy, loss 0.051
   - Factory 2: Flagged — higher final loss (0.111), uneven convergence
   - Factory 3: Solid — 97.5% accuracy, steady convergence

3. QC Concerns:
   - Factory 2 loss is 2x higher than Factory 1
   - Plateau observed in Factory 2 between rounds 2-3

4. Recommendations:
   - Investigate Factory 2 local data quality
   - Review Factory 2 training logs for rounds 2-3
   - Establish baseline loss targets per factory

5. System Health: YELLOW
```

Report saved to `agent/qc_report.txt`.

---

### Phase 5: AWS S3 Model Upload ✅

**What it is**: After training, FLARE saves the best global model as `FL_global_model.pt`. The upload script finds it and pushes it to AWS S3 with a timestamp.

**S3 path format**: `s3://<bucket>/federated-qc/FL_global_model_YYYYMMDD_HHMMSS.pt`

---

## Project Structure

```
federated-qc/
├── data/
│   ├── factory_1/              # 80 good, 40 defect images
│   ├── factory_2/              # 80 good, 40 defect images
│   ├── factory_3/              # 80 good, 40 defect images
│   └── generate_data.py        # Synthetic dataset generator
│
├── flare_job/
│   ├── job.py                  # FLARE Job API — defines federation
│   └── trainer.py              # Client trainer — local CNN training loop
│
├── model/
│   └── cnn.py                  # DefectCNN architecture
│
├── mlops/
│   └── mlflow_logger.py        # MLflow helper
│
├── agent/
│   ├── qc_agent.py             # Gemini agentic QC layer
│   └── qc_report.txt           # Latest generated QC report
│
├── cloud/
│   ├── s3_upload.py            # AWS S3 upload script
│   └── upload_record.txt       # Latest upload record
│
├── .env                        # API keys (never commit)
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Results

| Phase | Status | Key Output |
|---|---|---|
| Environment Setup | ✅ Complete | WSL2 + venv + all packages |
| Dataset + CNN | ✅ Complete | 360 synthetic images, CNN verified |
| Federated Training | ✅ Complete | 99.2% accuracy, 5 rounds, 3 factories |
| Gemini QC Agent | ✅ Complete | Real metrics → operational QC report |
| AWS S3 Upload | ✅ Complete | Global model stored in S3 |

---

## Notes

- NVIDIA FLARE does not support Windows natively — WSL2 is required
- Dataset is synthetically generated to simulate real manufacturing defects
- All 3 factories train locally; only model weights are shared with the FLARE server
- MLflow file store requires `MLFLOW_ALLOW_FILE_STORE=true` with MLflow 3.x

---

## Contact

**Priyanshu M Sharma** — MS Robotics & Autonomous Systems (AI), Arizona State University

---

## License

MIT License — See LICENSE file for details

---

**Built for the Industrial AI community**

import os
import boto3
from pathlib import Path
from datetime import datetime


def load_env():
    env_path = os.path.join(os.path.dirname(__file__), "../.env")
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, val = line.split("=", 1)
                    os.environ[key.strip()] = val.strip()


def find_global_model():
    search_dirs = [
        "/tmp/flare_workspace2",
        "/tmp/flare_workspace",
    ]
    candidates = []
    for base in search_dirs:
        for path in Path(base).rglob("FL_global_model.pt"):
            candidates.append(path)

    if not candidates:
        raise FileNotFoundError("FL_global_model.pt not found in FLARE workspace.")

    latest = max(candidates, key=lambda p: p.stat().st_mtime)
    print(f"Found model: {latest}")
    return latest


def upload_to_s3(model_path: Path):
    bucket = os.environ.get("AWS_BUCKET_NAME")
    region = os.environ.get("AWS_REGION", "us-east-1")

    if not bucket:
        raise ValueError("AWS_BUCKET_NAME not set in .env")

    s3 = boto3.client(
        "s3",
        aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
        region_name=region,
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    s3_key = f"federated-qc/FL_global_model_{timestamp}.pt"

    print(f"Uploading to s3://{bucket}/{s3_key} ...")
    s3.upload_file(str(model_path), bucket, s3_key)
    print(f"Upload complete.")
    print(f"S3 URI: s3://{bucket}/{s3_key}")

    return s3_key


def run():
    load_env()
    model_path = find_global_model()
    s3_key = upload_to_s3(model_path)

    record_path = os.path.join(os.path.dirname(__file__), "upload_record.txt")
    with open(record_path, "w") as f:
        f.write(f"Model: {model_path}\n")
        f.write(f"S3 Key: {s3_key}\n")
        f.write(f"Bucket: {os.environ.get('AWS_BUCKET_NAME')}\n")
    print(f"Record saved to {record_path}")


if __name__ == "__main__":
    run()

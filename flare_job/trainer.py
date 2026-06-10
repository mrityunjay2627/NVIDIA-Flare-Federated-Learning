import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from pathlib import Path
from PIL import Image
import mlflow
import nvflare.client as flare
from model.cnn import DefectCNN

MLFLOW_DIR = "/tmp/fed_mlruns"
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
mlflow.set_tracking_uri(f"file://{MLFLOW_DIR}")
mlflow.set_experiment("federated-qc")

class FactoryDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.samples = []
        self.transform = transform
        for label, class_name in enumerate(["good", "defect"]):
            class_dir = Path(data_dir) / class_name
            for img_path in class_dir.glob("*.png"):
                self.samples.append((img_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


flare.init()

site_name = flare.get_site_name()
factory_id = site_name.split("-")[1]
data_dir = Path(__file__).parent.parent / "data" / f"factory_{factory_id}"

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

dataset = FactoryDataset(data_dir, transform=transform)
loader = DataLoader(dataset, batch_size=16, shuffle=True)

device = torch.device("cpu")
model = DefectCNN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

with mlflow.start_run(run_name=f"factory_{factory_id}"):
    mlflow.log_params({"factory": factory_id, "lr": 1e-3, "batch_size": 16})

    while flare.is_running():
        input_model = flare.receive()
        round_num = input_model.current_round

        model.load_state_dict(input_model.params)
        model.train()

        total_loss, correct, total = 0.0, 0, 0

        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

        accuracy = correct / total
        avg_loss = total_loss / len(loader)

        print(f"[Factory {factory_id}] Round {round_num} — Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")

        mlflow.log_metrics({"loss": avg_loss, "accuracy": accuracy}, step=round_num)

        flare.send(flare.FLModel(
            params=model.state_dict(),
            meta={"num_samples": total}
        ))

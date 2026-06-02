import torch
import torch.nn as nn

class DefectCNN(nn.Module):
    def __init__(self):
        super(DefectCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),  # 3 channels in, 16 out
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                           # halve spatial dims

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 28 * 28, 128),  # input images resized to 224x224
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2)              # 2 classes: good, defect
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    model = DefectCNN()
    dummy = torch.randn(4, 3, 224, 224)   # batch of 4 images
    out = model(dummy)
    print(f"Output shape: {out.shape}")   # should be [4, 2]
    print("CNN looks good!")

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import os

# Data setup

eval_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

test_dataset = datasets.ImageFolder("fish_data/test", transform=eval_transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

class_names = test_dataset.classes
num_classes = len(class_names)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# CNN model definition (same architecture used for both models)

class FishCNN(nn.Module):
    def __init__(self, num_classes, dropout_rate=0.3):
        super(FishCNN, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 16 * 16, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

# Function to get predictions from a model

def get_predictions(model, loader):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return all_labels, all_preds

# Load and evaluate baseline model

print("\n--- Evaluating Baseline Model ---")
baseline_model = FishCNN(num_classes, dropout_rate=0.3).to(device)
baseline_model.load_state_dict(torch.load("output/baseline_model.pth", map_location=device))

baseline_labels, baseline_preds = get_predictions(baseline_model, test_loader)

baseline_report = classification_report(
    baseline_labels, baseline_preds, target_names=class_names, digits=4
)
print(baseline_report)

# Load and evaluate optimized model

print("\n--- Evaluating Optimized Model ---")
optimized_model = FishCNN(num_classes, dropout_rate=0.3).to(device)
optimized_model.load_state_dict(torch.load("output/optimized_model.pth", map_location=device))

optimized_labels, optimized_preds = get_predictions(optimized_model, test_loader)

optimized_report = classification_report(
    optimized_labels, optimized_preds, target_names=class_names, digits=4
)
print(optimized_report)

# Save both reports to a text file

os.makedirs("output", exist_ok=True)

with open("output/classification_reports.txt", "w") as f:
    f.write("BASELINE MODEL - Test Set Classification Report\n")
    f.write("=" * 60 + "\n")
    f.write(baseline_report)
    f.write("\n\n")
    f.write("OPTIMIZED MODEL - Test Set Classification Report\n")
    f.write("=" * 60 + "\n")
    f.write(optimized_report)

print("\nClassification reports saved to output/classification_reports.txt")

# Confusion matrix for optimized model

cm = confusion_matrix(optimized_labels, optimized_preds)

fig, ax = plt.subplots(figsize=(9, 8))
im = ax.imshow(cm, cmap="Blues")

ax.set_xticks(np.arange(num_classes))
ax.set_yticks(np.arange(num_classes))
ax.set_xticklabels(class_names, rotation=45, ha="right")
ax.set_yticklabels(class_names)
ax.set_xlabel("Predicted Label")
ax.set_ylabel("True Label")
ax.set_title("Confusion Matrix - Optimized Model (Test Set)")

for i in range(num_classes):
    for j in range(num_classes):
        ax.text(j, i, cm[i, j], ha="center", va="center",
                 color="white" if cm[i, j] > cm.max() / 2 else "black")

plt.colorbar(im)
plt.tight_layout()
plt.savefig("output/confusion_matrix.png")
print("Confusion matrix saved to output/confusion_matrix.png")
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import json
import os

train_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

eval_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

train_dataset = datasets.ImageFolder("fish_data/train", transform=train_transform)
val_dataset = datasets.ImageFolder("fish_data/val", transform=eval_transform)

class_names = train_dataset.classes
num_classes = len(class_names)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

class FishCNN(nn.Module):
    def __init__(self, num_classes, dropout_rate):
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

def train_and_evaluate(learning_rate, batch_size, dropout_rate, num_epochs=3):

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = FishCNN(num_classes, dropout_rate).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            val_correct += (predicted == labels).sum().item()
            val_total += labels.size(0)

    avg_val_loss = val_loss / val_total
    val_accuracy = val_correct / val_total

    return avg_val_loss, val_accuracy

configs = [
    {"learning_rate": 0.01,   "batch_size": 32, "dropout_rate": 0.3},
    {"learning_rate": 0.001,  "batch_size": 32, "dropout_rate": 0.3},
    {"learning_rate": 0.0001, "batch_size": 32, "dropout_rate": 0.3},
    {"learning_rate": 0.001,  "batch_size": 64, "dropout_rate": 0.3},
    {"learning_rate": 0.001,  "batch_size": 32, "dropout_rate": 0.5},
    {"learning_rate": 0.001,  "batch_size": 64, "dropout_rate": 0.5},
]

results = []

for i, config in enumerate(configs):
    print(f"\nTesting config {i+1}/{len(configs)}: {config}")

    val_loss, val_acc = train_and_evaluate(
        learning_rate=config["learning_rate"],
        batch_size=config["batch_size"],
        dropout_rate=config["dropout_rate"]
    )

    print(f"Result -> Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

    results.append({
        "config": config,
        "val_loss": val_loss,
        "val_accuracy": val_acc
    })

best_result = min(results, key=lambda r: r["val_loss"])

print("\n\nAll results:")
for r in results:
    print(r)

print("\nBest configuration (lowest validation loss):")
print(best_result)

os.makedirs("output", exist_ok=True)
with open("output/hyperparameter_results.json", "w") as f:
    json.dump({"all_results": results, "best_result": best_result}, f, indent=2)

print("\nResults saved to output/hyperparameter_results.json")
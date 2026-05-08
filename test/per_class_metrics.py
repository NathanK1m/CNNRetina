from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import densenet121
from sklearn.metrics import precision_score, recall_score, f1_score


DATA_ROOT = Path(r"F:\College\DataScience\CNNRetina\data\preprocesseddata\test")
MODEL_PATH = Path(r"F:\College\DataScience\CNNRetina\cnn_oct_model3.pth")

BATCH_SIZE = 32
DEVICE = torch.device("cuda")


def get_test_loader():
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    test_dataset = datasets.ImageFolder(DATA_ROOT, transform=transform)
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        pin_memory=True
    )

    return test_loader, test_dataset.classes


def load_model(num_classes):
    model = densenet121(weights=None)
    model.classifier = nn.Linear(model.classifier.in_features, num_classes)

    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(DEVICE)
    model.eval()

    return model


def get_predictions(model, loader):
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return np.array(all_labels), np.array(all_preds)


def plot_per_class_metrics(labels, preds, class_names):
    precision = precision_score(labels, preds, average=None, zero_division=0)
    recall = recall_score(labels, preds, average=None, zero_division=0)
    f1 = f1_score(labels, preds, average=None, zero_division=0)

    x = np.arange(len(class_names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.bar(x - width, precision, width, label="Precision")
    ax.bar(x, recall, width, label="Recall")
    ax.bar(x + width, f1, width, label="F1")

    ax.set_xlabel("Class")
    ax.set_ylabel("Score")
    ax.set_title("Per-Class Precision, Recall, and F1 Score")
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_ylim(0.85, 1.0)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig("per_class_metrics.png", dpi=150)
    plt.show()
    print("Saved per_class_metrics.png")


def main():
    test_loader, class_names = get_test_loader()
    model = load_model(len(class_names))
    labels, preds = get_predictions(model, test_loader)
    plot_per_class_metrics(labels, preds, class_names)


if __name__ == "__main__":
    main()

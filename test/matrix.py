from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import densenet121
from sklearn.metrics import confusion_matrix, classification_report


DATA_ROOT = Path(r"F:\College\DataScience\CNNRetina\data\preprocesseddata\test")
MODEL_PATH = Path(r"F:\College\DataScience\CNNRetina\cnn_oct_model.pth")

BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_test_loader():
    test_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    test_dataset = datasets.ImageFolder(DATA_ROOT, transform=test_transform)
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


def plot_confusion_matrix(labels, preds, class_names):
    cm = confusion_matrix(labels, preds)
    num_classes = len(class_names)

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, cmap="Blues")

    ax.set_xticks(range(num_classes))
    ax.set_yticks(range(num_classes))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)

    for i in range(num_classes):
        for j in range(num_classes):
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color=color)

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix - DenseNet121 OCT Classification")
    fig.colorbar(im)

    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150)
    plt.show()


def main():
    test_loader, class_names = get_test_loader()
    model = load_model(num_classes=len(class_names))
    labels, preds = get_predictions(model, test_loader)

    print(classification_report(labels, preds, target_names=class_names))

    plot_confusion_matrix(labels, preds, class_names)


if __name__ == "__main__":
    main()
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np

from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import densenet121
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)


DATA_ROOT = Path(r"F:\College\DataScience\CNNRetina\data\preprocesseddata\test")
MODEL_PATH = Path(r"F:\College\DataScience\CNNRetina\cnn_oct_model.pth")

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


def main():
    test_loader, class_names = get_test_loader()
    model = load_model(len(class_names))
    labels, preds = get_predictions(model, test_loader)

    print(f"Overall Metrics\n")
    print(f"Accuracy:  {accuracy_score(labels, preds):.4f}")
    print(f"Precision: {precision_score(labels, preds, average='macro', zero_division=0):.4f}")
    print(f"Recall:    {recall_score(labels, preds, average='macro', zero_division=0):.4f}")
    print(f"F1 Score:  {f1_score(labels, preds, average='macro', zero_division=0):.4f}")

    print(f"Per-Class Report\n")
    print(classification_report(labels, preds, target_names=class_names))

    print("Confusion Matrix")
    print(confusion_matrix(labels, preds))


if __name__ == "__main__":
    main()
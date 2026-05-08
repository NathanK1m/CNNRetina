from pathlib import Path
import copy

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import densenet121, DenseNet121_Weights

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


DATA_ROOT = Path(r"F:\College\DataScience\CNNRetina\data\contrastcorrected")

BATCH_SIZE = 32
NUM_EPOCHS = 20
LEARNING_RATE = 1e-4

DEVICE = torch.device("cuda")


def get_dataloaders():
    train_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    val_test_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    train_dataset = datasets.ImageFolder(DATA_ROOT / "train", transform=train_transform)
    val_dataset = datasets.ImageFolder(DATA_ROOT / "val", transform=val_test_transform)
    test_dataset = datasets.ImageFolder(DATA_ROOT / "test", transform=val_test_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader, train_dataset.classes


def build_model(num_classes):
    weights = DenseNet121_Weights.DEFAULT
    model = densenet121(weights=weights)

    num_features = model.classifier.in_features
    model.classifier = nn.Linear(num_features, num_classes)

    model = model.to(DEVICE)
    return model


def train_one_epoch(model, loader, criterion, optimizer):
    model.train()

    running_loss = 0.0
    all_preds = []
    all_labels = []

    for images, labels in loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.detach().cpu().numpy())
        all_labels.extend(labels.detach().cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)

    return epoch_loss, epoch_acc


def evaluate(model, loader, criterion):
    model.eval()

    running_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)

            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.detach().cpu().numpy())
            all_labels.extend(labels.detach().cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)

    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average="macro", zero_division=0)
    recall = recall_score(all_labels, all_preds, average="macro", zero_division=0)
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    return epoch_loss, acc, precision, recall, f1, all_labels, all_preds


def main():
    train_loader, val_loader, test_loader, class_names = get_dataloaders()
    num_classes = len(class_names)

    print(f"Number of classes: {num_classes}")
    print(f"Train images: {len(train_loader.dataset)}")
    print(f"Val images:   {len(val_loader.dataset)}")
    print(f"Test images:  {len(test_loader.dataset)}")

    model = build_model(num_classes)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

    best_val_f1 = 0.0
    best_model_weights = copy.deepcopy(model.state_dict())

    for epoch in range(NUM_EPOCHS):
        print(f"Epoch {epoch + 1}/{NUM_EPOCHS}\n")
        print("----------------------------------------")

        train_loss, train_acc = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer
        )

        val_loss, val_acc, val_precision, val_recall, val_f1, _, _ = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion
        )

        print(f"Train loss: {train_loss:.4f} | Train acc: {train_acc:.4f}")
        print(
            f"Val loss:   {val_loss:.4f} | "
            f"Val acc: {val_acc:.4f} | "
            f"Precision: {val_precision:.4f} | "
            f"Recall: {val_recall:.4f} | "
            f"F1: {val_f1:.4f}"
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_model_weights = copy.deepcopy(model.state_dict())
            torch.save({
                "model_state_dict": best_model_weights,
                "class_names": class_names,
                "val_f1": best_val_f1
            }, "best_densenet121_oct3.pth")

            print("Saved new best model.")

    print(f"Training finished.\n")
    print(f"Best validation F1: {best_val_f1:.4f}")

    model.load_state_dict(best_model_weights)

    test_loss, test_acc, test_precision, test_recall, test_f1, test_labels, test_preds = evaluate(
        model=model,
        loader=test_loader,
        criterion=criterion
    )

    print("Final test results\n")
    print("----------------------------------------")
    print(f"Test loss:      {test_loss:.4f}")
    print(f"Test accuracy:  {test_acc:.4f}")
    print(f"Test precision: {test_precision:.4f}")
    print(f"Test recall:    {test_recall:.4f}")
    print(f"Test F1:        {test_f1:.4f}")

    print("Confusion matrix:\n")
    print(confusion_matrix(test_labels, test_preds))


if __name__ == "__main__":
    main()
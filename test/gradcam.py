import random
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np


MODEL_PATH = Path(r"F:\College\DataScience\CNNRetina\cnn_oct_model.pth")
DATA_DIR = Path(r"F:\College\DataScience\CNNRetina\data\preprocesseddata\test")
OUTPUT_DIR = Path(r"F:\College\DataScience\CNNRetina\gradcam_output")
NUM_IMAGES = 350

DEVICE = torch.device("cuda")


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.model.eval()
        self.gradients = None
        self.activations = None

        target_layer.register_forward_hook(self._forward_hook)
        target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, input, output):
        self.activations = output.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor):
        output = self.model(input_tensor)
        pred_class = output.argmax(dim=1).item()
        confidence = F.softmax(output, dim=1)[0, pred_class].item() * 100

        self.model.zero_grad()
        output[0, pred_class].backward()

        weights = self.gradients.mean(dim=[2, 3], keepdim=True)
        cam = F.relu((weights * self.activations).sum(dim=1, keepdim=True))

        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()

        cam = F.interpolate(cam, size=(224, 224), mode='bilinear', align_corners=False)
        return cam.squeeze().cpu().numpy(), pred_class, confidence


def overlay_heatmap(img_array, heatmap, alpha=0.4):
    colormap = cm.jet(heatmap)[:, :, :3]
    return np.clip((1 - alpha) * img_array + alpha * colormap, 0, 1)


def process_images():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    classes = sorted([d.name for d in DATA_DIR.iterdir() if d.is_dir()])

    model = models.densenet121(weights=None)
    model.classifier = torch.nn.Linear(model.classifier.in_features, len(classes))
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    target_layer = model.features.denseblock4.denselayer16.conv2
    gradcam = GradCAM(model, target_layer)

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    for class_name in classes:
        class_dir = DATA_DIR / class_name
        image_files = list(class_dir.glob("*.jpg"))

        selected = random.sample(image_files, NUM_IMAGES)

        class_output = OUTPUT_DIR / class_name
        class_output.mkdir(parents=True, exist_ok=True)

        for img_path in selected:
            img = Image.open(img_path)
            input_tensor = transform(img).unsqueeze(0).to(DEVICE)

            heatmap, pred_class, confidence = gradcam.generate(input_tensor)
            pred_name = classes[pred_class]
            
            if pred_name == class_name:
                correct = "CORRECT"
            else:
                correct = "INCORRECT"
                
            img_display = img.convert("L").resize((224, 224))
            img_array = np.array(img_display).astype(np.float32) / 255.0
            img_array = np.stack([img_array] * 3, axis=-1)

            overlay = overlay_heatmap(img_array, heatmap)

            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            axes[0].imshow(img_array)
            axes[0].set_title("Original")
            axes[0].axis("off")

            axes[1].imshow(heatmap, cmap="jet")
            axes[1].set_title("Grad-CAM Heatmap")
            axes[1].axis("off")

            axes[2].imshow(overlay)
            axes[2].set_title(f"Pred: {pred_name} ({confidence:.1f}%)")
            axes[2].axis("off")

            fig.suptitle(f"True: {class_name} | Predicted: {pred_name} [{correct}]",
                         fontsize=14, fontweight="bold")

            plt.savefig(class_output / f"{img_path.stem}_gradcam.png",
                        bbox_inches="tight", dpi=150)
            plt.close()


if __name__ == "__main__":
    process_images()
from pathlib import Path
from PIL import Image
import numpy as np
from scipy import ndimage


INPUT_ROOT = Path(r"F:\College\DataScience\CNNRetina\data\rawdata")
OUTPUT_ROOT = Path(r"F:\College\DataScience\CNNRetina\data\croppeddata")


def find_border_white(gray: np.ndarray, threshold: int = 250) -> np.ndarray:
    h, w = gray.shape
    min_component_size = (h * w) * 0.01

    white_mask = gray >= threshold
    labeled, _ = ndimage.label(white_mask)

    border_labels = set()
    border_labels.update(labeled[0, :].tolist())
    border_labels.update(labeled[h - 1, :].tolist())
    border_labels.update(labeled[:, 0].tolist())
    border_labels.update(labeled[:, w - 1].tolist())
    border_labels.discard(0)

    valid_labels = [
        label for label in border_labels
        if np.sum(labeled == label) >= min_component_size
    ]

    if not valid_labels:
        return np.zeros_like(gray, dtype=bool)

    return np.isin(labeled, valid_labels)


def autocrop_white(img: Image.Image, threshold: int = 250) -> Image.Image:
    white_ratio = 0.3
    min_size = 50

    while True:
        arr = np.array(img)
        gray = np.mean(arr, axis=2)
        h, w = gray.shape

        if h < min_size or w < min_size:
            break

        border_mask = find_border_white(gray, threshold)

        if not border_mask.any():
            break

        edges = {
            "top": np.mean(border_mask[0, :]),
            "bottom": np.mean(border_mask[h - 1, :]),
            "left": np.mean(border_mask[:, 0]),
            "right": np.mean(border_mask[:, w - 1]),
        }
        worst_edge = max(edges, key=edges.get)

        if edges[worst_edge] < 0.05:
            break

        row_white_pct = np.mean(border_mask, axis=1)
        col_white_pct = np.mean(border_mask, axis=0)

        crop_amount = 0
        if worst_edge == "top":
            for i in range(h):
                if row_white_pct[i] < white_ratio:
                    break
                crop_amount += 1
        elif worst_edge == "bottom":
            for i in range(h - 1, -1, -1):
                if row_white_pct[i] < white_ratio:
                    break
                crop_amount += 1
        elif worst_edge == "left":
            for i in range(w):
                if col_white_pct[i] < white_ratio:
                    break
                crop_amount += 1
        elif worst_edge == "right":
            for i in range(w - 1, -1, -1):
                if col_white_pct[i] < white_ratio:
                    break
                crop_amount += 1

        if crop_amount == 0:
            break

        if worst_edge in ("top", "bottom"):
            crop_amount = min(crop_amount, h // 2)
            if (h - crop_amount) < min_size:
                break
        else:
            crop_amount = min(crop_amount, w // 2)
            if (w - crop_amount) < min_size:
                break

        if worst_edge == "top":
            img = img.crop((0, crop_amount, w, h))
        elif worst_edge == "bottom":
            img = img.crop((0, 0, w, h - crop_amount))
        elif worst_edge == "left":
            img = img.crop((crop_amount, 0, w, h))
        elif worst_edge == "right":
            img = img.crop((0, 0, w - crop_amount, h))

    return img


def process_dataset(input_root: Path, output_root: Path):
    image_paths = list(input_root.rglob("*.jpg"))

    if not image_paths:
        print(f"No images found in: {input_root}")
        return

    print(f"Found {len(image_paths)} images.")

    size_changes = []

    for i, img_path in enumerate(image_paths, start=1):
        relative = img_path.relative_to(input_root)
        dest = output_root / relative
        dest.parent.mkdir(parents=True, exist_ok=True)

        img = Image.open(img_path).convert("RGB")
        original_size = img.size
        img = autocrop_white(img)
        new_size = img.size

        if original_size != new_size:
            pct = (1 - (new_size[0] * new_size[1]) / (original_size[0] * original_size[1])) * 100
            size_changes.append((str(relative), original_size, new_size, pct))

        img.save(dest, quality=95)

        if i % 500 == 0:
            print(f"Processed {i}/{len(image_paths)} images")

    print(f"{len(image_paths)} images processed.")

    badly_cropped = [s for s in size_changes if s[3] > 70]
    if badly_cropped:
        print(f"\nDeleting {len(badly_cropped)} images that lost >70% of their area:")
        for name, orig, new, pct in sorted(badly_cropped, key=lambda x: -x[3]):
            (output_root / name).unlink(missing_ok=True)
            print(f"Deleted {name}: {orig} → {new} ({pct:.1f}% removed)")

    moderate = [s for s in size_changes if 50 < s[3] <= 70]
    if moderate:
        print(f"\n{len(moderate)} images lost 50-70% of their area (kept):")
        for name, orig, new, pct in sorted(moderate, key=lambda x: -x[3]):
            print(f"{name}: {orig} → {new} ({pct:.1f}% removed)")


if __name__ == "__main__":
    process_dataset(INPUT_ROOT, OUTPUT_ROOT)
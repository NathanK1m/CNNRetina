from pathlib import Path
from PIL import Image
import numpy as np
import hashlib
import random


INPUT_ROOT = Path(r"F:\College\DataScience\CNNRetina\data\croppeddata")
OUTPUT_ROOT = Path(r"F:\College\DataScience\CNNRetina\data\preprocesseddata")

TARGET_MEAN = 128.0
TARGET_STD = 45.0
BLACK_THRESHOLD = 5
MIN_STD = 1e-6

RANDOMIZE_TRAIN_PADDING = True
RANDOMIZE_VAL_TEST_PADDING = False
PADDING_JITTER_FRACTION = 0.5
RANDOM_SEED = 123


def get_split_name(img_path: Path, input_root: Path) -> str:
    relative_parts = img_path.relative_to(input_root).parts
    if len(relative_parts) == 0:
        return ""
    return relative_parts[0].lower()


def stable_random_for_path(path: Path) -> random.Random:
    key = str(path).encode("utf-8")
    digest = hashlib.md5(key).hexdigest()
    path_seed = int(digest[:8], 16)
    return random.Random(RANDOM_SEED + path_seed)


def normalize_brightness_contrast(img: Image.Image) -> Image.Image:
    img = img.convert("L")
    arr = np.array(img).astype(np.float32)

    valid_pixels = arr[arr > BLACK_THRESHOLD]

    if valid_pixels.size < 100:
        valid_pixels = arr.flatten()

    mean = np.mean(valid_pixels)
    std = np.std(valid_pixels)

    if std < MIN_STD:
        return img

    normalized = (arr - mean) / std
    normalized = normalized * TARGET_STD + TARGET_MEAN
    normalized = np.clip(normalized, 0, 255)
    normalized[arr <= BLACK_THRESHOLD] = 0

    return Image.fromarray(normalized.astype(np.uint8), mode="L")


def pad_top_bottom_random(img: Image.Image, img_path: Path, input_root: Path) -> Image.Image:
    width, height = img.size

    if width <= height:
        return img

    total_padding = width - height
    split_name = get_split_name(img_path, input_root)

    if split_name == "train":
        use_random_padding = RANDOMIZE_TRAIN_PADDING
    elif split_name in {"val", "test"}:
        use_random_padding = RANDOMIZE_VAL_TEST_PADDING
    else:
        use_random_padding = False

    if use_random_padding:
        rng = stable_random_for_path(img_path)
        center_top = total_padding // 2
        max_jitter = int(total_padding * PADDING_JITTER_FRACTION)

        top = center_top + rng.randint(-max_jitter, max_jitter)
        top = max(0, min(top, total_padding))
    else:
        top = total_padding // 2

    padded = Image.new("L", (width, width), color=0)
    padded.paste(img, (0, top))

    return padded


def process_dataset(input_root: Path, output_root: Path):
    image_paths = list(input_root.rglob("*.jpg"))

    for i, img_path in enumerate(image_paths, start=1):
        relative = img_path.relative_to(input_root)
        dest = output_root / relative
        dest.parent.mkdir(parents=True, exist_ok=True)

        img = Image.open(img_path).convert("L")
        normalized = normalize_brightness_contrast(img)
        padded = pad_top_bottom_random(normalized, img_path, input_root)
        padded.save(dest, quality=95)

        if i % 500 == 0:
            print(f"Processed {i}/{len(image_paths)} images")


if __name__ == "__main__":
    process_dataset(INPUT_ROOT, OUTPUT_ROOT)
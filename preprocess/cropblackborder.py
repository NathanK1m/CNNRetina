from pathlib import Path
from PIL import Image


INPUT_ROOT = Path(r"F:\College\DataScience\CNNRetina\data\rawdata")
OUTPUT_ROOT = Path(r"F:\College\DataScience\CNNRetina\data\croppeddata")

CROP_HEIGHT = 420
CROP_WIDTH = 650


def center_crop(img: Image.Image, crop_height: int, crop_width: int) -> Image.Image:
    width, height = img.size

    if crop_width > width or crop_height > height:
        raise ValueError(
            f"Crop size ({crop_width}x{crop_height}) is larger than image size ({width}x{height})"
        )

    left = (width - crop_width) // 2
    top = (height - crop_height) // 2

    return img.crop((left, top, left + crop_width, top + crop_height))


def process_dataset(input_root: Path, output_root: Path):
    image_paths = list(input_root.rglob("*.jpg"))

    if not image_paths:
        print(f"No images found in: {input_root}")
        return

    print(f"Found {len(image_paths)} images.")
    print(f"Center crop size: {CROP_WIDTH}x{CROP_HEIGHT}")

    for i, img_path in enumerate(image_paths, start=1):
        relative = img_path.relative_to(input_root)
        dest = output_root / relative
        dest.parent.mkdir(parents=True, exist_ok=True)

        img = Image.open(img_path).convert("L")
        cropped = center_crop(img, CROP_HEIGHT, CROP_WIDTH)
        cropped.save(dest)

        if i % 500 == 0:
            print(f"Processed {i}/{len(image_paths)} images")

    print(f"{len(image_paths)} images processed.")


if __name__ == "__main__":
    process_dataset(INPUT_ROOT, OUTPUT_ROOT)
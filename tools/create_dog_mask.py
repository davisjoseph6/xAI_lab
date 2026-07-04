"""Create an approximate 224x224 dog-region mask.

Usage:
    python tools/create_dog_mask.py images/model_input_224.png images/dog_mask_224.png

Instructions:
    1. A window opens showing the model input image.
    2. Left-click around the visible outline of the dog.
    3. Press Enter when finished.
    4. The script saves:
       - images/dog_mask_224.png
       - images/dog_mask_224_preview.png

Mask convention:
    white = dog
    black = snow/background
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit(
            "Usage: python tools/create_dog_mask.py "
            "images/model_input_224.png images/dog_mask_224.png"
        )

    image_path = Path(sys.argv[1])
    mask_path = Path(sys.argv[2])

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = Image.open(image_path).convert("RGB")

    if image.size != (224, 224):
        raise ValueError(
            f"Expected input image to be 224x224, got {image.size}"
        )

    image_array = np.asarray(image)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(image_array)
    ax.set_title(
        "Click around the DOG outline. Press Enter when finished."
    )
    ax.axis("off")

    points = plt.ginput(n=-1, timeout=0)
    plt.close(fig)

    if len(points) < 3:
        raise SystemExit(
            "You need at least 3 points to create a polygon mask."
        )

    polygon = [(int(round(x)), int(round(y))) for x, y in points]

    mask = Image.new("L", image.size, 0)
    draw = ImageDraw.Draw(mask)
    draw.polygon(polygon, fill=255)

    mask_path.parent.mkdir(parents=True, exist_ok=True)
    mask.save(mask_path)

    # Make a preview: dog region lightly highlighted in white.
    mask_array = np.asarray(mask) / 255.0
    preview = image_array.copy().astype(float)
    preview = preview * 0.55 + 255.0 * mask_array[..., None] * 0.45
    preview = np.clip(preview, 0, 255).astype("uint8")

    preview_path = mask_path.with_name(
        mask_path.stem + "_preview.png"
    )
    Image.fromarray(preview).save(preview_path)

    print(f"Saved mask: {mask_path}")
    print(f"Saved preview: {preview_path}")
    print(f"Mask size: {mask.size}")
    print("White = dog, black = background")


if __name__ == "__main__":
    main()

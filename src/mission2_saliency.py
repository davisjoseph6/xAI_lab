"""Mission 2, part A: original-model saliency explanations."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from captum.attr import (
    IntegratedGradients,
    LayerAttribution,
    LayerGradCam,
    Saliency,
)
from PIL import Image, ImageFilter
from torchvision.models import (
    ResNet50_Weights,
    resnet50,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from utilities import (  # noqa: E402
    ensure_directory,
    normalise_for_display,
    save_json,
    set_global_seed,
    sha256_file,
)


IMAGE_PATH = PROJECT_ROOT / "images" / "husky.jpg"
OUTPUT_DIRECTORY = PROJECT_ROOT / "outputs" / "mission2"

RANDOM_SEED = 42
INTEGRATED_GRADIENT_STEPS = 64
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


def load_model_and_inputs(
    image_path: Path,
) -> tuple[
    torch.nn.Module,
    ResNet50_Weights,
    Image.Image,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Load ResNet-50 and construct input and baseline tensors."""
    weights = ResNet50_Weights.DEFAULT
    model = resnet50(weights=weights)
    model.eval()
    model.to(DEVICE)

    image = Image.open(image_path).convert("RGB")
    transform = weights.transforms()

    input_tensor = transform(image).unsqueeze(0).to(DEVICE)

    # Create the black baseline in RGB image space, not by assuming
    # that a zero normalised tensor means black.
    black_image = Image.new(
        mode="RGB",
        size=image.size,
        color=(0, 0, 0),
    )
    black_baseline = transform(
        black_image
    ).unsqueeze(0).to(DEVICE)

    blurred_image = image.filter(
        ImageFilter.GaussianBlur(radius=20)
    )
    blurred_baseline = transform(
        blurred_image
    ).unsqueeze(0).to(DEVICE)

    return (
        model,
        weights,
        image,
        input_tensor,
        black_baseline,
        blurred_baseline,
    )


def tensor_to_rgb(
    normalised_tensor: torch.Tensor,
    weights: ResNet50_Weights,
) -> np.ndarray:
    """Convert the model input tensor back into display RGB."""
    transform = weights.transforms()

    mean = torch.tensor(
        transform.mean,
        device=normalised_tensor.device,
    ).view(1, 3, 1, 1)

    standard_deviation = torch.tensor(
        transform.std,
        device=normalised_tensor.device,
    ).view(1, 3, 1, 1)

    image = (
        normalised_tensor * standard_deviation + mean
    ).clamp(0, 1)

    return (
        image[0]
        .detach()
        .cpu()
        .permute(1, 2, 0)
        .numpy()
    )


def compute_attributions(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    black_baseline: torch.Tensor,
    blurred_baseline: torch.Tensor,
    target_class: int,
) -> dict:
    """Calculate all raw and display-space attribution maps."""
    model.eval()

    saliency_input = input_tensor.detach().clone()
    saliency_input.requires_grad_(True)

    saliency_method = Saliency(model)
    saliency_raw = saliency_method.attribute(
        saliency_input,
        target=target_class,
        abs=False,
    )

    integrated_gradients = IntegratedGradients(model)

    ig_black_raw, ig_black_delta = (
        integrated_gradients.attribute(
            input_tensor.detach().clone(),
            baselines=black_baseline,
            target=target_class,
            n_steps=INTEGRATED_GRADIENT_STEPS,
            method="gausslegendre",
            return_convergence_delta=True,
        )
    )

    ig_blurred_raw, ig_blurred_delta = (
        integrated_gradients.attribute(
            input_tensor.detach().clone(),
            baselines=blurred_baseline,
            target=target_class,
            n_steps=INTEGRATED_GRADIENT_STEPS,
            method="gausslegendre",
            return_convergence_delta=True,
        )
    )

    # Exact layer choice must be reported.
    gradcam_layer = model.layer4[-1].conv3
    gradcam_method = LayerGradCam(model, gradcam_layer)

    gradcam_raw = gradcam_method.attribute(
        input_tensor.detach().clone(),
        target=target_class,
        relu_attributions=True,
    )

    gradcam_upsampled = LayerAttribution.interpolate(
        gradcam_raw,
        interpolate_dims=input_tensor.shape[-2:],
        interpolate_mode="bilinear",
    )

    # These spatial maps are used consistently for visualisation and
    # model-randomisation comparisons. The full raw arrays are also saved.
    spatial_maps = {
        "saliency": (
            saliency_raw.abs()
            .amax(dim=1)
            .squeeze(0)
            .detach()
            .cpu()
            .numpy()
        ),
        "integrated_gradients_black": (
            ig_black_raw.abs()
            .sum(dim=1)
            .squeeze(0)
            .detach()
            .cpu()
            .numpy()
        ),
        "integrated_gradients_blurred": (
            ig_blurred_raw.abs()
            .sum(dim=1)
            .squeeze(0)
            .detach()
            .cpu()
            .numpy()
        ),
        "gradcam": (
            gradcam_upsampled
            .squeeze(0)
            .squeeze(0)
            .detach()
            .cpu()
            .numpy()
        ),
    }

    return {
        "raw": {
            "saliency": saliency_raw.detach().cpu(),
            "integrated_gradients_black": (
                ig_black_raw.detach().cpu()
            ),
            "integrated_gradients_blurred": (
                ig_blurred_raw.detach().cpu()
            ),
            "gradcam_original_resolution": (
                gradcam_raw.detach().cpu()
            ),
            "gradcam_upsampled": (
                gradcam_upsampled.detach().cpu()
            ),
        },
        "spatial": spatial_maps,
        "diagnostics": {
            "ig_black_convergence_delta": float(
                ig_black_delta.detach().cpu().item()
            ),
            "ig_blurred_convergence_delta": float(
                ig_blurred_delta.detach().cpu().item()
            ),
            "gradcam_layer": "layer4[-1].conv3",
        },
    }


def save_attribution_figure(
    input_rgb: np.ndarray,
    attribution: np.ndarray,
    title: str,
    output_path: Path,
) -> None:
    """Save the attribution alone and overlaid on the input."""
    display_map = normalise_for_display(attribution)

    figure = plt.figure(figsize=(12, 5))

    first_axis = figure.add_subplot(1, 2, 1)
    first_axis.imshow(display_map)
    first_axis.set_title(f"{title}: attribution map")
    first_axis.axis("off")

    second_axis = figure.add_subplot(1, 2, 2)
    second_axis.imshow(input_rgb)
    second_axis.imshow(display_map, alpha=0.50)
    second_axis.set_title(f"{title}: visual overlay")
    second_axis.axis("off")

    figure.tight_layout()
    figure.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(figure)


def main() -> None:
    set_global_seed(RANDOM_SEED)
    ensure_directory(OUTPUT_DIRECTORY)

    if not IMAGE_PATH.exists():
        raise FileNotFoundError(
            f"Place the chosen image at {IMAGE_PATH}."
        )

    (
        model,
        weights,
        _original_image,
        input_tensor,
        black_baseline,
        blurred_baseline,
    ) = load_model_and_inputs(IMAGE_PATH)

    with torch.no_grad():
        logits = model(input_tensor)
        probabilities = torch.softmax(logits, dim=1)[0]

    top_probabilities, top_indices = torch.topk(
        probabilities,
        k=5,
    )

    categories = weights.meta["categories"]

    top_predictions = [
        {
            "class_index": int(index),
            "class_name": categories[int(index)],
            "probability": float(probability),
            "logit": float(logits[0, int(index)]),
        }
        for probability, index in zip(
            top_probabilities,
            top_indices,
            strict=True,
        )
    ]

    target_class = int(top_indices[0])
    target_name = categories[target_class]

    acceptable_terms = (
        "husky",
        "malamute",
        "eskimo dog",
    )

    if not any(
        term in target_name.lower()
        for term in acceptable_terms
    ):
        raise RuntimeError(
            "The model's top prediction is "
            f"'{target_name}', not a declared husky-related class. "
            "Select another image rather than changing the label."
        )

    attribution_results = compute_attributions(
        model=model,
        input_tensor=input_tensor,
        black_baseline=black_baseline,
        blurred_baseline=blurred_baseline,
        target_class=target_class,
    )

    torch.save(
        attribution_results["raw"],
        OUTPUT_DIRECTORY / "original_raw_attributions.pt",
    )

    input_rgb = tensor_to_rgb(input_tensor, weights)

    np.save(
        OUTPUT_DIRECTORY / "model_input_rgb.npy",
        input_rgb,
    )

    for method_name, attribution in attribution_results[
        "spatial"
    ].items():
        np.save(
            OUTPUT_DIRECTORY
            / f"original_{method_name}_spatial.npy",
            attribution,
        )

        save_attribution_figure(
            input_rgb=input_rgb,
            attribution=attribution,
            title=method_name.replace("_", " ").title(),
            output_path=(
                OUTPUT_DIRECTORY
                / f"original_{method_name}.png"
            ),
        )

    metadata = {
        "image_path": str(IMAGE_PATH),
        "image_sha256": sha256_file(IMAGE_PATH),
        "device": str(DEVICE),
        "weights": str(weights),
        "target_class_index": target_class,
        "target_class_name": target_name,
        "target_quantity": "pre-softmax class logit",
        "top_five_predictions": top_predictions,
        "saliency_setting": (
            "signed input gradients saved; maximum absolute "
            "channel value used for spatial display"
        ),
        "integrated_gradients_baselines": [
            "black RGB image passed through model preprocessing",
            "Gaussian-blurred original passed through preprocessing",
        ],
        "integrated_gradients_steps": (
            INTEGRATED_GRADIENT_STEPS
        ),
        "integrated_gradients_method": "gausslegendre",
        "gradcam_layer": "layer4[-1].conv3",
        "gradcam_relu_attributions": True,
        "gradcam_interpolation": "bilinear",
        **attribution_results["diagnostics"],
    }

    save_json(
        metadata,
        OUTPUT_DIRECTORY / "saliency_metadata.json",
    )

    print("Mission 2 original attributions completed.")
    print(f"Target: {target_name} ({target_class})")
    print("Top-five predictions:")

    for prediction in top_predictions:
        print(
            f"  {prediction['class_name']}: "
            f"{prediction['probability']:.4f}"
        )


if __name__ == "__main__":
    main()

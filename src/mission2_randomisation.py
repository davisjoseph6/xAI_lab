"""Mission 2, part B: model randomisation and region interventions."""

from __future__ import annotations

import copy
import hashlib
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mission2_saliency import (  # noqa: E402
    DEVICE,
    IMAGE_PATH,
    compute_attributions,
    load_model_and_inputs,
)
from utilities import (  # noqa: E402
    ensure_directory,
    safe_spearman,
    save_json,
    set_global_seed,
)


OUTPUT_DIRECTORY = PROJECT_ROOT / "outputs" / "mission2"
MASK_PATH = PROJECT_ROOT / "images" / "dog_mask_224.png"

RANDOMISATION_SEEDS = [101, 202, 303]

RANDOMISATION_STAGES = [
    "fc",
    "layer4",
    "layer3",
    "layer2",
    "layer1",
    "stem",
]


def state_dict_digest(model: torch.nn.Module) -> str:
    """Hash all model state tensors to detect accidental mutation."""
    digest = hashlib.sha256()

    for name, tensor in model.state_dict().items():
        digest.update(name.encode("utf-8"))
        digest.update(
            tensor.detach().cpu().contiguous().numpy().tobytes()
        )

    return digest.hexdigest()


def reset_module_tree(module: torch.nn.Module) -> None:
    """Reinitialise all parameterised children in a model block."""
    for child in module.modules():
        reset_parameters = getattr(
            child,
            "reset_parameters",
            None,
        )

        if callable(reset_parameters):
            reset_parameters()

        # Explicitly reset BatchNorm state so that learned running
        # statistics do not survive the randomisation.
        if isinstance(
            child,
            torch.nn.modules.batchnorm._BatchNorm,
        ):
            child.reset_running_stats()


def randomise_named_stage(
    model: torch.nn.Module,
    stage_name: str,
) -> None:
    """Randomise one newly added stage."""
    if stage_name == "fc":
        reset_module_tree(model.fc)
    elif stage_name == "layer4":
        reset_module_tree(model.layer4)
    elif stage_name == "layer3":
        reset_module_tree(model.layer3)
    elif stage_name == "layer2":
        reset_module_tree(model.layer2)
    elif stage_name == "layer1":
        reset_module_tree(model.layer1)
    elif stage_name == "stem":
        reset_module_tree(model.conv1)
        reset_module_tree(model.bn1)
    else:
        raise ValueError(
            f"Unknown randomisation stage: {stage_name}"
        )

    model.eval()


def create_null_correlations(
    original_map: np.ndarray,
    number_of_permutations: int = 200,
    seed: int = 77,
) -> list[float]:
    """Calculate correlations against shuffled versions of a map."""
    rng = np.random.default_rng(seed)
    original_flat = np.asarray(original_map).ravel()

    correlations = []

    for _ in range(number_of_permutations):
        shuffled = rng.permutation(original_flat)

        correlations.append(
            safe_spearman(
                original_flat,
                shuffled,
            )
        )

    return correlations


def run_randomisation_experiment(
    original_model: torch.nn.Module,
    input_tensor: torch.Tensor,
    black_baseline: torch.Tensor,
    blurred_baseline: torch.Tensor,
    target_class: int,
) -> pd.DataFrame:
    """Run cumulative top-down model randomisation."""
    original_results = compute_attributions(
        model=original_model,
        input_tensor=input_tensor,
        black_baseline=black_baseline,
        blurred_baseline=blurred_baseline,
        target_class=target_class,
    )

    original_maps = original_results["spatial"]
    rows = []

    randomised_map_directory = (
        OUTPUT_DIRECTORY / "randomised_maps"
    )
    ensure_directory(randomised_map_directory)

    for seed in RANDOMISATION_SEEDS:
        set_global_seed(seed)

        # Each seed starts from an untouched copy of the trained model.
        randomised_model = copy.deepcopy(original_model)
        randomised_model.to(DEVICE)
        randomised_model.eval()

        for stage_number, stage_name in enumerate(
            RANDOMISATION_STAGES,
            start=1,
        ):
            randomise_named_stage(
                randomised_model,
                stage_name,
            )

            with torch.no_grad():
                target_logit = float(
                    randomised_model(input_tensor)[0, target_class]
                )

            randomised_results = compute_attributions(
                model=randomised_model,
                input_tensor=input_tensor,
                black_baseline=black_baseline,
                blurred_baseline=blurred_baseline,
                target_class=target_class,
            )

            arrays_to_save = {}

            for method_name, randomised_map in (
                randomised_results["spatial"].items()
            ):
                correlation = safe_spearman(
                    original_maps[method_name],
                    randomised_map,
                )

                rows.append(
                    {
                        "seed": seed,
                        "stage_number": stage_number,
                        "randomised_through": stage_name,
                        "method": method_name,
                        "spearman_correlation": correlation,
                        "fixed_target_logit": target_logit,
                    }
                )

                arrays_to_save[method_name] = randomised_map

            np.savez_compressed(
                randomised_map_directory
                / f"seed_{seed}_through_{stage_name}.npz",
                **arrays_to_save,
            )

    return pd.DataFrame(rows)


def summarise_randomisation(
    results: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate mean and standard deviation across randomisation seeds."""
    summary = (
        results.groupby(
            [
                "stage_number",
                "randomised_through",
                "method",
            ],
            as_index=False,
        )
        .agg(
            mean_spearman=(
                "spearman_correlation",
                "mean",
            ),
            standard_deviation=(
                "spearman_correlation",
                "std",
            ),
            minimum_spearman=(
                "spearman_correlation",
                "min",
            ),
            maximum_spearman=(
                "spearman_correlation",
                "max",
            ),
            mean_fixed_target_logit=(
                "fixed_target_logit",
                "mean",
            ),
        )
    )

    return summary


def plot_randomisation_curves(
    summary: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot model-randomisation correlations for all methods."""
    figure, axis = plt.subplots(figsize=(11, 7))

    for method, group in summary.groupby("method"):
        ordered = group.sort_values("stage_number")

        axis.errorbar(
            ordered["stage_number"],
            ordered["mean_spearman"],
            yerr=ordered["standard_deviation"].fillna(0),
            marker="o",
            label=method.replace("_", " "),
        )

    axis.axhline(0, linewidth=1)
    axis.set_xticks(
        range(1, len(RANDOMISATION_STAGES) + 1),
        RANDOMISATION_STAGES,
        rotation=30,
    )
    axis.set_xlabel("Cumulative randomisation stage")
    axis.set_ylabel(
        "Spearman correlation with trained-model map"
    )
    axis.set_title(
        "Adebayo-style progressive model-randomisation test"
    )
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def load_aligned_dog_mask(
    mask_path: Path,
    required_height: int,
    required_width: int,
) -> torch.Tensor:
    """Load a binary mask in model-input coordinates."""
    mask_image = Image.open(mask_path).convert("L")

    if mask_image.size != (
        required_width,
        required_height,
    ):
        raise ValueError(
            "The dog mask must already be aligned with the model "
            f"input. Expected {(required_width, required_height)}, "
            f"received {mask_image.size}."
        )

    mask_array = np.asarray(mask_image, dtype=np.float32) / 255.0
    mask_array = (mask_array >= 0.5).astype(np.float32)

    return (
        torch.from_numpy(mask_array)
        .unsqueeze(0)
        .unsqueeze(0)
        .to(DEVICE)
    )


def model_output_record(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    target_class: int,
    condition: str,
    replacement: str,
) -> dict:
    """Measure model output for one intervention."""
    with torch.no_grad():
        logits = model(input_tensor)
        probabilities = torch.softmax(logits, dim=1)
        predicted_class = int(logits.argmax(dim=1).item())

    return {
        "condition": condition,
        "replacement": replacement,
        "target_logit": float(logits[0, target_class]),
        "target_probability": float(
            probabilities[0, target_class]
        ),
        "predicted_class_index": predicted_class,
    }


def run_region_interventions(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    black_baseline: torch.Tensor,
    blurred_baseline: torch.Tensor,
    target_class: int,
    mask_path: Path,
) -> pd.DataFrame:
    """Intervene separately on dog and background regions."""
    height, width = input_tensor.shape[-2:]

    dog_mask = load_aligned_dog_mask(
        mask_path,
        required_height=height,
        required_width=width,
    )

    background_mask = 1.0 - dog_mask

    # Zero in normalised space corresponds approximately to the
    # ImageNet channel mean, not to an RGB-black image.
    mean_normalised_baseline = torch.zeros_like(input_tensor)

    replacements = {
        "blurred_image": blurred_baseline,
        "black_rgb_image": black_baseline,
        "normalised_channel_mean": mean_normalised_baseline,
    }

    records = [
        model_output_record(
            model,
            input_tensor,
            target_class,
            condition="original",
            replacement="none",
        )
    ]

    for replacement_name, replacement_tensor in replacements.items():
        dog_obscured = (
            input_tensor * background_mask
            + replacement_tensor * dog_mask
        )

        background_obscured = (
            input_tensor * dog_mask
            + replacement_tensor * background_mask
        )

        records.append(
            model_output_record(
                model,
                dog_obscured,
                target_class,
                condition="dog_region_replaced",
                replacement=replacement_name,
            )
        )

        records.append(
            model_output_record(
                model,
                background_obscured,
                target_class,
                condition="background_region_replaced",
                replacement=replacement_name,
            )
        )

    result = pd.DataFrame(records)

    original_logit = float(
        result.loc[
            result["condition"] == "original",
            "target_logit",
        ].iloc[0]
    )

    original_probability = float(
        result.loc[
            result["condition"] == "original",
            "target_probability",
        ].iloc[0]
    )

    result["target_logit_change"] = (
        result["target_logit"] - original_logit
    )
    result["target_probability_change"] = (
        result["target_probability"] - original_probability
    )

    return result


def main() -> None:
    ensure_directory(OUTPUT_DIRECTORY)

    metadata_path = (
        OUTPUT_DIRECTORY / "saliency_metadata.json"
    )

    if not metadata_path.exists():
        raise FileNotFoundError(
            "Run mission2_saliency.py before this script."
        )

    import json

    with metadata_path.open("r", encoding="utf-8") as file:
        metadata = json.load(file)

    target_class = int(metadata["target_class_index"])

    (
        original_model,
        _weights,
        _image,
        input_tensor,
        black_baseline,
        blurred_baseline,
    ) = load_model_and_inputs(IMAGE_PATH)

    original_digest_before = state_dict_digest(
        original_model
    )

    randomisation_results = run_randomisation_experiment(
        original_model=original_model,
        input_tensor=input_tensor,
        black_baseline=black_baseline,
        blurred_baseline=blurred_baseline,
        target_class=target_class,
    )

    original_digest_after = state_dict_digest(
        original_model
    )

    if original_digest_before != original_digest_after:
        raise AssertionError(
            "The original trained model was modified in place."
        )

    randomisation_results.to_csv(
        OUTPUT_DIRECTORY
        / "randomisation_all_seed_results.csv",
        index=False,
    )

    summary = summarise_randomisation(
        randomisation_results
    )

    summary.to_csv(
        OUTPUT_DIRECTORY / "randomisation_summary.csv",
        index=False,
    )

    plot_randomisation_curves(
        summary,
        OUTPUT_DIRECTORY / "randomisation_curves.png",
    )

    original_maps = {
        method: np.load(
            OUTPUT_DIRECTORY
            / f"original_{method}_spatial.npy"
        )
        for method in [
            "saliency",
            "integrated_gradients_black",
            "integrated_gradients_blurred",
            "gradcam",
        ]
    }

    null_summary = {}

    for method, original_map in original_maps.items():
        null_values = create_null_correlations(
            original_map,
            number_of_permutations=200,
            seed=77,
        )

        null_summary[method] = {
            "mean": float(np.nanmean(null_values)),
            "standard_deviation": float(
                np.nanstd(null_values)
            ),
            "minimum": float(np.nanmin(null_values)),
            "maximum": float(np.nanmax(null_values)),
            "percentile_2_5": float(
                np.nanpercentile(null_values, 2.5)
            ),
            "percentile_97_5": float(
                np.nanpercentile(null_values, 97.5)
            ),
        }

    save_json(
        null_summary,
        OUTPUT_DIRECTORY
        / "shuffled_map_null_correlations.json",
    )

    if MASK_PATH.exists():
        interventions = run_region_interventions(
            model=original_model,
            input_tensor=input_tensor,
            black_baseline=black_baseline,
            blurred_baseline=blurred_baseline,
            target_class=target_class,
            mask_path=MASK_PATH,
        )

        interventions.to_csv(
            OUTPUT_DIRECTORY
            / "dog_background_interventions.csv",
            index=False,
        )

        intervention_status = (
            "Completed using images/dog_mask_224.png"
        )
    else:
        intervention_status = (
            "Not run: create images/dog_mask_224.png, with white "
            "for the dog and black for the background."
        )

    save_json(
        {
            "target_class_was_fixed": target_class,
            "randomisation_order": RANDOMISATION_STAGES,
            "randomisation_seeds": RANDOMISATION_SEEDS,
            "original_model_digest_unchanged": (
                original_digest_before
                == original_digest_after
            ),
            "batchnorm_running_statistics_reset": True,
            "correlations_calculated_from": (
                "raw numerical spatial arrays, not PNG files"
            ),
            "intervention_status": intervention_status,
        },
        OUTPUT_DIRECTORY
        / "randomisation_methodology.json",
    )

    print("Mission 2 model-randomisation test completed.")
    print("Inspect randomisation_summary.csv.")
    print("Inspect shuffled_map_null_correlations.json.")
    print(intervention_status)


if __name__ == "__main__":
    main()

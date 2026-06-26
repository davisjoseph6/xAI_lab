"""Mission 1, part B: SHAP, LIME, stability, and disagreement tests."""

from __future__ import annotations

import itertools
import sys
from pathlib import Path
from typing import Callable

import joblib
import lime.lime_tabular
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from utilities import (  # noqa: E402
    ensure_directory,
    safe_spearman,
    save_json,
    set_global_seed,
    top_k_jaccard,
)


OUTPUT_DIRECTORY = PROJECT_ROOT / "outputs" / "mission1"
MODEL_PATH = OUTPUT_DIRECTORY / "credit_model.joblib"

LIME_SEEDS = [11, 22, 33, 44, 55]
LIME_SAMPLE_COUNT = 5_000
SHAP_BACKGROUND_SIZE = 100


def select_bad_class_explanation(
    explanation: shap.Explanation,
    bad_class_column: int,
) -> shap.Explanation:
    """Select the bad-class output if SHAP returns a class dimension."""
    values = np.asarray(explanation.values)

    if values.ndim == 2:
        return explanation

    if values.ndim != 3:
        raise ValueError(
            f"Unexpected SHAP value dimensions: {values.shape}"
        )

    selected_values = values[:, :, bad_class_column]
    base_values = np.asarray(explanation.base_values)

    if base_values.ndim == 2:
        selected_base_values = base_values[:, bad_class_column]
    elif base_values.ndim == 1 and len(base_values) > 1:
        selected_base_values = base_values[bad_class_column]
    else:
        selected_base_values = base_values

    return shap.Explanation(
        values=selected_values,
        base_values=selected_base_values,
        data=explanation.data,
        feature_names=explanation.feature_names,
    )


def numerical_display_data(
    original_data: pd.DataFrame,
    artefact: dict,
) -> np.ndarray:
    """Represent original features numerically for plotting.

    Categorical values become category IDs. These IDs have no ordinal
    interpretation. We therefore disable categorical colour interpretation
    in the beeswarm plot.
    """
    display = np.zeros(
        (len(original_data), len(artefact["feature_names"])),
        dtype=float,
    )

    for column_index, feature in enumerate(artefact["feature_names"]):
        if feature in artefact["categorical_features"]:
            categories = artefact["category_names"][feature]
            mapping = {
                category: index
                for index, category in enumerate(categories)
            }

            display[:, column_index] = (
                original_data[feature]
                .astype(str)
                .map(mapping)
                .fillna(-1)
                .to_numpy()
            )
        else:
            display[:, column_index] = pd.to_numeric(
                original_data[feature]
            ).to_numpy()

    return display


def group_shap_by_original_feature(
    transformed_explanation: shap.Explanation,
    original_data: pd.DataFrame,
    artefact: dict,
) -> shap.Explanation:
    """Sum transformed SHAP values belonging to each original feature."""
    transformed_values = np.asarray(
        transformed_explanation.values,
        dtype=float,
    )
    transformed_groups = np.asarray(
        artefact["transformed_to_original"]
    )

    grouped_columns = []

    for feature in artefact["feature_names"]:
        mask = transformed_groups == feature

        if not np.any(mask):
            raise AssertionError(
                f"No transformed columns found for {feature}."
            )

        grouped_columns.append(
            transformed_values[:, mask].sum(axis=1)
        )

    grouped_values = np.column_stack(grouped_columns)

    return shap.Explanation(
        values=grouped_values,
        base_values=transformed_explanation.base_values,
        data=numerical_display_data(original_data, artefact),
        feature_names=artefact["feature_names"],
    )


def make_shap_explanation(
    model,
    transformed_background: np.ndarray,
    transformed_evaluation: np.ndarray,
    original_evaluation: pd.DataFrame,
    artefact: dict,
) -> tuple[shap.Explanation, shap.TreeExplainer]:
    """Calculate and group an interventional probability explanation."""
    explainer = shap.TreeExplainer(
        model=model,
        data=transformed_background,
        feature_perturbation="interventional",
        model_output="probability",
    )

    raw_explanation = explainer(
        transformed_evaluation,
        check_additivity=False,
    )

    selected_explanation = select_bad_class_explanation(
        raw_explanation,
        artefact["bad_class_column"],
    )

    grouped_explanation = group_shap_by_original_feature(
        selected_explanation,
        original_evaluation,
        artefact,
    )

    return grouped_explanation, explainer


def save_shap_outputs(
    name: str,
    explanation: shap.Explanation,
    selected_position: int,
    model_bad_probabilities: np.ndarray,
    output_directory: Path,
) -> dict:
    """Save plots, local values, and the numerical additivity test."""
    ensure_directory(output_directory)

    np.save(
        output_directory / f"{name}_grouped_values.npy",
        np.asarray(explanation.values),
    )

    local_frame = pd.DataFrame(
        {
            "feature": explanation.feature_names,
            "shap_value": explanation.values[selected_position],
            "absolute_shap_value": np.abs(
                explanation.values[selected_position]
            ),
        }
    ).sort_values(
        "absolute_shap_value",
        ascending=False,
    )

    local_frame.to_csv(
        output_directory / f"{name}_local_values.csv",
        index=False,
    )

    plt.close("all")
    shap.plots.beeswarm(
        explanation,
        max_display=len(explanation.feature_names),
        color=None,
        show=False,
    )
    plt.gcf().savefig(
        output_directory / f"{name}_beeswarm.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close("all")

    shap.plots.waterfall(
        explanation[selected_position],
        max_display=len(explanation.feature_names),
        show=False,
    )
    plt.gcf().savefig(
        output_directory / f"{name}_waterfall.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close("all")

    local_base = np.asarray(
        explanation.base_values
    )

    if local_base.ndim == 0:
        base_value = float(local_base)
    else:
        base_value = float(local_base[selected_position])

    reconstructed_probability = (
        base_value
        + float(explanation.values[selected_position].sum())
    )

    actual_probability = float(
        model_bad_probabilities[selected_position]
    )

    return {
        "base_value": base_value,
        "sum_of_grouped_shap_values": float(
            explanation.values[selected_position].sum()
        ),
        "reconstructed_bad_probability": reconstructed_probability,
        "actual_bad_probability": actual_probability,
        "absolute_additivity_error": abs(
            reconstructed_probability - actual_probability
        ),
    }


def encode_original_features(
    frame: pd.DataFrame,
    artefact: dict,
) -> np.ndarray:
    """Encode original features into LIME's numerical input format."""
    encoded = np.zeros(
        (len(frame), len(artefact["feature_names"])),
        dtype=float,
    )

    for column_index, feature in enumerate(artefact["feature_names"]):
        if feature in artefact["categorical_features"]:
            categories = artefact["category_names"][feature]
            mapping = {
                value: index
                for index, value in enumerate(categories)
            }

            values = frame[feature].astype(str).map(mapping)

            if values.isna().any():
                unknown = frame.loc[values.isna(), feature].unique()
                raise ValueError(
                    f"Unknown values for {feature}: {unknown}"
                )

            encoded[:, column_index] = values.to_numpy()
        else:
            encoded[:, column_index] = pd.to_numeric(
                frame[feature]
            ).to_numpy()

    return encoded


def decode_lime_matrix(
    matrix: np.ndarray,
    artefact: dict,
) -> pd.DataFrame:
    """Decode LIME perturbations into the model's original input format."""
    matrix = np.asarray(matrix)

    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)

    decoded: dict[str, np.ndarray | list[str]] = {}

    for column_index, feature in enumerate(artefact["feature_names"]):
        column = matrix[:, column_index]

        if feature in artefact["categorical_features"]:
            categories = artefact["category_names"][feature]
            category_indices = np.rint(column).astype(int)
            category_indices = np.clip(
                category_indices,
                0,
                len(categories) - 1,
            )

            decoded[feature] = [
                categories[index]
                for index in category_indices
            ]
        else:
            decoded[feature] = column.astype(float)

    return pd.DataFrame(
        decoded,
        columns=artefact["feature_names"],
    )


def make_lime_predictor(artefact: dict) -> Callable[[np.ndarray], np.ndarray]:
    """Return a predictor accepting LIME's encoded feature matrix."""

    def predict(encoded_matrix: np.ndarray) -> np.ndarray:
        decoded_frame = decode_lime_matrix(
            encoded_matrix,
            artefact,
        )

        transformed = artefact["preprocessor"].transform(
            decoded_frame
        )

        return artefact["model"].predict_proba(transformed)

    return predict


def run_lime_experiments(
    artefact: dict,
    output_directory: Path,
) -> pd.DataFrame:
    """Run LIME with five seeds while keeping other settings fixed."""
    encoded_train = encode_original_features(
        artefact["X_train"],
        artefact,
    )
    encoded_selected = encode_original_features(
        artefact["X_test"].iloc[
            [artefact["selected_test_position"]]
        ],
        artefact,
    )[0]

    categorical_indices = [
        artefact["feature_names"].index(feature)
        for feature in artefact["categorical_features"]
    ]

    categorical_names = {
        artefact["feature_names"].index(feature): artefact[
            "category_names"
        ][feature]
        for feature in artefact["categorical_features"]
    }

    predictor = make_lime_predictor(artefact)
    result_rows: list[dict] = []

    for seed in LIME_SEEDS:
        explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=encoded_train,
            feature_names=artefact["feature_names"],
            categorical_features=categorical_indices,
            categorical_names=categorical_names,
            class_names=["good", "bad"],
            mode="classification",
            discretize_continuous=True,
            discretizer="quartile",
            feature_selection="auto",
            sample_around_instance=False,
            random_state=seed,
        )

        explanation = explainer.explain_instance(
            data_row=encoded_selected,
            predict_fn=predictor,
            labels=(artefact["bad_class_column"],),
            num_features=len(artefact["feature_names"]),
            num_samples=LIME_SAMPLE_COUNT,
        )

        mapped_values = explanation.as_map()[
            artefact["bad_class_column"]
        ]

        ordered_values = sorted(
            mapped_values,
            key=lambda item: abs(item[1]),
            reverse=True,
        )

        for rank, (feature_index, weight) in enumerate(
            ordered_values,
            start=1,
        ):
            result_rows.append(
                {
                    "seed": seed,
                    "rank": rank,
                    "feature_index": int(feature_index),
                    "feature": artefact["feature_names"][
                        feature_index
                    ],
                    "weight_towards_bad_class": float(weight),
                    "absolute_weight": abs(float(weight)),
                    "surrogate_score": float(explanation.score),
                    "surrogate_local_prediction": float(
                        np.asarray(explanation.local_pred).ravel()[0]
                    ),
                }
            )

        figure = explanation.as_pyplot_figure(
            label=artefact["bad_class_column"]
        )
        figure.savefig(
            output_directory / f"lime_seed_{seed}.png",
            dpi=200,
            bbox_inches="tight",
        )
        plt.close(figure)

    results = pd.DataFrame(result_rows)
    results.to_csv(
        output_directory / "lime_all_runs.csv",
        index=False,
    )

    return results


def calculate_lime_stability(
    results: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate feature-frequency and pairwise-overlap measurements."""
    stability_rows = []

    for feature, group in results.groupby("feature"):
        top_three_count = int((group["rank"] <= 3).sum())
        top_five_count = int((group["rank"] <= 5).sum())

        signs = np.sign(group["weight_towards_bad_class"])
        positive_count = int((signs > 0).sum())
        negative_count = int((signs < 0).sum())

        sign_consistency = max(
            positive_count,
            negative_count,
        ) / len(group)

        stability_rows.append(
            {
                "feature": feature,
                "number_of_runs_present": int(
                    group["seed"].nunique()
                ),
                "top_three_count": top_three_count,
                "top_five_count": top_five_count,
                "median_rank": float(group["rank"].median()),
                "mean_weight": float(
                    group["weight_towards_bad_class"].mean()
                ),
                "weight_standard_deviation": float(
                    group["weight_towards_bad_class"].std(ddof=0)
                ),
                "positive_sign_count": positive_count,
                "negative_sign_count": negative_count,
                "sign_consistency_fraction": sign_consistency,
            }
        )

    stability = pd.DataFrame(stability_rows).sort_values(
        ["top_three_count", "top_five_count", "median_rank"],
        ascending=[False, False, True],
    )

    overlap_rows = []

    for first_seed, second_seed in itertools.combinations(
        LIME_SEEDS,
        2,
    ):
        first_features = (
            results.loc[results["seed"] == first_seed]
            .sort_values("rank")["feature"]
            .tolist()
        )
        second_features = (
            results.loc[results["seed"] == second_seed]
            .sort_values("rank")["feature"]
            .tolist()
        )

        overlap_rows.append(
            {
                "first_seed": first_seed,
                "second_seed": second_seed,
                "top_three_jaccard": top_k_jaccard(
                    first_features,
                    second_features,
                    k=3,
                ),
                "top_five_jaccard": top_k_jaccard(
                    first_features,
                    second_features,
                    k=5,
                ),
            }
        )

    return stability, pd.DataFrame(overlap_rows)


def compare_backgrounds(
    first: shap.Explanation,
    second: shap.Explanation,
    selected_position: int,
) -> dict:
    """Quantify local SHAP changes caused by the background choice."""
    first_values = np.asarray(first.values[selected_position])
    second_values = np.asarray(second.values[selected_position])

    first_order = np.argsort(-np.abs(first_values))
    second_order = np.argsort(-np.abs(second_values))

    first_features = [
        first.feature_names[index]
        for index in first_order
    ]
    second_features = [
        second.feature_names[index]
        for index in second_order
    ]

    nonzero_mask = (first_values != 0) | (second_values != 0)

    sign_agreement = np.mean(
        np.sign(first_values[nonzero_mask])
        == np.sign(second_values[nonzero_mask])
    )

    return {
        "absolute_attribution_spearman": safe_spearman(
            np.abs(first_values),
            np.abs(second_values),
        ),
        "top_three_jaccard": top_k_jaccard(
            first_features,
            second_features,
            k=3,
        ),
        "top_five_jaccard": top_k_jaccard(
            first_features,
            second_features,
            k=5,
        ),
        "sign_agreement_fraction": float(sign_agreement),
        "maximum_absolute_attribution_change": float(
            np.max(np.abs(first_values - second_values))
        ),
        "first_background_top_five": first_features[:5],
        "second_background_top_five": second_features[:5],
    }


def save_lime_overlay(
    lime_results: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot LIME weights for all seeds in one figure."""
    top_features = (
        lime_results.groupby("feature")["absolute_weight"]
        .mean()
        .sort_values(ascending=False)
        .head(10)
        .index
    )

    selected = lime_results[
        lime_results["feature"].isin(top_features)
    ]

    pivot = selected.pivot_table(
        index="feature",
        columns="seed",
        values="weight_towards_bad_class",
        fill_value=0.0,
    )

    pivot.plot(
        kind="barh",
        figsize=(10, 7),
    )

    plt.axvline(0, linewidth=1)
    plt.xlabel("LIME coefficient towards bad-risk class")
    plt.ylabel("Original feature")
    plt.title("LIME explanations across five random seeds")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def create_shap_lime_comparison(
    shap_explanation: shap.Explanation,
    lime_stability: pd.DataFrame,
    selected_position: int,
) -> pd.DataFrame:
    """Create a feature-level comparison table."""
    local_shap = pd.DataFrame(
        {
            "feature": shap_explanation.feature_names,
            "shap_value": shap_explanation.values[
                selected_position
            ],
        }
    )

    local_shap["absolute_shap"] = np.abs(
        local_shap["shap_value"]
    )
    local_shap["shap_rank"] = (
        local_shap["absolute_shap"]
        .rank(method="min", ascending=False)
        .astype(int)
    )

    comparison = local_shap.merge(
        lime_stability,
        on="feature",
        how="left",
    )

    comparison["lime_median_rank_or_999"] = (
        comparison["median_rank"].fillna(999)
    )

    comparison["rank_difference"] = np.abs(
        comparison["shap_rank"]
        - comparison["lime_median_rank_or_999"]
    )

    comparison["shap_sign"] = np.sign(
        comparison["shap_value"]
    )
    comparison["lime_mean_sign"] = np.sign(
        comparison["mean_weight"].fillna(0)
    )

    comparison["sign_agreement"] = (
        comparison["shap_sign"]
        == comparison["lime_mean_sign"]
    )

    return comparison.sort_values("shap_rank")


def run_feature_sweeps(
    artefact: dict,
    comparison: pd.DataFrame,
    output_path: Path,
) -> pd.DataFrame:
    """Test disputed features directly against the classifier."""
    selected = artefact["X_test"].iloc[
        artefact["selected_test_position"]
    ].copy()

    disputed = comparison[
        (comparison["rank_difference"] >= 3)
        | (~comparison["sign_agreement"])
        | (comparison["number_of_runs_present"].fillna(0) < 3)
    ]["feature"].head(5)

    rows = []

    for feature in disputed:
        if feature in artefact["categorical_features"]:
            candidate_values = artefact["category_names"][feature]
        else:
            candidate_values = (
                artefact["X_train"][feature]
                .quantile([0.10, 0.25, 0.50, 0.75, 0.90])
                .drop_duplicates()
                .tolist()
            )

        for candidate in candidate_values:
            modified = selected.copy()
            modified[feature] = candidate

            frame = pd.DataFrame(
                [modified],
                columns=artefact["feature_names"],
            )

            transformed = artefact["preprocessor"].transform(frame)

            probability = artefact["model"].predict_proba(
                transformed
            )[0, artefact["bad_class_column"]]

            rows.append(
                {
                    "feature": feature,
                    "candidate_value": str(candidate),
                    "bad_probability": float(probability),
                }
            )

    result = pd.DataFrame(rows)
    result.to_csv(output_path, index=False)
    return result


def main() -> None:
    ensure_directory(OUTPUT_DIRECTORY)
    set_global_seed(42)

    artefact = joblib.load(MODEL_PATH)

    transformed_train = artefact["preprocessor"].transform(
        artefact["X_train"]
    )
    transformed_test = artefact["preprocessor"].transform(
        artefact["X_test"]
    )

    model_probabilities = artefact["model"].predict_proba(
        transformed_test
    )[:, artefact["bad_class_column"]]

    rng = np.random.default_rng(42)

    representative_size = min(
        SHAP_BACKGROUND_SIZE,
        len(transformed_train),
    )
    representative_indices = rng.choice(
        len(transformed_train),
        size=representative_size,
        replace=False,
    )

    good_training_indices = np.flatnonzero(
        artefact["y_train"].to_numpy() == 0
    )
    good_background_size = min(
        SHAP_BACKGROUND_SIZE,
        len(good_training_indices),
    )
    good_background_indices = rng.choice(
        good_training_indices,
        size=good_background_size,
        replace=False,
    )

    background_representative = transformed_train[
        representative_indices
    ]
    background_good_only = transformed_train[
        good_background_indices
    ]

    representative_explanation, _ = make_shap_explanation(
        model=artefact["model"],
        transformed_background=background_representative,
        transformed_evaluation=transformed_test,
        original_evaluation=artefact["X_test"],
        artefact=artefact,
    )

    good_only_explanation, _ = make_shap_explanation(
        model=artefact["model"],
        transformed_background=background_good_only,
        transformed_evaluation=transformed_test,
        original_evaluation=artefact["X_test"],
        artefact=artefact,
    )

    selected_position = artefact["selected_test_position"]

    representative_additivity = save_shap_outputs(
        name="shap_representative_background",
        explanation=representative_explanation,
        selected_position=selected_position,
        model_bad_probabilities=model_probabilities,
        output_directory=OUTPUT_DIRECTORY,
    )

    good_only_additivity = save_shap_outputs(
        name="shap_good_only_background",
        explanation=good_only_explanation,
        selected_position=selected_position,
        model_bad_probabilities=model_probabilities,
        output_directory=OUTPUT_DIRECTORY,
    )

    save_json(
        {
            "representative_background": representative_additivity,
            "good_only_background": good_only_additivity,
        },
        OUTPUT_DIRECTORY / "shap_additivity_checks.json",
    )

    background_comparison = compare_backgrounds(
        representative_explanation,
        good_only_explanation,
        selected_position,
    )

    save_json(
        background_comparison,
        OUTPUT_DIRECTORY / "shap_background_sensitivity.json",
    )

    lime_results = run_lime_experiments(
        artefact,
        OUTPUT_DIRECTORY,
    )

    lime_stability, lime_pairwise_overlap = (
        calculate_lime_stability(lime_results)
    )

    lime_stability.to_csv(
        OUTPUT_DIRECTORY / "lime_stability_by_feature.csv",
        index=False,
    )
    lime_pairwise_overlap.to_csv(
        OUTPUT_DIRECTORY / "lime_pairwise_overlap.csv",
        index=False,
    )

    save_lime_overlay(
        lime_results,
        OUTPUT_DIRECTORY / "lime_seed_overlay.png",
    )

    comparison = create_shap_lime_comparison(
        representative_explanation,
        lime_stability,
        selected_position,
    )

    comparison.to_csv(
        OUTPUT_DIRECTORY / "shap_lime_comparison.csv",
        index=False,
    )

    run_feature_sweeps(
        artefact,
        comparison,
        OUTPUT_DIRECTORY / "disputed_feature_sweeps.csv",
    )

    save_json(
        {
            "target_class": "bad credit risk",
            "shap_output": "predicted bad-class probability",
            "shap_feature_perturbation": "interventional",
            "representative_background_size": (
                representative_size
            ),
            "good_only_background_size": good_background_size,
            "lime_seeds": LIME_SEEDS,
            "lime_samples_per_run": LIME_SAMPLE_COUNT,
            "lime_discretizer": "quartile",
            "lime_sample_around_instance": False,
        },
        OUTPUT_DIRECTORY / "explanation_defaults.json",
    )

    print("Mission 1 explanations completed.")
    print(
        "Do not write the verdict until both students have inspected:"
    )
    print("  - shap_background_sensitivity.json")
    print("  - lime_stability_by_feature.csv")
    print("  - lime_pairwise_overlap.csv")
    print("  - shap_lime_comparison.csv")
    print("  - disputed_feature_sweeps.csv")


if __name__ == "__main__":
    main()

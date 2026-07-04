"""Mission 1, part A: train and validate the credit-risk classifier.

This script deliberately one-hot encodes every categorical feature before
training XGBoost. Native XGBoost categorical splitting is disabled because
the subsequent experiment uses interventional TreeSHAP with explicit
background datasets.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from ucimlrepo import fetch_ucirepo
from xgboost import XGBClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from utilities import (  # noqa: E402
    ensure_directory,
    save_json,
    set_global_seed,
)


RANDOM_SEED = 42
TEST_SIZE = 0.20

OUTPUT_DIRECTORY = PROJECT_ROOT / "outputs" / "mission1"
MODEL_PATH = OUTPUT_DIRECTORY / "credit_model.joblib"


FEATURE_NAMES = [
    "checking_status",
    "duration_months",
    "credit_history",
    "purpose",
    "credit_amount",
    "savings_status",
    "employment_since",
    "installment_rate",
    "personal_status_sex",
    "other_debtors",
    "residence_since",
    "property",
    "age_years",
    "other_installment_plans",
    "housing",
    "existing_credits",
    "job",
    "dependants",
    "telephone",
    "foreign_worker",
]


CATEGORICAL_FEATURES = [
    "checking_status",
    "credit_history",
    "purpose",
    "savings_status",
    "employment_since",
    "personal_status_sex",
    "other_debtors",
    "property",
    "other_installment_plans",
    "housing",
    "job",
    "telephone",
    "foreign_worker",
]


NUMERIC_FEATURES = [
    feature
    for feature in FEATURE_NAMES
    if feature not in CATEGORICAL_FEATURES
]


def load_dataset() -> tuple[pd.DataFrame, pd.Series]:
    """Load and validate the UCI German Credit dataset."""
    dataset = fetch_ucirepo(id=144)

    features = dataset.data.features.copy()
    raw_target = dataset.data.targets.copy()

    if features.shape[1] != 20:
        raise ValueError(
            "Expected 20 input features but received "
            f"{features.shape[1]}."
        )

    features.columns = FEATURE_NAMES

    # Keep categorical variables as symbolic strings until the
    # OneHotEncoder transforms them.
    for column in CATEGORICAL_FEATURES:
        features[column] = features[column].astype(str)

    for column in NUMERIC_FEATURES:
        features[column] = pd.to_numeric(
            features[column],
            errors="raise",
        )

    target_values = pd.to_numeric(
        raw_target.iloc[:, 0],
        errors="raise",
    ).astype(int)

    observed_labels = set(target_values.unique())

    if observed_labels != {1, 2}:
        raise ValueError(
            "The expected raw labels are {1, 2}, where 1 is good "
            f"and 2 is bad. Observed labels: {observed_labels}"
        )

    # Model convention:
    # 0 = good credit risk
    # 1 = bad credit risk
    target = (target_values == 2).astype(int)
    target.name = "bad_credit"

    return features, target


def build_preprocessor() -> ColumnTransformer:
    """Construct the one-hot preprocessing transformation."""
    categorical_encoder = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False,
        dtype=np.float32,
    )

    return ColumnTransformer(
        transformers=[
            (
                "categorical",
                categorical_encoder,
                CATEGORICAL_FEATURES,
            ),
            (
                "numeric",
                "passthrough",
                NUMERIC_FEATURES,
            ),
        ],
        remainder="drop",
        sparse_threshold=0.0,
        verbose_feature_names_out=False,
    )


def build_classifier() -> XGBClassifier:
    """Construct the numerical gradient-boosted classifier.

    All original categorical variables are one-hot encoded before this
    classifier receives them. XGBoost's native categorical processing must
    therefore remain disabled.
    """
    return XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        n_estimators=250,
        max_depth=3,
        learning_rate=0.05,
        min_child_weight=1,
        subsample=0.90,
        colsample_bytree=0.90,
        reg_lambda=1.0,
        tree_method="hist",
        enable_categorical=False,
        random_state=RANDOM_SEED,
        n_jobs=1,
    )


def to_dense_float32_matrix(
    transformed_data: Any,
    matrix_name: str,
) -> np.ndarray:
    """Convert preprocessed data to a finite, dense float32 matrix."""
    if hasattr(transformed_data, "toarray"):
        transformed_data = transformed_data.toarray()

    matrix = np.asarray(
        transformed_data,
        dtype=np.float32,
    )

    matrix = np.ascontiguousarray(matrix)

    if matrix.ndim != 2:
        raise AssertionError(
            f"{matrix_name} must be two-dimensional. "
            f"Observed shape: {matrix.shape}"
        )

    if matrix.dtype != np.float32:
        raise AssertionError(
            f"{matrix_name} must have dtype float32. "
            f"Observed dtype: {matrix.dtype}"
        )

    if matrix.shape[0] == 0:
        raise AssertionError(
            f"{matrix_name} contains no observations."
        )

    if matrix.shape[1] == 0:
        raise AssertionError(
            f"{matrix_name} contains no features."
        )

    if not np.isfinite(matrix).all():
        invalid_count = int(
            np.size(matrix) - np.isfinite(matrix).sum()
        )

        raise ValueError(
            f"{matrix_name} contains {invalid_count} NaN or "
            "infinite values."
        )

    return matrix


def transformed_feature_information(
    preprocessor: ColumnTransformer,
) -> tuple[list[str], list[str], dict[str, list[str]]]:
    """Recover transformed names and original-feature ownership."""
    encoder = preprocessor.named_transformers_["categorical"]

    transformed_names = list(
        encoder.get_feature_names_out(CATEGORICAL_FEATURES)
    )
    transformed_names.extend(NUMERIC_FEATURES)

    transformed_to_original: list[str] = []
    category_names: dict[str, list[str]] = {}

    for feature, categories in zip(
        CATEGORICAL_FEATURES,
        encoder.categories_,
        strict=True,
    ):
        string_categories = [
            str(value)
            for value in categories
        ]

        category_names[feature] = string_categories

        transformed_to_original.extend(
            [feature] * len(string_categories)
        )

    transformed_to_original.extend(NUMERIC_FEATURES)

    if len(transformed_names) != len(transformed_to_original):
        raise AssertionError(
            "Transformed-feature names and the original-feature "
            "mapping have different lengths."
        )

    return (
        transformed_names,
        transformed_to_original,
        category_names,
    )


def inspect_json_for_categorical_splits(
    node: Any,
    path: str = "model",
) -> list[str]:
    """Search an XGBoost JSON model for native categorical split data."""
    findings: list[str] = []

    if isinstance(node, dict):
        for key, value in node.items():
            current_path = f"{path}.{key}"

            if key == "split_type" and isinstance(value, list):
                categorical_node_count = sum(
                    int(split_type) == 1
                    for split_type in value
                )

                if categorical_node_count > 0:
                    findings.append(
                        f"{current_path} contains "
                        f"{categorical_node_count} categorical nodes"
                    )

            if key in {
                "categories",
                "categories_nodes",
                "categories_segments",
                "categories_sizes",
            }:
                if isinstance(value, list) and len(value) > 0:
                    findings.append(
                        f"{current_path} contains "
                        f"{len(value)} categorical entries"
                    )

            findings.extend(
                inspect_json_for_categorical_splits(
                    value,
                    current_path,
                )
            )

    elif isinstance(node, list):
        for index, value in enumerate(node):
            findings.extend(
                inspect_json_for_categorical_splits(
                    value,
                    f"{path}[{index}]",
                )
            )

    return findings


def validate_booster_has_no_categorical_splits(
    classifier: XGBClassifier,
) -> dict[str, Any]:
    """Validate that the trained booster is purely numerical."""
    enable_categorical = classifier.get_params().get(
        "enable_categorical"
    )

    if enable_categorical is True:
        raise AssertionError(
            "XGBoost native categorical handling is enabled. "
            "It must be disabled because the data has already been "
            "one-hot encoded."
        )

    booster = classifier.get_booster()
    booster_feature_types = booster.feature_types

    categorical_feature_positions: list[int] = []

    if booster_feature_types is not None:
        categorical_feature_positions = [
            index
            for index, feature_type in enumerate(
                booster_feature_types
            )
            if str(feature_type).lower() == "c"
        ]

    if categorical_feature_positions:
        raise AssertionError(
            "The booster records native categorical features at "
            f"positions {categorical_feature_positions}. "
            "Interventional TreeSHAP cannot explain this model."
        )

    # Inspect the serialised tree structure as an additional safeguard.
    with tempfile.TemporaryDirectory() as temporary_directory:
        model_json_path = (
            Path(temporary_directory)
            / "xgboost_model_validation.json"
        )

        booster.save_model(model_json_path)

        with model_json_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            model_json = json.load(file)

    categorical_split_findings = (
        inspect_json_for_categorical_splits(model_json)
    )

    if categorical_split_findings:
        formatted_findings = "\n".join(
            f"  - {finding}"
            for finding in categorical_split_findings[:20]
        )

        raise AssertionError(
            "The serialised XGBoost model contains native "
            "categorical split information:\n"
            f"{formatted_findings}"
        )

    return {
        "enable_categorical": enable_categorical,
        "booster_feature_types": booster_feature_types,
        "categorical_feature_positions": (
            categorical_feature_positions
        ),
        "categorical_split_findings": (
            categorical_split_findings
        ),
        "validated_as_purely_numerical": True,
    }


def validate_feature_mapping(
    transformed_train: np.ndarray,
    transformed_test: np.ndarray,
    transformed_names: list[str],
    transformed_to_original: list[str],
) -> None:
    """Check that transformed matrices and feature metadata align."""
    expected_feature_count = len(transformed_names)

    if transformed_train.shape[1] != expected_feature_count:
        raise AssertionError(
            "The transformed training matrix has "
            f"{transformed_train.shape[1]} columns, but the feature "
            f"metadata contains {expected_feature_count} names."
        )

    if transformed_test.shape[1] != expected_feature_count:
        raise AssertionError(
            "The transformed test matrix has "
            f"{transformed_test.shape[1]} columns, but the feature "
            f"metadata contains {expected_feature_count} names."
        )

    if len(transformed_to_original) != expected_feature_count:
        raise AssertionError(
            "The original-feature grouping map has "
            f"{len(transformed_to_original)} entries, but there are "
            f"{expected_feature_count} transformed features."
        )

    unknown_groups = set(transformed_to_original) - set(
        FEATURE_NAMES
    )

    if unknown_groups:
        raise AssertionError(
            "The transformed-feature mapping contains unknown "
            f"original features: {sorted(unknown_groups)}"
        )


def main() -> None:
    set_global_seed(RANDOM_SEED)
    ensure_directory(OUTPUT_DIRECTORY)

    features, target = load_dataset()
    row_identifiers = np.arange(len(features))

    train_ids, test_ids = train_test_split(
        row_identifiers,
        test_size=TEST_SIZE,
        stratify=target,
        random_state=RANDOM_SEED,
    )

    train_features = (
        features
        .iloc[train_ids]
        .reset_index(drop=True)
    )

    test_features = (
        features
        .iloc[test_ids]
        .reset_index(drop=True)
    )

    train_target = (
        target
        .iloc[train_ids]
        .reset_index(drop=True)
    )

    test_target = (
        target
        .iloc[test_ids]
        .reset_index(drop=True)
    )

    preprocessor = build_preprocessor()

    transformed_train_raw = preprocessor.fit_transform(
        train_features
    )

    transformed_test_raw = preprocessor.transform(
        test_features
    )

    transformed_train = to_dense_float32_matrix(
        transformed_train_raw,
        matrix_name="transformed training matrix",
    )

    transformed_test = to_dense_float32_matrix(
        transformed_test_raw,
        matrix_name="transformed test matrix",
    )

    (
        transformed_names,
        transformed_to_original,
        category_names,
    ) = transformed_feature_information(preprocessor)

    validate_feature_mapping(
        transformed_train=transformed_train,
        transformed_test=transformed_test,
        transformed_names=transformed_names,
        transformed_to_original=transformed_to_original,
    )

    classifier = build_classifier()

    classifier.fit(
        transformed_train,
        train_target.to_numpy(dtype=np.int32),
    )

    booster_validation = (
        validate_booster_has_no_categorical_splits(
            classifier
        )
    )

    if set(classifier.classes_) != {0, 1}:
        raise AssertionError(
            f"Unexpected model classes: {classifier.classes_}"
        )

    bad_class_matches = np.flatnonzero(
        classifier.classes_ == 1
    )

    if len(bad_class_matches) != 1:
        raise AssertionError(
            "Exactly one probability column must correspond to "
            f"the bad-risk class. Classes: {classifier.classes_}"
        )

    bad_class_column = int(bad_class_matches[0])

    predicted_probabilities = classifier.predict_proba(
        transformed_test
    )

    if predicted_probabilities.shape != (
        len(test_features),
        2,
    ):
        raise AssertionError(
            "Expected a two-column probability matrix, but received "
            f"shape {predicted_probabilities.shape}."
        )

    bad_probabilities = predicted_probabilities[
        :,
        bad_class_column,
    ]

    predictions = classifier.predict(
        transformed_test
    ).astype(int)

    rejected_positions = np.flatnonzero(
        predictions == 1
    )

    if len(rejected_positions) == 0:
        raise RuntimeError(
            "The model rejected no test applicants. Inspect the "
            "model rather than silently changing the selection rule."
        )

    # Predeclared selection rule:
    # Choose the rejected test applicant with the lowest original row ID.
    rejected_row_ids = test_ids[rejected_positions]

    selected_position = int(
        rejected_positions[
            np.argmin(rejected_row_ids)
        ]
    )

    selected_original_row_id = int(
        test_ids[selected_position]
    )

    selected_prediction = int(
        predictions[selected_position]
    )

    if selected_prediction != 1:
        raise AssertionError(
            "The deterministic selection rule did not select a "
            "rejected applicant."
        )

    metrics = {
        "balanced_accuracy": balanced_accuracy_score(
            test_target,
            predictions,
        ),
        "roc_auc_bad_class": roc_auc_score(
            test_target,
            bad_probabilities,
        ),
        "confusion_matrix_rows_actual_columns_predicted": (
            confusion_matrix(
                test_target,
                predictions,
                labels=[0, 1],
            ).tolist()
        ),
        "classification_report": classification_report(
            test_target,
            predictions,
            labels=[0, 1],
            target_names=["good", "bad"],
            output_dict=True,
            zero_division=0,
        ),
        "model_classes": classifier.classes_.tolist(),
        "bad_class_probability_column": bad_class_column,
        "test_observation_count": len(test_features),
        "predicted_good_count": int(
            np.sum(predictions == 0)
        ),
        "predicted_bad_count": int(
            np.sum(predictions == 1)
        ),
        "selected_test_position": selected_position,
        "selected_original_row_id": selected_original_row_id,
        "selected_true_class": int(
            test_target.iloc[selected_position]
        ),
        "selected_predicted_class": selected_prediction,
        "selected_bad_probability": float(
            bad_probabilities[selected_position]
        ),
    }

    defaults = {
        "dataset": "UCI Statlog German Credit Data",
        "uci_dataset_id": 144,
        "raw_target_mapping": {
            "1": "good",
            "2": "bad",
        },
        "model_target_mapping": {
            "0": "good",
            "1": "bad",
        },
        "split_seed": RANDOM_SEED,
        "model_seed": RANDOM_SEED,
        "test_fraction": TEST_SIZE,
        "stratified_split": True,
        "selection_rule": (
            "Rejected test applicant with the lowest original row ID"
        ),
        "decision_threshold": 0.5,
        "classifier_parameters": classifier.get_params(),
        "categorical_features": CATEGORICAL_FEATURES,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_preprocessing": (
            "OneHotEncoder with handle_unknown='ignore'"
        ),
        "one_hot_output_dtype": "float32",
        "model_input_dtype": str(
            transformed_train.dtype
        ),
        "model_input_is_dense": True,
        "transformed_feature_count": int(
            transformed_train.shape[1]
        ),
        "xgboost_categorical_validation": (
            booster_validation
        ),
    }

    dataset_profile = {
        "number_of_rows": len(features),
        "number_of_features": features.shape[1],
        "bad_class_count": int(target.sum()),
        "good_class_count": int(
            (target == 0).sum()
        ),
        "duplicate_feature_rows": int(
            features.duplicated().sum()
        ),
        "missing_values_by_feature": (
            features.isna().sum().to_dict()
        ),
        "numeric_ranges": {
            feature: {
                "minimum": float(
                    features[feature].min()
                ),
                "maximum": float(
                    features[feature].max()
                ),
                "median": float(
                    features[feature].median()
                ),
            }
            for feature in NUMERIC_FEATURES
        },
        "categorical_values": {
            feature: sorted(
                features[feature].unique().tolist()
            )
            for feature in CATEGORICAL_FEATURES
        },
    }

    selected_record = (
        test_features
        .iloc[selected_position]
        .to_dict()
    )

    selected_applicant = {
        "original_row_id": selected_original_row_id,
        "test_position": selected_position,
        "features": selected_record,
        "true_class": int(
            test_target.iloc[selected_position]
        ),
        "predicted_class": selected_prediction,
        "bad_probability": float(
            bad_probabilities[selected_position]
        ),
        "decision_threshold": 0.5,
    }

    artefact = {
        "preprocessor": preprocessor,
        "model": classifier,
        "feature_names": FEATURE_NAMES,
        "categorical_features": CATEGORICAL_FEATURES,
        "numeric_features": NUMERIC_FEATURES,
        "transformed_feature_names": transformed_names,
        "transformed_to_original": transformed_to_original,
        "category_names": category_names,
        "X_train": train_features,
        "X_test": test_features,
        "y_train": train_target,
        "y_test": test_target,
        "train_original_row_ids": train_ids,
        "test_original_row_ids": test_ids,
        "selected_test_position": selected_position,
        "selected_original_row_id": selected_original_row_id,
        "bad_class_column": bad_class_column,
        "random_seed": RANDOM_SEED,
        "transformed_matrix_dtype": str(
            transformed_train.dtype
        ),
        "transformed_feature_count": int(
            transformed_train.shape[1]
        ),
        "booster_validation": booster_validation,
    }

    # The new artefact replaces the previous incompatible model.
    joblib.dump(
        artefact,
        MODEL_PATH,
    )

    save_json(
        metrics,
        OUTPUT_DIRECTORY / "model_metrics.json",
    )

    save_json(
        defaults,
        OUTPUT_DIRECTORY / "model_defaults.json",
    )

    save_json(
        dataset_profile,
        OUTPUT_DIRECTORY / "dataset_profile.json",
    )

    save_json(
        selected_applicant,
        OUTPUT_DIRECTORY / "selected_applicant.json",
    )

    save_json(
        booster_validation,
        OUTPUT_DIRECTORY
        / "xgboost_categorical_validation.json",
    )

    print("Mission 1 classifier completed.")
    print(f"Artefact: {MODEL_PATH}")
    print(
        "Model input shape: "
        f"{transformed_train.shape}"
    )
    print(
        "Model input dtype: "
        f"{transformed_train.dtype}"
    )
    print(
        "Native categorical splits detected: no"
    )
    print(
        "XGBoost feature types: "
        f"{booster_validation['booster_feature_types']}"
    )
    print(
        "Selected applicant row: "
        f"{selected_original_row_id}"
    )
    print(
        "Selected applicant true class: "
        f"{int(test_target.iloc[selected_position])}"
    )
    print(
        "Selected applicant predicted class: "
        f"{selected_prediction}"
    )
    print(
        "Selected applicant bad-risk probability: "
        f"{bad_probabilities[selected_position]:.4f}"
    )


if __name__ == "__main__":
    main()

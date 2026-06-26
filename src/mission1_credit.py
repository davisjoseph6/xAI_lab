"""Mission 1, part A: train and validate the credit-risk classifier."""

from __future__ import annotations

import sys
from pathlib import Path

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

from utilities import ensure_directory, save_json, set_global_seed  # noqa: E402


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
            f"Expected 20 input features but received {features.shape[1]}."
        )

    features.columns = FEATURE_NAMES

    # Preserve categorical variables as symbolic strings.
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
        dtype=np.float64,
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
        verbose_feature_names_out=False,
    )


def build_classifier() -> XGBClassifier:
    """Construct the gradient-boosted classifier.

    These parameters are not claimed to be optimal. They define one
    reproducible model for the explanation experiment.
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
        random_state=RANDOM_SEED,
        n_jobs=1,
    )


def transformed_feature_information(
    preprocessor: ColumnTransformer,
) -> tuple[list[str], list[str], dict[str, list[str]]]:
    """Recover transformed names and their original-feature ownership."""
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
        string_categories = [str(value) for value in categories]
        category_names[feature] = string_categories

        transformed_to_original.extend(
            [feature] * len(string_categories)
        )

    transformed_to_original.extend(NUMERIC_FEATURES)

    if len(transformed_names) != len(transformed_to_original):
        raise AssertionError(
            "Transformed-feature names and group mapping have "
            "different lengths."
        )

    return (
        transformed_names,
        transformed_to_original,
        category_names,
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

    train_features = features.iloc[train_ids].reset_index(drop=True)
    test_features = features.iloc[test_ids].reset_index(drop=True)

    train_target = target.iloc[train_ids].reset_index(drop=True)
    test_target = target.iloc[test_ids].reset_index(drop=True)

    preprocessor = build_preprocessor()

    transformed_train = preprocessor.fit_transform(train_features)
    transformed_test = preprocessor.transform(test_features)

    classifier = build_classifier()
    classifier.fit(transformed_train, train_target)

    if set(classifier.classes_) != {0, 1}:
        raise AssertionError(
            f"Unexpected model classes: {classifier.classes_}"
        )

    bad_class_column = int(
        np.flatnonzero(classifier.classes_ == 1)[0]
    )

    predicted_probabilities = classifier.predict_proba(
        transformed_test
    )
    bad_probabilities = predicted_probabilities[:, bad_class_column]
    predictions = classifier.predict(transformed_test)

    rejected_positions = np.flatnonzero(predictions == 1)

    if len(rejected_positions) == 0:
        raise RuntimeError(
            "The model rejected no test applicants. Inspect the model "
            "rather than silently changing the selection rule."
        )

    # Predeclared selection rule:
    # Choose the rejected test applicant with the lowest original row ID.
    rejected_row_ids = test_ids[rejected_positions]
    selected_position = int(
        rejected_positions[np.argmin(rejected_row_ids)]
    )
    selected_original_row_id = int(test_ids[selected_position])

    (
        transformed_names,
        transformed_to_original,
        category_names,
    ) = transformed_feature_information(preprocessor)

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
            confusion_matrix(test_target, predictions).tolist()
        ),
        "classification_report": classification_report(
            test_target,
            predictions,
            target_names=["good", "bad"],
            output_dict=True,
            zero_division=0,
        ),
        "model_classes": classifier.classes_.tolist(),
        "bad_class_probability_column": bad_class_column,
        "selected_test_position": selected_position,
        "selected_original_row_id": selected_original_row_id,
        "selected_true_class": int(test_target.iloc[selected_position]),
        "selected_predicted_class": int(predictions[selected_position]),
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
        "selection_rule": (
            "Rejected test applicant with the lowest original row ID"
        ),
        "decision_threshold": 0.5,
        "classifier_parameters": classifier.get_params(),
        "categorical_features": CATEGORICAL_FEATURES,
        "numeric_features": NUMERIC_FEATURES,
    }

    dataset_profile = {
        "number_of_rows": len(features),
        "number_of_features": features.shape[1],
        "bad_class_count": int(target.sum()),
        "good_class_count": int((target == 0).sum()),
        "duplicate_feature_rows": int(features.duplicated().sum()),
        "missing_values_by_feature": (
            features.isna().sum().to_dict()
        ),
        "numeric_ranges": {
            feature: {
                "minimum": float(features[feature].min()),
                "maximum": float(features[feature].max()),
                "median": float(features[feature].median()),
            }
            for feature in NUMERIC_FEATURES
        },
        "categorical_values": {
            feature: sorted(features[feature].unique().tolist())
            for feature in CATEGORICAL_FEATURES
        },
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
        "bad_class_column": bad_class_column,
        "random_seed": RANDOM_SEED,
    }

    joblib.dump(artefact, MODEL_PATH)

    save_json(metrics, OUTPUT_DIRECTORY / "model_metrics.json")
    save_json(defaults, OUTPUT_DIRECTORY / "model_defaults.json")
    save_json(
        dataset_profile,
        OUTPUT_DIRECTORY / "dataset_profile.json",
    )

    selected_record = test_features.iloc[
        selected_position
    ].to_dict()

    save_json(
        {
            "original_row_id": selected_original_row_id,
            "features": selected_record,
            "true_class": int(test_target.iloc[selected_position]),
            "predicted_class": int(predictions[selected_position]),
            "bad_probability": float(
                bad_probabilities[selected_position]
            ),
        },
        OUTPUT_DIRECTORY / "selected_applicant.json",
    )

    print("Mission 1 classifier completed.")
    print(f"Artefact: {MODEL_PATH}")
    print(f"Selected applicant row: {selected_original_row_id}")
    print(
        "Selected applicant bad-risk probability: "
        f"{bad_probabilities[selected_position]:.4f}"
    )


if __name__ == "__main__":
    main()

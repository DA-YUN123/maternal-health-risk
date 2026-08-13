from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from src.config import FEATURES, LABELS, RANDOM_STATE, TARGET


@dataclass
class TrainingResult:
    best_model: Pipeline
    best_model_name: str
    best_params: dict
    comparison: pd.DataFrame
    cv_results: pd.DataFrame
    detailed_metrics: dict
    confusion_matrices: dict[str, list[list[int]]]
    feature_importance: pd.DataFrame
    test_predictions: pd.DataFrame
    split_summary: dict


def _model_specs() -> dict:
    scaled_features = ColumnTransformer(
        [("numeric", StandardScaler(), FEATURES)],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return {
        "logistic_regression": {
            "pipeline": Pipeline(
                [
                    ("preprocess", scaled_features),
                    (
                        "model",
                        LogisticRegression(
                            class_weight="balanced",
                            max_iter=5000,
                            random_state=RANDOM_STATE,
                        ),
                    ),
                ]
            ),
            "parameters": {"model__C": [0.1, 1.0, 10.0]},
        },
        "decision_tree": {
            "pipeline": Pipeline(
                [
                    (
                        "model",
                        DecisionTreeClassifier(
                            class_weight="balanced",
                            random_state=RANDOM_STATE,
                        ),
                    )
                ]
            ),
            "parameters": {
                "model__max_depth": [3, 5, None],
                "model__min_samples_leaf": [1, 3, 5],
            },
        },
        "random_forest": {
            "pipeline": Pipeline(
                [
                    (
                        "model",
                        RandomForestClassifier(
                            class_weight="balanced_subsample",
                            n_estimators=250,
                            n_jobs=-1,
                            random_state=RANDOM_STATE,
                        ),
                    )
                ]
            ),
            "parameters": {
                "model__max_depth": [5, None],
                "model__min_samples_leaf": [1, 3],
                "model__max_features": ["sqrt"],
            },
        },
    }


def train_and_evaluate(clean_frame: pd.DataFrame) -> TrainingResult:
    X = clean_frame[FEATURES].copy()
    y = clean_frame[TARGET].astype("string")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    cross_validation = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    scoring = {
        "macro_f1": "f1_macro",
        "balanced_accuracy": "balanced_accuracy",
        "accuracy": "accuracy",
    }

    fitted_models: dict[str, Pipeline] = {}
    comparison_rows: list[dict] = []
    cv_rows: list[pd.DataFrame] = []
    detailed_metrics: dict = {}
    confusion_matrices: dict[str, list[list[int]]] = {}
    predictions_by_model: dict[str, np.ndarray] = {}
    probabilities_by_model: dict[str, np.ndarray] = {}

    for model_name, spec in _model_specs().items():
        search = GridSearchCV(
            estimator=spec["pipeline"],
            param_grid=spec["parameters"],
            scoring=scoring,
            refit="macro_f1",
            cv=cross_validation,
            n_jobs=1,
            return_train_score=False,
        )
        search.fit(X_train, y_train)
        fitted_models[model_name] = search.best_estimator_

        best_index = search.best_index_
        prediction = search.best_estimator_.predict(X_test)
        probability = search.best_estimator_.predict_proba(X_test)
        predictions_by_model[model_name] = prediction
        probabilities_by_model[model_name] = probability

        accuracy = float(accuracy_score(y_test, prediction))
        balanced_accuracy = float(balanced_accuracy_score(y_test, prediction))
        macro_f1 = float(f1_score(y_test, prediction, average="macro"))
        weighted_f1 = float(f1_score(y_test, prediction, average="weighted"))
        comparison_rows.append(
            {
                "model": model_name,
                "cv_macro_f1_mean": float(search.cv_results_["mean_test_macro_f1"][best_index]),
                "cv_macro_f1_std": float(search.cv_results_["std_test_macro_f1"][best_index]),
                "cv_balanced_accuracy_mean": float(
                    search.cv_results_["mean_test_balanced_accuracy"][best_index]
                ),
                "holdout_accuracy": accuracy,
                "holdout_balanced_accuracy": balanced_accuracy,
                "holdout_macro_f1": macro_f1,
                "holdout_weighted_f1": weighted_f1,
                "best_parameters": json.dumps(search.best_params_, ensure_ascii=False),
            }
        )

        report = classification_report(
            y_test,
            prediction,
            labels=LABELS,
            output_dict=True,
            zero_division=0,
        )
        detailed_metrics[model_name] = report
        confusion_matrices[model_name] = confusion_matrix(
            y_test,
            prediction,
            labels=LABELS,
        ).tolist()

        model_cv = pd.DataFrame(search.cv_results_)
        keep_columns = [
            "params",
            "mean_test_macro_f1",
            "std_test_macro_f1",
            "mean_test_balanced_accuracy",
            "mean_test_accuracy",
            "rank_test_macro_f1",
        ]
        model_cv = model_cv[keep_columns].copy()
        model_cv.insert(0, "model", model_name)
        model_cv["params"] = model_cv["params"].map(
            lambda value: json.dumps(value, ensure_ascii=False, sort_keys=True)
        )
        cv_rows.append(model_cv)

    comparison = pd.DataFrame(comparison_rows).sort_values(
        ["cv_macro_f1_mean", "model"], ascending=[False, True]
    ).reset_index(drop=True)
    best_model_name = str(comparison.iloc[0]["model"])
    best_model = fitted_models[best_model_name]

    importance_result = permutation_importance(
        best_model,
        X_test,
        y_test,
        scoring="f1_macro",
        n_repeats=30,
        random_state=RANDOM_STATE,
        n_jobs=1,
    )
    feature_importance = pd.DataFrame(
        {
            "feature": FEATURES,
            "importance_mean": importance_result.importances_mean,
            "importance_std": importance_result.importances_std,
        }
    ).sort_values("importance_mean", ascending=False, ignore_index=True)

    best_prediction = predictions_by_model[best_model_name]
    best_probability = probabilities_by_model[best_model_name]
    test_predictions = X_test.reset_index(drop=False).rename(columns={"index": "source_row_index"})
    test_predictions["actual"] = y_test.reset_index(drop=True)
    test_predictions["predicted"] = best_prediction
    model_classes = list(best_model.classes_)
    for class_index, label in enumerate(model_classes):
        test_predictions[f"probability_{label.replace(' ', '_')}"] = best_probability[:, class_index]

    split_summary = {
        "train_rows": int(len(X_train)),
        "holdout_rows": int(len(X_test)),
        "test_size": 0.20,
        "random_state": RANDOM_STATE,
        "cross_validation": "StratifiedKFold(n_splits=5, shuffle=True, random_state=42)",
        "model_selection_metric": "cross-validated macro F1 on the training split",
        "train_class_counts": {
            label: int((y_train == label).sum()) for label in LABELS
        },
        "holdout_class_counts": {
            label: int((y_test == label).sum()) for label in LABELS
        },
    }
    best_params = json.loads(
        comparison.loc[comparison["model"] == best_model_name, "best_parameters"].iloc[0]
    )
    return TrainingResult(
        best_model=best_model,
        best_model_name=best_model_name,
        best_params=best_params,
        comparison=comparison,
        cv_results=pd.concat(cv_rows, ignore_index=True),
        detailed_metrics=detailed_metrics,
        confusion_matrices=confusion_matrices,
        feature_importance=feature_importance,
        test_predictions=test_predictions,
        split_summary=split_summary,
    )


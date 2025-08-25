import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

try:
    from xgboost import XGBClassifier  # type: ignore
except Exception:  # pragma: no cover - xgboost optional at import time
    XGBClassifier = None  # type: ignore


RANDOM_SEED = 42
ARTIFACTS_DIR = Path("artifacts")
DATA_PATH = Path("diabetes_prediction_dataset.csv")


NUMERIC_FEATURES = [
    "age",
    "bmi",
    "HbA1c_level",
    "blood_glucose_level",
]

BINARY_NUMERIC = [
    "hypertension",
    "heart_disease",
]

CATEGORICAL_FEATURES = [
    "gender",
    "smoking_history",
]

TARGET_COLUMN = "diabetes"


@dataclass
class ModelResult:
    model_name: str
    pipeline: Pipeline
    val_auc: float
    threshold: float
    metrics: Dict[str, float]


def ensure_artifacts_dir() -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


def build_preprocessors() -> Tuple[ColumnTransformer, ColumnTransformer]:
    numeric_all = NUMERIC_FEATURES + BINARY_NUMERIC
    basic_numeric = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )
    scaled_numeric = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocess_basic = ColumnTransformer(
        transformers=[
            ("num", basic_numeric, numeric_all),
            ("cat", categorical, CATEGORICAL_FEATURES),
        ]
    )
    preprocess_scaled = ColumnTransformer(
        transformers=[
            ("num", scaled_numeric, numeric_all),
            ("cat", categorical, CATEGORICAL_FEATURES),
        ]
    )
    return preprocess_basic, preprocess_scaled


def get_models(preprocess_basic: ColumnTransformer, preprocess_scaled: ColumnTransformer) -> Dict[str, Pipeline]:
    models: Dict[str, Pipeline] = {}

    models["logreg"] = Pipeline(
        steps=[
            ("preprocess", preprocess_basic),
            ("clf", LogisticRegression(max_iter=1000, n_jobs=None)),
        ]
    )

    models["rf"] = Pipeline(
        steps=[
            ("preprocess", preprocess_basic),
            ("clf", RandomForestClassifier(n_estimators=300, random_state=RANDOM_SEED, n_jobs=-1)),
        ]
    )

    if XGBClassifier is not None:
        models["xgb"] = Pipeline(
            steps=[
                ("preprocess", preprocess_basic),
                (
                    "clf",
                    XGBClassifier(
                        n_estimators=300,
                        learning_rate=0.1,
                        max_depth=4,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        random_state=RANDOM_SEED,
                        eval_metric="logloss",
                        n_jobs=-1,
                        reg_lambda=1.0,
                    ),
                ),
            ]
        )

    models["svm_rbf"] = Pipeline(
        steps=[
            ("preprocess", preprocess_scaled),
            ("clf", SVC(kernel="rbf", probability=True, C=1.0, gamma="scale", random_state=RANDOM_SEED)),
        ]
    )

    models["mlp"] = Pipeline(
        steps=[
            ("preprocess", preprocess_scaled),
            ("clf", MLPClassifier(hidden_layer_sizes=(64, 32), activation="relu", max_iter=200, random_state=RANDOM_SEED)),
        ]
    )

    return models


def find_threshold(y_true: np.ndarray, y_proba: np.ndarray, target_recall: float = 0.90, min_specificity: float = 0.80) -> Tuple[float, Dict[str, float]]:
    thresholds = np.linspace(0.01, 0.99, 99)
    best = {
        "threshold": 0.5,
        "recall": 0.0,
        "specificity": 0.0,
        "f1": 0.0,
        "accuracy": 0.0,
    }
    selected_threshold = 0.5

    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        accuracy = accuracy_score(y_true, y_pred)
        _, _, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)

        meets = (recall >= target_recall) and (specificity >= min_specificity)
        if meets:
            # prioritize highest specificity, then highest recall, then f1
            if (
                (specificity > best["specificity"]) or
                (np.isclose(specificity, best["specificity"]) and recall > best["recall"]) or
                (np.isclose(specificity, best["specificity"]) and np.isclose(recall, best["recall"]) and f1 > best["f1"]) 
            ):
                best = {
                    "threshold": float(t),
                    "recall": float(recall),
                    "specificity": float(specificity),
                    "f1": float(f1),
                    "accuracy": float(accuracy),
                }
                selected_threshold = float(t)

    if selected_threshold == 0.5 and best["recall"] == 0.0:
        # fallback: choose threshold that maximizes F1
        f1_best = -1.0
        for t in thresholds:
            y_pred = (y_proba >= t).astype(int)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            if f1 > f1_best:
                f1_best = f1
                selected_threshold = float(t)
        # recompute metrics at selected threshold
        y_pred = (y_proba >= selected_threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        accuracy = accuracy_score(y_true, y_pred)
        _, _, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
        best = {
            "threshold": float(selected_threshold),
            "recall": float(recall),
            "specificity": float(specificity),
            "f1": float(f1),
            "accuracy": float(accuracy),
        }

    return selected_threshold, best


def evaluate_model(name: str, pipeline: Pipeline, X_val: pd.DataFrame, y_val: pd.Series) -> ModelResult:
    proba = pipeline.predict_proba(X_val)[:, 1]
    val_auc = roc_auc_score(y_val, proba)
    threshold, metrics = find_threshold(y_val.values, proba)
    return ModelResult(model_name=name, pipeline=pipeline, val_auc=val_auc, threshold=threshold, metrics=metrics)


def plot_roc(pipeline: Pipeline, X: pd.DataFrame, y: pd.Series, out_path: Path) -> None:
    proba = pipeline.predict_proba(X)[:, 1]
    fpr, tpr, _ = roc_curve(y, proba)
    score = auc(fpr, tpr)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"ROC AUC = {score:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve (Test)")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_confusion(y_true: np.ndarray, y_pred: np.ndarray, out_path: Path) -> None:
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    fig, ax = plt.subplots(figsize=(5, 4))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    plt.title("Confusion Matrix (Test @ threshold)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def compute_shap_summary(pipeline: Pipeline, X_sample: pd.DataFrame, out_path: Path) -> None:
    try:
        model = pipeline.named_steps["clf"]
        # extract transformed feature names
        pre = pipeline.named_steps["preprocess"]
        # preprocessor is already fitted within the trained pipeline; do not refit here
        try:
            feature_names = pre.get_feature_names_out().tolist()
        except Exception:
            feature_names = [f"f{i}" for i in range(pre.transform(X_sample).shape[1])]

        X_trans = pre.transform(X_sample)

        if hasattr(model, "get_booster") or model.__class__.__name__.lower().startswith("randomforest"):
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_trans)
            # For XGB binary, shap_values may be array; for RF list with two classes
            if isinstance(shap_values, list):
                values = shap_values[1]
            else:
                values = shap_values
        elif model.__class__.__name__.lower().startswith("logisticregression"):
            explainer = shap.LinearExplainer(model, X_trans)
            values = explainer.shap_values(X_trans)
        else:
            # fallback KernelExplainer on small subset
            sample_idx = np.random.RandomState(RANDOM_SEED).choice(np.arange(X_trans.shape[0]), size=min(100, X_trans.shape[0]), replace=False)
            background = X_trans[sample_idx]
            explainer = shap.KernelExplainer(model.predict_proba, background)
            values = explainer.shap_values(X_trans[:100])[1]

        plt.figure(figsize=(8, 6))
        shap.summary_plot(values, features=X_trans, feature_names=feature_names, show=False)
        plt.tight_layout()
        plt.savefig(out_path, dpi=160, bbox_inches="tight")
        plt.close()
    except Exception as e:
        # graceful degradation
        with open(out_path.with_suffix(".txt"), "w") as f:
            f.write(f"SHAP summary generation failed: {e}\n")


def plot_local_explanations(pipeline: Pipeline, X_sample: pd.DataFrame, out_path: Path) -> None:
    try:
        model = pipeline.named_steps["clf"]
        pre = pipeline.named_steps["preprocess"]
        # use the already-fitted preprocessor from the trained pipeline
        X_trans = pre.transform(X_sample)
        try:
            feature_names = pre.get_feature_names_out().tolist()
        except Exception:
            feature_names = [f"f{i}" for i in range(X_trans.shape[1])]

        if hasattr(model, "get_booster") or model.__class__.__name__.lower().startswith("randomforest"):
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_trans)
            if isinstance(shap_values, list):
                values = shap_values[1]
            else:
                values = shap_values
        elif model.__class__.__name__.lower().startswith("logisticregression"):
            explainer = shap.LinearExplainer(model, X_trans)
            values = explainer.shap_values(X_trans)
        else:
            sample_idx = np.random.RandomState(RANDOM_SEED).choice(np.arange(X_trans.shape[0]), size=min(100, X_trans.shape[0]), replace=False)
            background = X_trans[sample_idx]
            explainer = shap.KernelExplainer(model.predict_proba, background)
            values = explainer.shap_values(X_trans[:50])[1]

        # pick two cases: highest and lowest predicted risk
        proba = pipeline.predict_proba(X_sample)[:, 1]
        idx_high = int(np.argmax(proba))
        idx_low = int(np.argmin(proba))

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        shap.plots.bar(shap.Explanation(values=values[idx_high], feature_names=feature_names), show=False, max_display=10, ax=axes[0])
        axes[0].set_title("High-risk case: top features")
        shap.plots.bar(shap.Explanation(values=values[idx_low], feature_names=feature_names), show=False, max_display=10, ax=axes[1])
        axes[1].set_title("Low-risk case: top features")
        plt.tight_layout()
        plt.savefig(out_path, dpi=160)
        plt.close()
    except Exception as e:
        with open(out_path.with_suffix(".txt"), "w") as f:
            f.write(f"Local explanations generation failed: {e}\n")


def plot_manual_pdp(pipeline: Pipeline, X_ref: pd.DataFrame, feature: str, grid: np.ndarray, out_path: Path) -> None:
    X_base = X_ref.copy()
    preds = []
    for v in grid:
        X_mod = X_base.copy()
        X_mod[feature] = v
        p = pipeline.predict_proba(X_mod)[:, 1].mean()
        preds.append(p)
    plt.figure(figsize=(6, 5))
    plt.plot(grid, preds, marker="o")
    plt.xlabel(feature)
    plt.ylabel("Predicted risk (mean)")
    plt.title(f"PDP for {feature}")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def main() -> None:
    ensure_artifacts_dir()
    df = pd.read_csv(DATA_PATH)

    features = NUMERIC_FEATURES + BINARY_NUMERIC + CATEGORICAL_FEATURES
    X = df[features]
    y = df[TARGET_COLUMN].astype(int)

    # Train/Val/Test split: 70/15/15 stratified
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=RANDOM_SEED
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=RANDOM_SEED
    )

    preprocess_basic, preprocess_scaled = build_preprocessors()
    model_pipelines = get_models(preprocess_basic, preprocess_scaled)

    results: List[ModelResult] = []
    for name, pipe in model_pipelines.items():
        pipe.fit(X_train, y_train)
        res = evaluate_model(name, pipe, X_val, y_val)
        results.append(res)

    # Write metrics.csv (validation)
    rows = []
    for r in results:
        rows.append({
            "model": r.model_name,
            "val_auc": r.val_auc,
            "threshold": r.threshold,
            "recall": r.metrics["recall"],
            "specificity": r.metrics["specificity"],
            "f1": r.metrics["f1"],
            "accuracy": r.metrics["accuracy"],
        })
    metrics_df = pd.DataFrame(rows).sort_values(by=["recall", "specificity", "val_auc"], ascending=False)
    metrics_path = ARTIFACTS_DIR / "metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)

    # Choose best model: first that meets recall>=0.90 and spec>=0.80, highest AUC; fallback highest AUC
    eligible = [r for r in results if (r.metrics["recall"] >= 0.90 and r.metrics["specificity"] >= 0.80)]
    if eligible:
        best = sorted(eligible, key=lambda r: (r.val_auc, r.metrics["specificity"], r.metrics["recall"]), reverse=True)[0]
    else:
        best = sorted(results, key=lambda r: r.val_auc, reverse=True)[0]

    # Refit best on train+val
    best_pipeline = model_pipelines[best.model_name]
    X_trval = pd.concat([X_train, X_val], axis=0)
    y_trval = pd.concat([y_train, y_val], axis=0)
    best_pipeline.fit(X_trval, y_trval)

    # Evaluate on test at locked threshold
    test_proba = best_pipeline.predict_proba(X_test)[:, 1]
    y_test_pred = (test_proba >= best.threshold).astype(int)
    cm = confusion_matrix(y_test, y_test_pred)
    tn, fp, fn, tp = cm.ravel()
    test_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    test_specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_f1 = f1_score(y_test, y_test_pred, zero_division=0)
    test_auc = roc_auc_score(y_test, test_proba)

    # Save plots
    plot_roc(best_pipeline, X_test, y_test, ARTIFACTS_DIR / "roc.png")
    plot_confusion(y_test.values, y_test_pred, ARTIFACTS_DIR / "confusion_matrix.png")

    # SHAP summary and local explanations on a sample of test
    sample = X_test.sample(n=min(500, len(X_test)), random_state=RANDOM_SEED)
    compute_shap_summary(best_pipeline, sample, ARTIFACTS_DIR / "shap_summary.png")
    plot_local_explanations(best_pipeline, sample, ARTIFACTS_DIR / "local_explanations.png")

    # PDPs for glucose and BMI (manual)
    # Use ranges based on percentiles
    for feat, out_name in [("blood_glucose_level", "pdp_glucose.png"), ("bmi", "pdp_bmi.png")]:
        vals = X_trval[feat]
        grid = np.linspace(np.percentile(vals, 1), np.percentile(vals, 99), 25)
        plot_manual_pdp(best_pipeline, sample, feat, grid, ARTIFACTS_DIR / out_name)

    # Persist best model and threshold
    joblib.dump(best_pipeline, ARTIFACTS_DIR / "best_model.joblib")
    with open(ARTIFACTS_DIR / "threshold.json", "w") as f:
        json.dump({
            "model": best.model_name,
            "threshold": best.threshold,
            "val_auc": best.val_auc,
            "val_metrics": best.metrics,
            "test_metrics": {
                "recall": test_recall,
                "specificity": test_specificity,
                "accuracy": test_accuracy,
                "f1": test_f1,
                "auc": test_auc,
            },
        }, f, indent=2)

    # Write model selection notes
    notes = []
    notes.append(f"Best model: {best.model_name}")
    notes.append(f"Validation AUC: {best.val_auc:.4f}")
    notes.append(f"Chosen threshold (val): {best.threshold:.2f}")
    notes.append("Validation metrics at threshold:")
    for k in ["recall", "specificity", "accuracy", "f1"]:
        notes.append(f"- {k}: {best.metrics[k]:.4f}")
    notes.append("\nTest metrics at locked threshold:")
    notes.append(f"- recall: {test_recall:.4f}")
    notes.append(f"- specificity: {test_specificity:.4f}")
    notes.append(f"- accuracy: {test_accuracy:.4f}")
    notes.append(f"- f1: {test_f1:.4f}")
    notes.append(f"- auc: {test_auc:.4f}")
    Path("artifacts/model_selection_notes.md").write_text("\n".join(notes))

    print("Training complete. Artifacts written to artifacts/ directory.")


if __name__ == "__main__":
    main()



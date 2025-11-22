import os
import json
from datetime import datetime

import numpy as np
import pandas as pd

from datasets import load_dataset
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
import joblib
from huggingface_hub import HfApi, login

# =========================
# CONFIG
# =========================
HF_USERNAME = "amitcoolll"
HF_DATASET_REPO = f"{HF_USERNAME}/predcitiveanalysis"    # dataset space
HF_MODEL_REPO   = f"{HF_USERNAME}/predictiveanalysis"    # model hub repo

TARGET_COL = "Engine_Condition"
RANDOM_STATE = 42

NUM_COLS = [
    "Engine_RPM",
    "Lub_Oil_Pressure",
    "Fuel_Pressure",
    "Coolant_Pressure",
    "Lub_Oil_Temperature",
    "Coolant_Temperature",
]

DATA_DIR = "data"
ARTIFACT_DIR = "artifacts"

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(ARTIFACT_DIR, exist_ok=True)


def load_and_normalize():
    """
    Load dataset from Hugging Face dataset space and normalize column names
    (same logic as your notebook, but without Colab upload).
    """
    print(f"📥 Loading dataset from HF: {HF_DATASET_REPO}")
    ds = load_dataset(HF_DATASET_REPO)
    # use the 'train' split as full dataset for processing
    df = ds["train"].to_pandas()

    # Rename columns if original names exist
    rename_map = {
        "Engine rpm": "Engine_RPM",
        "Lub oil pressure": "Lub_Oil_Pressure",
        "Fuel pressure": "Fuel_Pressure",
        "Coolant pressure": "Coolant_Pressure",
        "lub oil temp": "Lub_Oil_Temperature",
        "Coolant temp": "Coolant_Temperature",
        "Engine Condition": "Engine_Condition",
    }
    df = df.rename(columns=rename_map)

    expected = [
        "Engine_RPM",
        "Lub_Oil_Pressure",
        "Fuel_Pressure",
        "Coolant_Pressure",
        "Lub_Oil_Temperature",
        "Coolant_Temperature",
        "Engine_Condition",
    ]
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")

    for c in expected[:-1]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["Engine_Condition"] = pd.to_numeric(
        df["Engine_Condition"], errors="coerce"
    ).astype("Int64")

    print("✅ Data loaded & normalized. Shape:", df.shape)
    return df


def clean_and_split(df: pd.DataFrame):
    """
    Cleaning, outlier clipping, train-test split and saving train/test CSVs.
    """
    print("🧹 Cleaning & splitting data…")
    dfc = df.copy().drop_duplicates()

    # Clip outliers at [1%, 99%]
    for c in NUM_COLS:
        lo, hi = dfc[c].quantile([0.01, 0.99])
        dfc[c] = dfc[c].clip(lower=lo, upper=hi)

    # Drop rows with missing target
    dfc = dfc[dfc["Engine_Condition"].notna()].copy()
    dfc["Engine_Condition"] = dfc["Engine_Condition"].astype(int)

    X, y = dfc[NUM_COLS], dfc["Engine_Condition"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)

    train_path = os.path.join(DATA_DIR, "train.csv")
    test_path = os.path.join(DATA_DIR, "test.csv")

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"✅ Saved {train_path} and {test_path}")
    return X_train, X_test, y_train, y_test


def build_and_train(X_train, X_test, y_train, y_test):
    """
    Train, tune and evaluate DecisionTree, RandomForest, GradientBoosting models.
    Save experiment logs and best model.
    """
    print("🤖 Training, tuning & tracking models…")

    preprocess = ColumnTransformer(
        [
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                NUM_COLS,
            )
        ]
    )

    candidates = [
        (
            "DecisionTree",
            DecisionTreeClassifier(random_state=RANDOM_STATE),
            {"clf__max_depth": [5, None]},
        ),
        (
            "RandomForest",
            RandomForestClassifier(random_state=RANDOM_STATE),
            {"clf__n_estimators": [100], "clf__max_depth": [None]},
        ),
        (
            "GradientBoosting",
            GradientBoostingClassifier(random_state=RANDOM_STATE),
            {"clf__n_estimators": [100], "clf__learning_rate": [0.1], "clf__max_depth": [3]},
        ),
    ]

    def evaluate_model(name, model, X_te, y_te):
        y_pred = model.predict(X_te)
        try:
            y_proba = model.predict_proba(X_te)[:, 1]
            roc = roc_auc_score(y_te, y_proba)
        except Exception:
            roc = np.nan
        return {
            "model": name,
            "accuracy": accuracy_score(y_te, y_pred),
            "precision": precision_score(y_te, y_pred, zero_division=0),
            "recall": recall_score(y_te, y_pred, zero_division=0),
            "f1": f1_score(y_te, y_pred, zero_division=0),
            "roc_auc": roc,
            "confusion_matrix": confusion_matrix(y_te, y_pred).tolist(),
        }

    experiments = []
    best_f1 = -1.0
    best_name, best_model, best_metrics = None, None, None

    for name, estimator, grid in candidates:
        print(f"🔄 Tuning {name}…")
        pipe = Pipeline([("preprocess", preprocess), ("clf", estimator)])
        gs = GridSearchCV(
            pipe,
            param_grid=grid,
            cv=3,
            scoring="f1",
            n_jobs=-1,
        )
        gs.fit(X_train, y_train)

        metrics = evaluate_model(name, gs.best_estimator_, X_test, y_test)
        experiments.append(
            {
                "timestamp": datetime.now().isoformat(),
                "model_name": name,
                "best_params": json.dumps(gs.best_params_),
                "cv_best_score_f1": gs.best_score_,
                "test_accuracy": metrics["accuracy"],
                "test_precision": metrics["precision"],
                "test_recall": metrics["recall"],
                "test_f1": metrics["f1"],
                "test_roc_auc": metrics["roc_auc"],
            }
        )

        print(
            f"✅ {name}: Test F1={metrics['f1']:.4f}, "
            f"Acc={metrics['accuracy']:.4f}"
        )

        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_name = name
            best_model = gs.best_estimator_
            best_metrics = metrics

    # Save experiments log
    exp_df = pd.DataFrame(experiments)
    exp_log_path = os.path.join(ARTIFACT_DIR, "experiments_log.csv")
    exp_df.to_csv(exp_log_path, index=False)
    print(f"📝 Experiments log saved to {exp_log_path}")

    # Save best model
    best_model_path = os.path.join(ARTIFACT_DIR, "best_model.joblib")
    joblib.dump(best_model, best_model_path)
    print(f"🏆 Best model ({best_name}) saved to {best_model_path}")

    return best_name, best_f1, best_metrics, best_model_path, exp_log_path


def upload_best_model(best_name, best_f1, best_metrics, best_model_path, exp_log_path):
    """
    Upload best model and README to Hugging Face model repo, using HF_TOKEN env var.
    """
    print(f"☁️ Uploading best model to Hugging Face: {HF_MODEL_REPO}")

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        print("⚠️ HF_TOKEN environment variable not set. Skipping model upload.")
        return

    login(token=hf_token)
    api = HfApi()

    api.create_repo(
        repo_id=HF_MODEL_REPO,
        repo_type="model",
        exist_ok=True,
    )

    # Upload model file
    api.upload_file(
        path_or_fileobj=best_model_path,
        path_in_repo="best_model.joblib",
        repo_id=HF_MODEL_REPO,
        repo_type="model",
    )

    # Build README text
    best_acc = best_metrics["accuracy"]
    best_params = None
    try:
        exp_df = pd.read_csv(exp_log_path)
        best_row = exp_df.sort_values("test_f1", ascending=False).iloc[0]
        best_params = best_row.get("best_params", None)
    except Exception:
        pass

    readme_lines = [
        "# Predictive Maintenance – Engine Condition Model",
        "",
        "This repository contains the **best-performing model** trained on engine sensor data",
        "to predict the target: `Engine_Condition`.",
        "",
        "## ✅ Best Model",
        f"- Algorithm: **{best_name}**",
        f"- Test F1-score: **{best_f1:.4f}**",
        f"- Test Accuracy: **{best_acc:.4f}**",
    ]
    if best_params is not None:
        readme_lines.append(f"- Best Parameters: `{best_params}`")
    readme_lines += [
        "",
        "## 📊 Features Used",
        "- Engine_RPM",
        "- Lub_Oil_Pressure",
        "- Fuel_Pressure",
        "- Coolant_Pressure",
        "- Lub_Oil_Temperature",
        "- Coolant_Temperature",
        "",
        "## 🔧 Training Setup",
        "- Models tried: DecisionTree, RandomForest, GradientBoosting",
        "- Cross-validation: 3-fold",
        "- Scoring metric: F1",
        "- Preprocessing: Median imputation + StandardScaler",
    ]

    readme_text = "\n".join(readme_lines)
    readme_path = os.path.join(ARTIFACT_DIR, "README.md")
    with open(readme_path, "w") as f:
        f.write(readme_text)

    api.upload_file(
        path_or_fileobj=readme_path,
        path_in_repo="README.md",
        repo_id=HF_MODEL_REPO,
        repo_type="model",
    )

    print("🎉 Model & README uploaded successfully!")
    print(f"🚀 View model: https://huggingface.co/{HF_MODEL_REPO}")


def main():
    df = load_and_normalize()
    X_train, X_test, y_train, y_test = clean_and_split(df)
    best_name, best_f1, best_metrics, best_model_path, exp_log_path = build_and_train(
        X_train, X_test, y_train, y_test
    )
    upload_best_model(best_name, best_f1, best_metrics, best_model_path, exp_log_path)
    print("✅ Full pipeline completed successfully.")


if __name__ == "__main__":
    main()

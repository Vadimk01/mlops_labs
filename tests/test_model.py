import json
import os

import pandas as pd


def test_data_schema_basic():
    data_path = os.getenv("DATA_PATH", "data/raw/creditcard.csv")
    assert os.path.exists(data_path), f"Data not found: {data_path}"

    df = pd.read_csv(data_path)

    required_cols = {"Class"}
    missing = required_cols - set(df.columns)
    assert not missing, f"Missing columns: {sorted(missing)}"

    assert df["Class"].notna().all(), "Class column contains missing values"
    assert set(df["Class"].unique()).issubset({0, 1}), "Class must contain only 0 and 1"
    assert df.shape[0] >= 50, "Too few rows for training experiment"


def test_no_critical_missing_values():
    data_path = os.getenv("DATA_PATH", "data/raw/creditcard.csv")
    assert os.path.exists(data_path), f"Data not found: {data_path}"

    df = pd.read_csv(data_path)

    critical_columns = ["Class"]
    for col in critical_columns:
        assert df[col].notna().all(), f"Critical column '{col}' contains missing values"


def test_artifacts_exist():
    model_path = os.getenv("MODEL_PATH", "models/model.pkl")
    metrics_path = os.getenv("METRICS_PATH", "models/metrics.json")
    cm_path = os.getenv("CM_PATH", "models/confusion_matrix.png")

    assert os.path.exists(model_path), f"{model_path} not found"
    assert os.path.exists(metrics_path), f"{metrics_path} not found"
    assert os.path.exists(cm_path), f"{cm_path} not found"


def test_metrics_file_structure():
    metrics_path = os.getenv("METRICS_PATH", "models/metrics.json")
    assert os.path.exists(metrics_path), f"{metrics_path} not found"

    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    required_keys = {"accuracy", "f1", "precision", "recall", "roc_auc"}
    missing = required_keys - set(metrics.keys())
    assert not missing, f"Missing metric keys: {sorted(missing)}"

    for key in required_keys:
        value = float(metrics[key])
        assert 0.0 <= value <= 1.0, f"Metric {key} must be between 0 and 1"


def test_quality_gate_f1():
    threshold = float(os.getenv("F1_THRESHOLD", "0.70"))
    metrics_path = os.getenv("METRICS_PATH", "models/metrics.json")

    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    f1 = float(metrics["f1"])
    assert f1 >= threshold, f"Quality Gate not passed: f1={f1:.4f} < {threshold:.2f}"

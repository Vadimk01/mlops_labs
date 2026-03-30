import os
import json
import pickle
import argparse
import warnings
import joblib

import mlflow
import mlflow.sklearn
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix
)
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")


def load_processed_data(file_path: str) -> pd.DataFrame:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Файл не знайдено: {file_path}")

    with open(file_path, "rb") as f:
        df = pickle.load(f)

    return df


def split_features_target(df: pd.DataFrame, target_column: str = "Class"):
    if target_column not in df.columns:
        raise ValueError(f"У датасеті немає цільової змінної '{target_column}'")

    X = df.drop(columns=[target_column])
    y = df[target_column]

    return X, y


def sample_data(
    df: pd.DataFrame,
    max_rows: int,
    target_column: str,
    random_state: int
) -> pd.DataFrame:
    if max_rows is None or len(df) <= max_rows:
        return df

    if target_column not in df.columns:
        raise ValueError(f"У датасеті немає цільової змінної '{target_column}'")

    class_counts = df[target_column].value_counts()

    if len(class_counts) < 2:
        return df.sample(n=max_rows, random_state=random_state).reset_index(drop=True)

    sampled_parts = []
    total_len = len(df)

    for class_value, class_count in class_counts.items():
        class_df = df[df[target_column] == class_value]
        n_class = max(1, round(max_rows * class_count / total_len))
        n_class = min(n_class, len(class_df))
        sampled_parts.append(class_df.sample(n=n_class, random_state=random_state))

    sampled_df = (
        pd.concat(sampled_parts)
        .sample(frac=1, random_state=random_state)
        .reset_index(drop=True)
    )

    if len(sampled_df) > max_rows:
        sampled_df = sampled_df.sample(n=max_rows, random_state=random_state).reset_index(drop=True)

    return sampled_df


def calculate_metrics(y_true, y_pred, y_proba):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_proba))
    }


def save_confusion_matrix(y_true, y_pred, output_path: str):
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def save_feature_importance(model, feature_names, output_path: str, top_n: int = 15):
    importances = pd.Series(model.feature_importances_, index=feature_names)
    importances = importances.sort_values(ascending=False).head(top_n)

    plt.figure(figsize=(10, 6))
    sns.barplot(x=importances.values, y=importances.index)
    plt.title(f"Top {top_n} Feature Importances")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def save_metrics(metrics: dict, output_path: str):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)


def parse_args():
    parser = argparse.ArgumentParser(description="Train RandomForest with MLflow")

    parser.add_argument(
        "data_path",
        type=str,
        help="Шлях до processed_data.pickle"
    )
    parser.add_argument(
        "models_dir",
        type=str,
        help="Папка для збереження моделі та артефактів"
    )

    parser.add_argument("--experiment_name", type=str, default="CreditCardFraudDetection")
    parser.add_argument("--run_name", type=str, default=None)

    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--max_depth", type=int, default=10)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--class_weight", type=str, default="balanced")
    parser.add_argument("--max_rows", type=int, default=None)

    parser.add_argument("--author", type=str, default="Vadym")
    parser.add_argument("--dataset_version", type=str, default="v1")
    parser.add_argument("--model_type", type=str, default="RandomForest")
    parser.add_argument("--target_column", type=str, default="Class")

    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.models_dir, exist_ok=True)

    df = load_processed_data(args.data_path)

    df = sample_data(
        df=df,
        max_rows=args.max_rows,
        target_column=args.target_column,
        random_state=args.random_state
    )

    X, y = split_features_target(df, target_column=args.target_column)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y
    )

    mlflow.set_experiment(args.experiment_name)

    with mlflow.start_run(run_name=args.run_name):
        mlflow.set_tag("author", args.author)
        mlflow.set_tag("dataset_version", args.dataset_version)
        mlflow.set_tag("model_type", args.model_type)

        mlflow.log_param("n_estimators", args.n_estimators)
        mlflow.log_param("max_depth", args.max_depth)
        mlflow.log_param("test_size", args.test_size)
        mlflow.log_param("random_state", args.random_state)
        mlflow.log_param("class_weight", args.class_weight)
        mlflow.log_param("max_rows", args.max_rows if args.max_rows is not None else "all")

        model = RandomForestClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            random_state=args.random_state,
            n_jobs=-1,
            class_weight=args.class_weight
        )

        model.fit(X_train, y_train)

        y_train_pred = model.predict(X_train)
        y_train_proba = model.predict_proba(X_train)[:, 1]

        y_test_pred = model.predict(X_test)
        y_test_proba = model.predict_proba(X_test)[:, 1]

        train_metrics = calculate_metrics(y_train, y_train_pred, y_train_proba)
        test_metrics = calculate_metrics(y_test, y_test_pred, y_test_proba)

        for metric_name, metric_value in train_metrics.items():
            mlflow.log_metric(f"train_{metric_name}", metric_value)

        for metric_name, metric_value in test_metrics.items():
            mlflow.log_metric(f"test_{metric_name}", metric_value)

        confusion_matrix_path = os.path.join(args.models_dir, "confusion_matrix.png")
        feature_importance_path = os.path.join(args.models_dir, "feature_importance.png")
        metrics_path = os.path.join(args.models_dir, "metrics.json")
        model_path = os.path.join(args.models_dir, "model.pkl")

        save_confusion_matrix(y_test, y_test_pred, confusion_matrix_path)
        save_feature_importance(model, X_train.columns, feature_importance_path, top_n=15)
        save_metrics(test_metrics, metrics_path)
        joblib.dump(model, model_path)

        mlflow.log_artifact(confusion_matrix_path)
        mlflow.log_artifact(feature_importance_path)
        mlflow.log_artifact(metrics_path)
        mlflow.log_artifact(model_path)
        mlflow.sklearn.log_model(model, "random_forest_model")

        print("Навчання завершено успішно.")
        print(f"Train accuracy: {train_metrics['accuracy']:.4f}")
        print(f"Train f1:       {train_metrics['f1']:.4f}")
        print(f"Test accuracy:  {test_metrics['accuracy']:.4f}")
        print(f"Test f1:        {test_metrics['f1']:.4f}")
        print(f"Test recall:    {test_metrics['recall']:.4f}")
        print(f"Test roc_auc:   {test_metrics['roc_auc']:.4f}")
        print(f"Модель та артефакти збережено у: {args.models_dir}")


if __name__ == "__main__":
    main()
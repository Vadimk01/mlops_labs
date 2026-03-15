import os
import argparse
import warnings

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

warnings.filterwarnings("ignore")


def load_prepared_data(data_dir: str):
    train_path = os.path.join(data_dir, "train.csv")
    test_path = os.path.join(data_dir, "test.csv")

    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Файл не знайдено: {train_path}")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Файл не знайдено: {test_path}")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    return train_df, test_df


def split_features_target(df: pd.DataFrame, target_column: str = "Class"):
    if target_column not in df.columns:
        raise ValueError(f"У датасеті немає цільової змінної '{target_column}'")

    X = df.drop(columns=[target_column])
    y = df[target_column]

    return X, y


def calculate_metrics(y_true, y_pred, y_proba):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba)
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


def parse_args():
    parser = argparse.ArgumentParser(description="Train RandomForest with MLflow")

    parser.add_argument("data_dir", type=str, help="Папка з prepared train/test файлами")
    parser.add_argument("models_dir", type=str, help="Папка для збереження моделі та артефактів")

    parser.add_argument("--experiment_name", type=str, default="CreditCardFraudDetection")
    parser.add_argument("--run_name", type=str, default=None)

    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--max_depth", type=int, default=10)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--class_weight", type=str, default="balanced")

    parser.add_argument("--author", type=str, default="Vadym")
    parser.add_argument("--dataset_version", type=str, default="v1")
    parser.add_argument("--model_type", type=str, default="RandomForest")
    parser.add_argument("--target_column", type=str, default="Class")

    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.models_dir, exist_ok=True)

    train_df, test_df = load_prepared_data(args.data_dir)

    X_train, y_train = split_features_target(train_df, target_column=args.target_column)
    X_test, y_test = split_features_target(test_df, target_column=args.target_column)

    mlflow.set_experiment(args.experiment_name)

    with mlflow.start_run(run_name=args.run_name):
        mlflow.set_tag("author", args.author)
        mlflow.set_tag("dataset_version", args.dataset_version)
        mlflow.set_tag("model_type", args.model_type)

        mlflow.log_param("n_estimators", args.n_estimators)
        mlflow.log_param("max_depth", args.max_depth)
        mlflow.log_param("random_state", args.random_state)
        mlflow.log_param("class_weight", args.class_weight)

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
        model_path = os.path.join(args.models_dir, "random_forest_model")

        save_confusion_matrix(y_test, y_test_pred, confusion_matrix_path)
        save_feature_importance(model, X_train.columns, feature_importance_path, top_n=15)

        mlflow.log_artifact(confusion_matrix_path)
        mlflow.log_artifact(feature_importance_path)
        mlflow.sklearn.log_model(model, "random_forest_model")
        mlflow.sklearn.save_model(model, model_path)

        print("Навчання завершено успішно.")
        print(f"Train accuracy: {train_metrics['accuracy']:.4f}")
        print(f"Train f1_score: {train_metrics['f1_score']:.4f}")
        print(f"Test accuracy:  {test_metrics['accuracy']:.4f}")
        print(f"Test f1_score:  {test_metrics['f1_score']:.4f}")
        print(f"Test recall:    {test_metrics['recall']:.4f}")
        print(f"Test roc_auc:   {test_metrics['roc_auc']:.4f}")
        print(f"Модель та артефакти збережено у: {args.models_dir}")


if __name__ == "__main__":
    main()
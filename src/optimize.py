import json
import os
import pickle
import warnings

import hydra
import joblib
import mlflow
import mlflow.sklearn
import optuna
import pandas as pd

from omegaconf import DictConfig, OmegaConf
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score
)
from sklearn.model_selection import train_test_split, cross_val_score

warnings.filterwarnings("ignore")


def load_processed_data(file_path: str) -> pd.DataFrame:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Файл не знайдено: {file_path}")

    with open(file_path, "rb") as f:
        df = pickle.load(f)

    return df


def split_features_target(df: pd.DataFrame, target_column: str):
    if target_column not in df.columns:
        raise ValueError(f"У датасеті немає цільової змінної '{target_column}'")

    X = df.drop(columns=[target_column])
    y = df[target_column]
    return X, y


def calculate_metrics(y_true, y_pred, y_proba=None):
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
    }

    if y_proba is not None:
        metrics["roc_auc"] = roc_auc_score(y_true, y_proba)

    return metrics


def build_model(cfg: DictConfig, trial: optuna.trial.Trial):
    model_type = cfg.model.type

    if model_type == "random_forest":
        params = {
            "n_estimators": trial.suggest_int(
                "n_estimators",
                cfg.model.search_space.n_estimators.low,
                cfg.model.search_space.n_estimators.high
            ),
            "max_depth": trial.suggest_int(
                "max_depth",
                cfg.model.search_space.max_depth.low,
                cfg.model.search_space.max_depth.high
            ),
            "min_samples_split": trial.suggest_int(
                "min_samples_split",
                cfg.model.search_space.min_samples_split.low,
                cfg.model.search_space.min_samples_split.high
            ),
            "min_samples_leaf": trial.suggest_int(
                "min_samples_leaf",
                cfg.model.search_space.min_samples_leaf.low,
                cfg.model.search_space.min_samples_leaf.high
            ),
            "random_state": cfg.seed,
            "class_weight": cfg.model.fixed_params.class_weight,
            "n_jobs": cfg.model.fixed_params.n_jobs,
        }

        model = RandomForestClassifier(**params)
        return model, params

    elif model_type == "logistic_regression":
        params = {
            "C": trial.suggest_float(
                "C",
                cfg.model.search_space.C.low,
                cfg.model.search_space.C.high,
                log=True
            ),
            "solver": trial.suggest_categorical(
                "solver",
                cfg.model.search_space.solver.values
            ),
            "penalty": trial.suggest_categorical(
                "penalty",
                cfg.model.search_space.penalty.values
            ),
            "class_weight": cfg.model.fixed_params.class_weight,
            "max_iter": cfg.model.fixed_params.max_iter,
            "random_state": cfg.seed,
        }

        model = LogisticRegression(**params)
        return model, params

    else:
        raise ValueError(f"Непідтримуваний тип моделі: {model_type}")


def get_sampler(cfg: DictConfig):
    sampler_name = cfg.hpo.sampler.lower()

    if sampler_name == "tpe":
        return optuna.samplers.TPESampler(seed=cfg.seed)
    elif sampler_name == "random":
        return optuna.samplers.RandomSampler(seed=cfg.seed)
    else:
        raise ValueError(f"Непідтримуваний sampler: {cfg.hpo.sampler}")


def save_best_params(best_params: dict, output_path: str):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(best_params, f, indent=4, ensure_ascii=False)


def save_experiment_config(cfg: DictConfig, output_path: str):
    config_text = OmegaConf.to_yaml(cfg)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(config_text)


def objective(trial, cfg, X_train, X_test, y_train, y_test):
    with mlflow.start_run(run_name=f"trial_{trial.number}", nested=True):
        model, params = build_model(cfg, trial)

        mlflow.log_params(params)
        mlflow.log_param("metric", cfg.hpo.metric)
        mlflow.log_param("use_cv", cfg.hpo.use_cv)

        mlflow.set_tag("trial_number", trial.number)
        mlflow.set_tag("sampler", cfg.hpo.sampler)
        mlflow.set_tag("model_type", cfg.model.type)
        mlflow.set_tag("seed", cfg.seed)

        if cfg.hpo.use_cv:
            cv_scores = cross_val_score(
                model,
                X_train,
                y_train,
                cv=cfg.hpo.cv_folds,
                scoring=cfg.hpo.metric,
                n_jobs=-1
            )

            objective_value = cv_scores.mean()

            mlflow.log_metric("cv_mean_score", objective_value)
            mlflow.log_metric("cv_std_score", cv_scores.std())

            model.fit(X_train, y_train)

            y_test_pred = model.predict(X_test)
            y_test_proba = (
                model.predict_proba(X_test)[:, 1]
                if hasattr(model, "predict_proba") else None
            )
            test_metrics = calculate_metrics(y_test, y_test_pred, y_test_proba)

            for metric_name, metric_value in test_metrics.items():
                mlflow.log_metric(f"test_{metric_name}", metric_value)

        else:
            model.fit(X_train, y_train)

            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            y_train_proba = (
                model.predict_proba(X_train)[:, 1]
                if hasattr(model, "predict_proba") else None
            )
            y_test_proba = (
                model.predict_proba(X_test)[:, 1]
                if hasattr(model, "predict_proba") else None
            )

            train_metrics = calculate_metrics(y_train, y_train_pred, y_train_proba)
            test_metrics = calculate_metrics(y_test, y_test_pred, y_test_proba)

            for metric_name, metric_value in train_metrics.items():
                mlflow.log_metric(f"train_{metric_name}", metric_value)

            for metric_name, metric_value in test_metrics.items():
                mlflow.log_metric(f"test_{metric_name}", metric_value)

            objective_value = test_metrics[cfg.hpo.metric]

        mlflow.log_metric("objective_value", objective_value)

        return objective_value


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig):
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)

    df = load_processed_data(cfg.data.processed_path)
    X, y = split_features_target(df, cfg.data.target_column)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=cfg.data.test_size,
        random_state=cfg.seed,
        stratify=y
    )

    os.makedirs("models", exist_ok=True)

    sampler = get_sampler(cfg)

    with mlflow.start_run(run_name="optuna_study"):
        mlflow.log_param("seed", cfg.seed)
        mlflow.log_param("model_type", cfg.model.type)
        mlflow.log_param("n_trials", cfg.hpo.n_trials)
        mlflow.log_param("sampler", cfg.hpo.sampler)
        mlflow.log_param("metric", cfg.hpo.metric)
        mlflow.log_param("direction", cfg.hpo.direction)
        mlflow.log_param("use_cv", cfg.hpo.use_cv)
        mlflow.log_param("cv_folds", cfg.hpo.cv_folds)
        mlflow.log_param("processed_path", cfg.data.processed_path)

        mlflow.set_tag("run_type", "parent_study")
        mlflow.set_tag("model_type", cfg.model.type)
        mlflow.set_tag("sampler", cfg.hpo.sampler)

        study = optuna.create_study(
            direction=cfg.hpo.direction,
            sampler=sampler
        )

        def objective_wrapper(trial):
            return objective(
                trial=trial,
                cfg=cfg,
                X_train=X_train,
                X_test=X_test,
                y_train=y_train,
                y_test=y_test
            )

        study.optimize(objective_wrapper, n_trials=cfg.hpo.n_trials)

        best_params = study.best_params
        best_value = study.best_value

        for param_name, param_value in best_params.items():
            mlflow.log_param(f"best_{param_name}", param_value)

        mlflow.log_metric("best_score", best_value)

        best_params_path = os.path.join("models", "best_params.json")
        save_best_params(best_params, best_params_path)
        mlflow.log_artifact(best_params_path)

        config_artifact_path = os.path.join("models", "experiment_config.txt")
        save_experiment_config(cfg, config_artifact_path)
        mlflow.log_artifact(config_artifact_path)

        best_model, _ = build_model(cfg, study.best_trial)
        best_model.fit(X_train, y_train)

        y_test_pred = best_model.predict(X_test)
        y_test_proba = (
            best_model.predict_proba(X_test)[:, 1]
            if hasattr(best_model, "predict_proba") else None
        )

        final_metrics = calculate_metrics(y_test, y_test_pred, y_test_proba)

        for metric_name, metric_value in final_metrics.items():
            mlflow.log_metric(f"final_{metric_name}", metric_value)

        best_model_path = os.path.join("models", "best_model.pkl")
        joblib.dump(best_model, best_model_path)

        mlflow.log_artifact(best_model_path)

        if cfg.mlflow.log_model:
            mlflow.sklearn.log_model(best_model, "best_model")

        print("Оптимізація завершена успішно.")
        print(f"Best trial score: {best_value:.4f}")
        print(f"Best params: {best_params}")
        print(f"Best model saved to: {best_model_path}")


if __name__ == "__main__":
    main()
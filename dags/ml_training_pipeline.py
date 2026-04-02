from datetime import datetime

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import BranchPythonOperator


def choose_registration():
    return "register_model"


with DAG(
    dag_id="ml_training_pipeline",
    start_date=datetime(2026, 4, 1),
    schedule_interval=None,
    catchup=False,
    tags=["mlops", "training", "lab5"],
) as dag:

    start = EmptyOperator(task_id="start")

    check_data = BashOperator(
        task_id="check_data",
        bash_command="test -f /opt/airflow/project/data/raw/creditcard.csv",
    )

    prepare_data = BashOperator(
        task_id="prepare_data",
        bash_command=(
            "python /opt/airflow/project/src/prepare.py "
            "/opt/airflow/project/data/raw/creditcard.csv "
            "/opt/airflow/project/data/processed/processed_data.pickle"
        ),
    )

    train_model = BashOperator(
        task_id="train_model",
        bash_command=(
            "python /opt/airflow/project/src/train.py "
            "/opt/airflow/project/data/processed/processed_data.pickle "
            "/opt/airflow/project/models --max_rows 10000"
        ),
    )

    evaluate_model = BashOperator(
        task_id="evaluate_model",
        bash_command=(
            'python -c "import json; '
            "f=open('/opt/airflow/project/models/metrics.json', 'r', encoding='utf-8'); "
            "metrics=json.load(f); "
            "print('F1:', metrics['f1']); "
            "assert float(metrics['f1']) >= 0.70, 'Quality Gate failed'\""
        ),
    )

    branching = BranchPythonOperator(
        task_id="branching",
        python_callable=choose_registration,
    )

    register_model = BashOperator(
        task_id="register_model",
        bash_command="echo Model is accepted and ready for registration",
    )

    finish = EmptyOperator(task_id="finish")

    start >> check_data >> prepare_data >> train_model >> evaluate_model >> branching
    branching >> register_model >> finish

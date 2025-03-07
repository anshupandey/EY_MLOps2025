from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime

def task1():
    print("Hello from EY team")

with DAG(dag_id='test133',schedule=None, start_date=datetime(2025,3,7),catchup=False) as dag:
    mytask = PythonOperator(task_id='task1',python_callable=task1)

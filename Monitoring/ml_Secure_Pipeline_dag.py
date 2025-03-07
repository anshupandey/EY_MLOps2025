from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib
import os
from cryptography.fernet import Fernet
import logging

# Ensure the necessary directories exist
os.makedirs('data', exist_ok=True)
os.makedirs('results', exist_ok=True)

# Step 1: Data Ingestion with Encryption
def data_ingestion():
    print("Starting data ingestion...")
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target

    # Encrypt sensitive data (e.g., target column)
    key = Fernet.generate_key()
    cipher_suite = Fernet(key)
    df['target'] = df['target'].apply(lambda x: cipher_suite.encrypt(str(x).encode()))
    df.to_csv('data/encrypted_breast_cancer.csv', index=False)
    print("Data ingestion completed. Encrypted data saved to 'data/encrypted_breast_cancer.csv'.")

# Step 2: Data Preprocessing
def data_preprocessing():
    print("Starting data preprocessing...")
    df = pd.read_csv('data/encrypted_breast_cancer.csv')

    # Decrypt the target column
    key = Fernet.generate_key()
    cipher_suite = Fernet(key)
    df['target'] = df['target'].apply(lambda x: int(cipher_suite.decrypt(eval(x)).decode()))

    X = df.drop('target', axis=1)
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Save preprocessed data
    pd.DataFrame(X_train).to_csv('data/X_train.csv', index=False)
    pd.DataFrame(X_test).to_csv('data/X_test.csv', index=False)
    pd.DataFrame(y_train, columns=['target']).to_csv('data/y_train.csv', index=False)
    pd.DataFrame(y_test, columns=['target']).to_csv('data/y_test.csv', index=False)

    # Save the scaler
    joblib.dump(scaler, 'results/scaler.pkl')
    print("Data preprocessing completed. Preprocessed data and scaler saved.")

# Step 3: Model Training with Encryption
def model_training():
    print("Starting model training...")
    X_train = pd.read_csv('data/X_train.csv')
    y_train = pd.read_csv('data/y_train.csv')

    # Convert y_train to a 1D array
    y_train = y_train.values.ravel()

    # Train the model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Encrypt and save the trained model
    key = Fernet.generate_key()
    cipher_suite = Fernet(key)
    joblib.dump(model, 'results/model.pkl')
    with open('results/model.pkl', 'rb') as f:
        model_data = f.read()
    encrypted_model = cipher_suite.encrypt(model_data)
    with open('results/encrypted_model.pkl', 'wb') as f:
        f.write(encrypted_model)
    print("Model training completed. Encrypted model saved to 'results/encrypted_model.pkl'.")

# Step 4: Model Evaluation with Logging
def model_evaluation():
    print("Starting model evaluation...")
    X_test = pd.read_csv('data/X_test.csv')
    y_test = pd.read_csv('data/y_test.csv')

    # Convert y_test to a 1D array
    y_test = y_test.values.ravel()

    # Load the model
    key = Fernet.generate_key()
    cipher_suite = Fernet(key)
    with open('results/encrypted_model.pkl', 'rb') as f:
        encrypted_model = f.read()
    decrypted_model = cipher_suite.decrypt(encrypted_model)
    with open('results/decrypted_model.pkl', 'wb') as f:
        f.write(decrypted_model)
    model = joblib.load('results/decrypted_model.pkl')

    # Make predictions
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy}")

    # Log the accuracy
    logging.basicConfig(filename='pipeline_audit.log', level=logging.INFO)
    logging.info(f"Model Accuracy: {accuracy}")

    # Save the accuracy result
    with open('results/accuracy.txt', 'w') as f:
        f.write(f"Accuracy: {accuracy}")
    print("Model evaluation completed. Accuracy saved to 'results/accuracy.txt'.")

# Define the DAG
default_args = {
    'owner': 'airflow',
    'start_date': datetime(2023, 10, 1),
    'retries': 1,
}

dag = DAG(
    'ml_pipeline_dag',
    default_args=default_args,
    description='End-to-End ML Pipeline with Security and Compliance',
    schedule_interval='@daily',  # Run the DAG daily
    catchup=False,  # Disable catchup to avoid backfilling
)

# Define tasks
ingestion_task = PythonOperator(
    task_id='data_ingestion',
    python_callable=data_ingestion,
    dag=dag,
)

preprocessing_task = PythonOperator(
    task_id='data_preprocessing',
    python_callable=data_preprocessing,
    dag=dag,
)

training_task = PythonOperator(
    task_id='model_training',
    python_callable=model_training,
    dag=dag,
)

evaluation_task = PythonOperator(
    task_id='model_evaluation',
    python_callable=model_evaluation,
    dag=dag,
)

# Define task dependencies
ingestion_task >> preprocessing_task >> training_task >> evaluation_task

from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib
import os

# Ensure the necessary directories exist
os.makedirs('data', exist_ok=True)
os.makedirs('results', exist_ok=True)

# Step 1: Data Ingestion
def data_ingestion():
    data = load_iris()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target
    df.to_csv('data/iris.csv', index=False) # Ensure the dataset is saved as 'data/iris.csv'
    print("Data ingestion completed. Dataset loaded.")

# Step 2: Data Preprocessing
def data_preprocessing():
    print("Starting data preprocessing...")
    df = pd.read_csv('data/iris.csv')
    X = df.drop('target', axis=1)  # Features
    y = df['target']  # Target

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale the features
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

# Step 3: Model Training
def model_training():
    print("Starting model training...")
    X_train = pd.read_csv('data/X_train.csv')
    y_train = pd.read_csv('data/y_train.csv')

    # Convert y_train to a 1D array
    y_train = y_train.values.ravel()

    # Train the model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Save the trained model
    joblib.dump(model, 'results/model.pkl')
    print("Model training completed. Model saved to 'results/model.pkl'.")

# Step 4: Model Evaluation
def model_evaluation():
    print("Starting model evaluation...")
    X_test = pd.read_csv('data/X_test.csv')
    y_test = pd.read_csv('data/y_test.csv')

    # Convert y_test to a 1D array
    y_test = y_test.values.ravel()

    # Load the model
    model = joblib.load('results/model.pkl')

    # Make predictions
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy}")

    # Save the accuracy result
    with open('results/accuracy.txt', 'w') as f:
        f.write(f"Accuracy: {accuracy}")
    print("Model evaluation completed. Accuracy saved to 'results/accuracy.txt'.")

# Step 5: Model Serving (Flask API)
def model_serving():
    print("Starting model serving...")
    # Load the trained model
    model = joblib.load('results/model.pkl')

    # Load the scaler
    scaler = joblib.load('results/scaler.pkl')

    # Create a Flask app
    from flask import Flask, request, jsonify
    app = Flask(__name__)

    @app.route('/predict', methods=['POST'])
    def predict():
        # Get input data from the request
        data = request.get_json()
        
        # Convert input data to a DataFrame with the correct feature names
        input_data = pd.DataFrame([data], columns=[
            "sepal length (cm)",
            "sepal width (cm)",
            "petal length (cm)",
            "petal width (cm)"
        ])
        
        # Scale the input data
        input_data_scaled = scaler.transform(input_data)
        
        # Make a prediction
        prediction = model.predict(input_data_scaled)
        return jsonify({'prediction': int(prediction[0])})

    # Run the Flask app
    print("Model serving started. API is running on http://0.0.0.0:5000/predict")
    app.run(host='0.0.0.0', port=5000)

# Define the DAG
default_args = {
    'owner': 'airflow',
    'start_date': datetime(2023, 10, 1),
    'retries': 1,
}

dag = DAG(
    'ml_pipeline_dag',
    default_args=default_args,
    description='End-to-End ML Pipeline with Airflow',
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

serving_task = PythonOperator(
    task_id='model_serving',
    python_callable=model_serving,
    dag=dag,
)

# Define task dependencies
ingestion_task >> preprocessing_task >> training_task >> evaluation_task >> serving_task
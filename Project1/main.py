import pandas as pd
import mlflow, mlflow.sklearn
import argparse
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import tree    
from mlflow.models import infer_signature

mlflow.set_tracking_uri(uri="http://4.188.75.152:5000")
mlflow.set_experiment("Banking_Churn")

def dataloader():
    url = "https://raw.githubusercontent.com/anshupandey/Machine_Learning_Training/refs/heads/master/datasets/Bank_churn_modelling.csv"
    df = pd.read_csv(url)
    x = df[['CreditScore', 'Geography','Gender', 'Age','Balance', 'NumOfProducts', 'IsActiveMember']]
    y = df['Exited']

    pipeline = ColumnTransformer([('encoder',OneHotEncoder(drop='first'),[1,2]),
                                ('scaler',StandardScaler(),[0,3,4,5])],remainder='passthrough')

    pipeline.fit(x)
    x2 = pd.DataFrame(pipeline.transform(x),columns=pipeline.get_feature_names_out())
    x_train, x_test, y_train, y_test = train_test_split(x2,y,test_size=0.2,random_state=1, stratify=y)

    return x_train,y_train,x_test,y_test

def main():
    mlflow.sklearn.autolog()

    parser = argparse.ArgumentParser()

    # add arguments
    parser.add_argument("--data",dest='training_data', type=str)
    parser.add_argument("--max_depth",dest='max_depth', type=int)

    args = parser.parse_args()
    df = pd.read_csv(args.training_data)
    max_depth = args.max_depth
    print("data loaded for  shape : ",df.shape)
    print("argement max_depth is ",max_depth)

    x_train,y_train,x_test,y_test = dataloader()
    if 'prediction' in x_test.columns:
        x_test.drop(columns=['prediction'],inplace=True)

    eval_data = x_test
    eval_data['label'] = y_test
    mlflow.enable_system_metrics_logging()
    with mlflow.start_run(run_name="Anshu_BCM_"+str(np.random.randint(1000)),log_system_metrics=True):
        # create a model object using class LogisticRegression
        model = tree.DecisionTreeClassifier(random_state=5, max_depth=max_depth, min_samples_leaf=20)

        # train the model using train set : x_train, and y_train
        model.fit(x_train,y_train)

        #log the model manually with signature
        signature = infer_signature(x_train,y_train)

        model_info = mlflow.sklearn.log_model(model,"churn_model",signature=signature)

        model_uri = model_info.model_uri
        evalution_config = {"log_model_explainability":True, "metric_prefix":"test_",}

        result = mlflow.evaluate(model_uri,eval_data,targets='label',
                                model_type='classifier',
                                evaluator_config=evalution_config)

    return result

if __name__=="__main__":
    main()

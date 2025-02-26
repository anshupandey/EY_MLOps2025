from flask import Flask, request
import pandas as pd
import mlflow, mlflow.sklearn
import json


app = Flask(__name__)

model_name="CPM001"
alias = 'stagging'

model = mlflow.pyfunc.load_model(f"models:/{model_name}@{alias}")


@app.route("/get_schema")
def func1():
    schema = {'encoder__Geography_Germany': "double (required)", 'encoder__Geography_Spain': "double (required)", 
              'encoder__Gender_Male': "double (required)", 'scaler__CreditScore': "double (required)", 
              'scaler__Age': "double (required)", 'scaler__Balance': "double (required)", 
              'scaler__NumOfProducts': "double (required)", 'remainder__IsActiveMember': "double (required)"}

    return json.dumps(schema)

@app.route("/predict", methods=['GET','POST'])
def func2():
    data = request.data
    data = data.decode()
    print(data, type(data))
    data = json.loads(data)
    df = pd.DataFrame(data)
    output = model.predict(df)
    data['prediction'] = str(output[0])
    return json.dumps(data)

if __name__=="__main__":
    app.run(debug=True,port=8000)
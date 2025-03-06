
import os
import joblib
import json 
import logging
import numpy
def init():
    global model
    path = os.path.join(os.getenv("AZUREML_MODEL_DIR"),"outputs/model.pkl") # AZUREML_MODEL_DIR = ./azureml-models/MODEL_NAME/VERSION
    #logging.info(path)
    model = joblib.load(path)
    logging.info("initialization completed")

def run(raw_data):
    logging.info("model: request received")
    data = json.loads(raw_data)["data"]
    data = numpy.array(data)
    result = model.predict(data)
    logging.info("request procsesed")
    return result.tolist()

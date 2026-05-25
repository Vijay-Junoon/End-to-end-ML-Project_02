import mlflow
from sklearn.metrics import accuracy_score
import pandas as pd
import pickle
import os
import yaml

os.environ["MLFLOW_TRACKING_URI"] = "https://dagshub.com/vijay262005/ML_pipeline_02.mlflow"
os.environ["MLFLOW_TRACKING_USERNAME"] = "vijay262005"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "9da24b8499c61a98888907caf7183592abc4c819"

mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])

params = yaml.safe_load(open("params.yaml"))["train"]

def evaluate(data_path,model_path):
  dataset = pd.read_csv(data_path)
  X = dataset.iloc[:,:-1]
  y = dataset.iloc[:,-1]

  model = pickle.load(open(model_path,'rb'))

  predictions = model.predict(X)

  accuracy = accuracy_score(y,predictions)
  mlflow.log_metric('accuracy',accuracy)

if __name__ == "__main__":
  evaluate(params["data"],params["model"])
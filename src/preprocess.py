import mlflow
import os
import pandas as pd
import numpy as np
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder


os.environ["MLFLOW_TRACKING_URI"] = "https://dagshub.com/vijay262005/ML_pipeline_02.mlflow"
os.environ["MLFLOW_TRACKING_USERNAAME"] = "vijay262005"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "9da24b8499c61a98888907caf7183592abc4c819"

params = yaml.safe_load(open("params.yaml"))["preprocess"]

def preprocess(input_path,output_path):

  dataset = pd.read_csv(input_path)
  ct = ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[0])],remainder="passthrough")
  dataset = pd.DataFrame(ct.fit_transform(dataset))
  os.makedirs(os.path.dirname(output_path),exist_ok = True)
  dataset.to_csv(output_path,header=False,index=None)
  print(f"Preprocessed data saved to {output_path}")


if __name__ == "__main__":
  preprocess(params["input"],params["output"])






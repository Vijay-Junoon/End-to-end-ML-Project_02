import mlflow
import os
from mlflow.models import infer_signature
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from sklearn.model_selection import train_test_split,GridSearchCV
import pandas as pd
import yaml
from urllib.parse import urlparse
import pickle

os.environ["MLFLOW_TRACKING_URI"] = "https://dagshub.com/vijay262005/ML_pipeline_02.mlflow"
os.environ["MLFLOW_TRACKING_USERNAME"] = "vijay262005"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "9da24b8499c61a98888907caf7183592abc4c819"

params = yaml.safe_load(open("params.yaml"))["train"]

def hyperparameterTuning(X_train,y_train,params):
    classifier = RandomForestClassifier()
    grid_search = GridSearchCV(estimator=classifier,param_grid = params,cv=3,verbose=2,n_jobs=-1)
    grid_search.fit(X_train,y_train)
    return grid_search


def train(data_path,model_path):
  
  dataset = pd.read_csv(data_path)
  X = dataset.iloc[:,:-1]
  y = dataset.iloc[:,-1]


  mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])

  with mlflow.start_run():
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2, random_state = 42)
    params = {
      'n_estimators': [100,200],
      'max_depth' : [5,10],
      'min_samples_split' : [2,4],
    }

    signature = infer_signature(X_train,y_train)

    grid_search = hyperparameterTuning(X_train,y_train,params)
    model = grid_search.best_estimator_
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test,y_pred)
    cm = confusion_matrix(y_test,y_pred)
    cr = classification_report(y_test,y_pred)

    mlflow.log_params(grid_search.best_params_)
    mlflow.log_metric("accuracy",accuracy)
    mlflow.log_text(str(cm),"ConfutionMatrix.txt")
    mlflow.log_text(cr,"ClassificationReport.txt")


    url = urlparse(mlflow.get_tracking_uri()).scheme
    if url != "file":
       mlflow.sklearn.log_model(model,'model',registered_model_name='predictor')
    else:
      mlflow.sklearn.log_model(model,'model',signature = signature)

    os.makedirs(os.path.dirname(model_path),exist_ok = True)
    pickle.dump(model,open(model_path,'wb'))

    print(f"Model has been saved in {model_path}")


if __name__ == "__main__":
   train(params["data"],params["model"])
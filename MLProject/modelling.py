from pathlib import Path
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
base = Path(__file__).parent
train = pd.read_csv(base/"breast_cancer_preprocessing"/"train.csv")
test = pd.read_csv(base/"breast_cancer_preprocessing"/"test.csv")
Xtr,ytr=train.drop(columns="diagnosis"),train.diagnosis
Xte,yte=test.drop(columns="diagnosis"),test.diagnosis
mlflow.set_experiment("CI Breast Cancer Reza Harahap")
with mlflow.start_run():
    model=RandomForestClassifier(n_estimators=150,max_depth=8,random_state=42).fit(Xtr,ytr)
    pred=model.predict(Xte)
    mlflow.log_metrics({"accuracy":accuracy_score(yte,pred),"f1":f1_score(yte,pred)})
    mlflow.log_params(model.get_params())
    mlflow.sklearn.log_model(model,"model",input_example=Xte.head(3))

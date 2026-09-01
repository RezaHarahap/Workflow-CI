from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "breast_cancer_preprocessing"

train = pd.read_csv(DATA_DIR / "train.csv")
test = pd.read_csv(DATA_DIR / "test.csv")
X_train, y_train = train.drop(columns="diagnosis"), train["diagnosis"]
X_test, y_test = test.drop(columns="diagnosis"), test["diagnosis"]

# MLflow Projects already creates and activates a run through MLFLOW_RUN_ID.
model = RandomForestClassifier(
    n_estimators=150,
    max_depth=8,
    random_state=42,
)
model.fit(X_train, y_train)
predictions = model.predict(X_test)

mlflow.log_metrics({
    "accuracy": accuracy_score(y_test, predictions),
    "f1": f1_score(y_test, predictions),
})
mlflow.log_params(model.get_params())
mlflow.sklearn.log_model(
    model,
    artifact_path="model",
    input_example=X_test.head(3),
)
print("Training and MLflow logging completed successfully.")

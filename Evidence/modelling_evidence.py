from pathlib import Path
import json
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "MLProject" / "breast_cancer_preprocessing" / "breast_cancer_clean.csv"
TRACKING = ROOT / "evidence_mlruns"
mlflow.set_tracking_uri(TRACKING.as_uri())
mlflow.set_experiment("Breast Cancer Classification - Reza Harahap")

df = pd.read_csv(DATA)
X = df.drop(columns=["target"])
y = df["target"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

mlflow.sklearn.autolog(log_model_signatures=True, log_input_examples=True)
with mlflow.start_run(run_name="baseline_autolog_reza_harahap") as run:
    model = RandomForestClassifier(n_estimators=120, random_state=42)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred),
        "recall": recall_score(y_test, pred),
        "f1_score": f1_score(y_test, pred),
    }
    run_id = run.info.run_id
    experiment_id = run.info.experiment_id

state = {
    "run_id": run_id,
    "experiment_id": experiment_id,
    "tracking_uri": TRACKING.as_uri(),
    "model_uri": f"runs:/{run_id}/model",
    "metrics": metrics,
    "payload": {
        "dataframe_split": {
            "columns": list(X_test.columns),
            "data": [X_test.iloc[0].astype(float).tolist()],
        }
    },
}
(ROOT / "evidence_state.json").write_text(json.dumps(state, indent=2))
print(json.dumps(state, indent=2))

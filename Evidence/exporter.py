import json
import time
from pathlib import Path
import requests
from prometheus_client import Counter, Gauge, Histogram, start_http_server

ROOT = Path(__file__).resolve().parents[1]
state = json.loads((ROOT / "evidence_state.json").read_text())
payload = state["payload"]

REQUESTS = Counter("model_requests_total", "Total prediction requests")
ERRORS = Counter("model_errors_total", "Total failed prediction requests")
LATENCY = Histogram("model_request_latency_seconds", "Prediction latency seconds")
POSITIVE = Counter("model_positive_predictions_total", "Positive predictions")
MODEL_UP = Gauge("model_up", "Whether the model endpoint is reachable")
LAST_PREDICTION = Gauge("model_last_prediction", "Latest numeric prediction")

start_http_server(8000)
while True:
    started = time.perf_counter()
    REQUESTS.inc()
    try:
        response = requests.post("http://127.0.0.1:5001/invocations", json=payload, timeout=5)
        response.raise_for_status()
        result = response.json()
        value = result.get("predictions", [0])[0]
        if isinstance(value, dict):
            value = next(iter(value.values()))
        value = float(value)
        LAST_PREDICTION.set(value)
        if value >= 0.5:
            POSITIVE.inc()
        MODEL_UP.set(1)
    except Exception:
        ERRORS.inc()
        MODEL_UP.set(0)
    LATENCY.observe(time.perf_counter() - started)
    time.sleep(0.35)

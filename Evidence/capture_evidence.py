import json
import time
from pathlib import Path
import requests
from PIL import Image
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "evidence-output"
state = json.loads((ROOT / "evidence_state.json").read_text())
model_out = OUT / "Membangun_model"
mon = OUT / "Monitoring dan Logging"
prom_out = mon / "4.bukti monitoring Prometheus"
graf_out = mon / "5.bukti monitoring Grafana"
for folder in (model_out, prom_out, graf_out):
    folder.mkdir(parents=True, exist_ok=True)

ds = requests.post("http://127.0.0.1:3000/api/datasources", auth=("admin","admin"), json={
    "name":"Prometheus","type":"prometheus","url":"http://127.0.0.1:9090",
    "access":"proxy","isDefault":True
})
ds.raise_for_status()
uid = ds.json().get("datasource", {}).get("uid") or requests.get(
    "http://127.0.0.1:3000/api/datasources/name/Prometheus", auth=("admin","admin")
).json()["uid"]
panels=[]
queries=[
    ("Total Prediction Requests","model_requests_total","stat"),
    ("Prediction Error Rate","rate(model_errors_total[1m])","timeseries"),
    ("Average Request Latency","rate(model_request_latency_seconds_sum[1m]) / rate(model_request_latency_seconds_count[1m])","timeseries"),
    ("Model Availability","model_up","stat"),
    ("Positive Predictions","model_positive_predictions_total","stat"),
]
for i,(title,expr,kind) in enumerate(queries):
    panels.append({
        "id":i+1,"title":title,"type":kind,
        "gridPos":{"h":8,"w":8,"x":(i%3)*8,"y":(i//3)*8},
        "datasource":{"type":"prometheus","uid":uid},
        "targets":[{"refId":"A","expr":expr,"datasource":{"type":"prometheus","uid":uid}}],
        "fieldConfig":{"defaults":{},"overrides":[]},
        "options":{"reduceOptions":{"values":False,"calcs":["lastNotNull"],"fields":""},"orientation":"auto"}
    })
dashboard={"dashboard":{
    "id":None,"uid":"reza-harahap-msml","title":"reza_harahap","tags":["dicoding","mlops"],
    "timezone":"browser","schemaVersion":39,"version":0,"refresh":"5s","panels":panels,
    "time":{"from":"now-15m","to":"now"}
},"overwrite":True}
r=requests.post("http://127.0.0.1:3000/api/dashboards/db",auth=("admin","admin"),json=dashboard)
r.raise_for_status()

prediction=requests.post("http://127.0.0.1:5001/invocations",json=state["payload"])
serving_html=f"""<!doctype html><html><head><meta charset="utf-8"><style>
body{{font-family:Arial;background:#0f172a;color:#e2e8f0;padding:48px}} .card{{background:#1e293b;padding:28px;border-radius:14px}}
.ok{{color:#4ade80;font-size:28px}} pre{{background:#020617;padding:18px;border-radius:8px;white-space:pre-wrap}}
</style></head><body><div class="card"><h1>MLflow Model Serving — Reza Harahap</h1>
<p class="ok">● Endpoint aktif — HTTP {prediction.status_code}</p>
<p><b>Endpoint:</b> http://127.0.0.1:5001/invocations</p>
<p><b>Model URI:</b> {state["model_uri"]}</p><h2>Prediction response</h2>
<pre>{json.dumps(prediction.json(),indent=2)}</pre></div></body></html>"""
(ROOT/"serving-report.html").write_text(serving_html)

options=Options()
options.add_argument("--headless=new")
options.add_argument("--no-sandbox")
options.add_argument("--disable-dev-shm-usage")
options.add_argument("--window-size=1600,1000")
driver=webdriver.Chrome(options=options)

def snap(url,path,wait=7):
    driver.get(url); time.sleep(wait)
    png=path.with_suffix(".png")
    driver.save_screenshot(str(png))
    Image.open(png).convert("RGB").save(path,quality=92)
    png.unlink()

snap(f"http://127.0.0.1:5000/#/experiments/{state['experiment_id']}",model_out/"screenshoot_dashboard.jpg",10)
run_url=f"http://127.0.0.1:5000/#/experiments/{state['experiment_id']}/runs/{state['run_id']}"
driver.get(run_url)
artifact_tab=WebDriverWait(driver,20).until(
    EC.element_to_be_clickable((By.XPATH,"//*[normalize-space()='Artifacts']"))
)
driver.execute_script("arguments[0].click();",artifact_tab)
time.sleep(8)
assert "Artifacts" in driver.page_source and ("model" in driver.page_source or "MLmodel" in driver.page_source)
artifact_path=model_out/"screenshoot_artifak.jpg"
artifact_png=artifact_path.with_suffix(".png")
driver.save_screenshot(str(artifact_png))
Image.open(artifact_png).convert("RGB").save(artifact_path,quality=92)
artifact_png.unlink()
snap("http://127.0.0.1:8800/serving-report.html",mon/"1.bukti_serving.jpg",2)
prom_queries=[
    ("model_requests_total","1.monitoring_requests.jpg"),
    ("model_up","2.monitoring_model_up.jpg"),
    ("model_request_latency_seconds_count","3.monitoring_latency.jpg"),
]
for expr,name in prom_queries:
    snap("http://127.0.0.1:9090/graph?g0.expr="+expr+"&g0.tab=1",prom_out/name,5)
graf_url="http://127.0.0.1:3000/d/reza-harahap-msml/reza-harahap?orgId=1&from=now-15m&to=now"
snap(graf_url,graf_out/"1.dashboard_reza_harahap.jpg",12)
for i,name in enumerate(["2.monitoring_requests.jpg","3.monitoring_latency.jpg"],start=1):
    snap(graf_url,graf_out/name,6)
driver.quit()
(OUT/"evidence_state.json").write_text(json.dumps(state,indent=2))

"""
Shared model loader for Vercel serverless functions.
Underscore prefix ensures Vercel does NOT expose this as an endpoint.
"""

import os
import pickle

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "modelo_fake_news.pkl")
METRICS_PATH = os.path.join(BASE_DIR, "metricas.pkl")

pipeline_global = None
metricas_global = {"accuracy": 0, "report": {}}

try:
    with open(MODEL_PATH, "rb") as f:
        pipeline_global = pickle.load(f)
    with open(METRICS_PATH, "rb") as f:
        metricas_global = pickle.load(f)
except Exception as e:
    metricas_global["error"] = str(e)

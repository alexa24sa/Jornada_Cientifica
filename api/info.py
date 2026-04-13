"""Vercel serverless function: GET /api/info"""

from flask import Flask, jsonify
from api._model import metricas_global

app = Flask(__name__)


@app.route("/api/info")
def info():
    return jsonify({
        "accuracy": round(metricas_global.get("accuracy", 0) * 100, 2),
        "modelo": "TF-IDF + Logistic Regression",
        "registros": 29531,
    })

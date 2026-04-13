"""
Serverless Flask app for Vercel — Fake News Detection.
Loads pre-trained model from .pkl files and serves predictions.
"""

import os
import re
import pickle

from flask import Flask, render_template, request, jsonify
import requests
from bs4 import BeautifulSoup

# ---------------------------------------------------------------------------
# Paths (relative to project root, one level up from api/)
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "modelo_fake_news.pkl")
METRICS_PATH = os.path.join(BASE_DIR, "metricas.pkl")
TEMPLATE_DIR = os.path.join(BASE_DIR, "templates")

# ---------------------------------------------------------------------------
# Load model once at cold-start
# ---------------------------------------------------------------------------

def _load_model():
    with open(MODEL_PATH, "rb") as f:
        pipeline = pickle.load(f)
    with open(METRICS_PATH, "rb") as f:
        metricas = pickle.load(f)
    return pipeline, metricas

pipeline_global, metricas_global = _load_model()

# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}


def limpiar_texto(texto):
    texto = str(texto).lower()
    texto = re.sub(r"http\S+|www\S+", "", texto)
    texto = re.sub(r"[^a-záéíóúñü\s]", " ", texto)
    texto = re.sub(r"\s+", " ", texto).strip()
    return texto


def extraer_texto_url(url):
    try:
        from newspaper import Article
        articulo = Article(url)
        articulo.download()
        articulo.parse()
        texto = (articulo.title or "") + " " + (articulo.text or "")
        if len(texto.strip()) > 50:
            return texto.strip(), None
    except Exception:
        pass

    try:
        resp = requests.get(url, headers=HEADERS, timeout=15, verify=True)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()

        article = soup.find("article")
        if article:
            texto = article.get_text(separator=" ", strip=True)
        else:
            paragraphs = soup.find_all("p")
            texto = " ".join(p.get_text(strip=True) for p in paragraphs)

        titulo = ""
        h1 = soup.find("h1")
        if h1:
            titulo = h1.get_text(strip=True) + " "

        texto_final = titulo + texto
        if len(texto_final.strip()) < 30:
            return None, "No se pudo extraer suficiente texto de la URL."
        return texto_final.strip(), None
    except requests.exceptions.RequestException as e:
        return None, f"Error al acceder a la URL: {e}"


def predecir(pipeline, texto):
    texto_limpio = limpiar_texto(texto)
    proba = pipeline.predict_proba([texto_limpio])[0]
    pred = pipeline.predict([texto_limpio])[0]
    return {
        "prediccion": "REAL" if pred == 1 else "FAKE",
        "confianza": round(float(max(proba)) * 100, 2),
        "prob_fake": round(float(proba[0]) * 100, 2),
        "prob_real": round(float(proba[1]) * 100, 2),
        "texto_extraido": texto[:500] + ("…" if len(texto) > 500 else ""),
    }


# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------

app = Flask(__name__, template_folder=TEMPLATE_DIR)


@app.route("/")
def index():
    return render_template("index.html", metricas=metricas_global)


@app.route("/analizar", methods=["POST"])
def analizar():
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "No se recibieron datos JSON."}), 400

    url = data.get("url", "").strip()
    texto_manual = data.get("texto", "").strip()

    if not url and not texto_manual:
        return jsonify({"error": "Proporciona una URL o un texto para analizar."}), 400

    if url:
        if not re.match(r"^https?://", url):
            return jsonify({"error": "La URL debe comenzar con http:// o https://"}), 400
        texto, error = extraer_texto_url(url)
        if error:
            return jsonify({"error": error}), 400
    else:
        texto = texto_manual

    resultado = predecir(pipeline_global, texto)
    resultado["accuracy_modelo"] = round(metricas_global["accuracy"] * 100, 2)
    return jsonify(resultado)

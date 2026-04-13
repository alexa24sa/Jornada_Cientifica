"""
Detección de Fake News - Aplicación Web
========================================
Entrena un modelo con dos datasets de Kaggle y expone una interfaz web
donde el usuario ingresa una URL; la app hace web scraping del texto
y predice si es FAKE o REAL con su porcentaje de confianza.
"""

import os
import re
import pickle
import warnings

import numpy as np
from flask import Flask, render_template, request, jsonify

import requests
from bs4 import BeautifulSoup

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# 1. DESCARGA Y PREPARACIÓN DE DATOS
# ---------------------------------------------------------------------------

MODEL_PATH = os.path.join(os.path.dirname(__file__), "modelo_fake_news.pkl")
METRICS_PATH = os.path.join(os.path.dirname(__file__), "metricas.pkl")


def descargar_datasets():
    """Descarga ambos datasets de Kaggle y devuelve las rutas."""
    import kagglehub
    path1 = kagglehub.dataset_download("algord/fake-news")
    path2 = kagglehub.dataset_download("hassanamin/textdb3")
    return path1, path2


def preparar_datos(path1, path2):
    """
    Combina ambos datasets en un solo DataFrame con columnas:
      - text  : título + texto del artículo (cuando esté disponible)
      - label : 0 = FAKE, 1 = REAL
    """
    import pandas as pd

    # --- Dataset 1: FakeNewsNet (tiene título pero NO tiene cuerpo de texto) ---
    df1 = pd.read_csv(os.path.join(path1, "FakeNewsNet.csv"))
    df1 = df1[["title", "real"]].dropna(subset=["title"])
    df1 = df1.rename(columns={"title": "text", "real": "label"})
    df1["label"] = df1["label"].astype(int)  # 1=real, 0=fake

    # --- Dataset 2: fake_or_real_news (tiene título + texto completo) ---
    df2 = pd.read_csv(os.path.join(path2, "fake_or_real_news.csv"))
    df2["text"] = df2["title"].fillna("") + " " + df2["text"].fillna("")
    df2["label"] = df2["label"].map({"REAL": 1, "FAKE": 0})
    df2 = df2[["text", "label"]].dropna()

    # --- Combinar ---
    df = pd.concat([df1, df2], ignore_index=True)
    df = df.dropna(subset=["text", "label"])
    df["label"] = df["label"].astype(int)

    print(f"[INFO] Dataset combinado: {len(df)} registros")
    print(f"       REAL: {(df['label']==1).sum()}  |  FAKE: {(df['label']==0).sum()}")
    return df


# ---------------------------------------------------------------------------
# 2. ENTRENAMIENTO DEL MODELO
# ---------------------------------------------------------------------------

def limpiar_texto(texto):
    """Limpieza básica de texto."""
    texto = str(texto).lower()
    texto = re.sub(r"http\S+|www\S+", "", texto)       # quitar URLs
    texto = re.sub(r"[^a-záéíóúñü\s]", " ", texto)     # solo letras
    texto = re.sub(r"\s+", " ", texto).strip()
    return texto


def entrenar_modelo():
    """Entrena el pipeline TF-IDF + Logistic Regression y guarda en disco."""
    import pandas as pd
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, classification_report
    from sklearn.pipeline import Pipeline

    path1, path2 = descargar_datasets()
    df = preparar_datos(path1, path2)

    df["text"] = df["text"].apply(limpiar_texto)

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
    )

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=50_000, ngram_range=(1, 2), sublinear_tf=True)),
        ("clf", LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs", n_jobs=-1)),
    ])

    print("[INFO] Entrenando modelo …")
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=["FAKE", "REAL"], output_dict=True)

    print(f"[INFO] Accuracy en test: {acc*100:.2f}%")
    print(classification_report(y_test, y_pred, target_names=["FAKE", "REAL"]))

    # Guardar modelo y métricas
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(pipeline, f)
    with open(METRICS_PATH, "wb") as f:
        pickle.dump({"accuracy": acc, "report": report}, f)

    print(f"[INFO] Modelo guardado en {MODEL_PATH}")
    return pipeline, {"accuracy": acc, "report": report}


def cargar_modelo():
    """Carga el modelo ya entrenado. Si no existe, lo entrena primero."""
    if os.path.exists(MODEL_PATH) and os.path.exists(METRICS_PATH):
        with open(MODEL_PATH, "rb") as f:
            pipeline = pickle.load(f)
        with open(METRICS_PATH, "rb") as f:
            metricas = pickle.load(f)
        print(f"[INFO] Modelo cargado. Accuracy: {metricas['accuracy']*100:.2f}%")
        return pipeline, metricas
    else:
        return entrenar_modelo()


# ---------------------------------------------------------------------------
# 3. WEB SCRAPING
# ---------------------------------------------------------------------------

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}


def extraer_texto_url(url):
    """Extrae el texto principal de una URL mediante web scraping."""
    try:
        from newspaper import Article
        articulo = Article(url)
        articulo.download()
        articulo.parse()
        texto = articulo.title or ""
        texto += " " + (articulo.text or "")
        if len(texto.strip()) > 50:
            return texto.strip(), None
    except Exception:
        pass

    # Fallback con BeautifulSoup
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15, verify=True)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        # Eliminar scripts y estilos
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()

        # Intentar extraer el artículo de tags comunes
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


# ---------------------------------------------------------------------------
# 4. PREDICCIÓN
# ---------------------------------------------------------------------------

def predecir(pipeline, texto):
    """
    Retorna dict con la predicción y probabilidades.
    """
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
# 5. APLICACIÓN WEB (Flask)
# ---------------------------------------------------------------------------

app = Flask(__name__, template_folder=os.path.join(os.path.dirname(__file__), "templates"))

pipeline_global = None
metricas_global = None


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

    # Si se proporcionó URL, hacer scraping
    if url:
        # Validación básica de URL
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


@app.route("/reentrenar", methods=["POST"])
def reentrenar():
    global pipeline_global, metricas_global
    pipeline_global, metricas_global = entrenar_modelo()
    return jsonify({
        "mensaje": "Modelo reentrenado exitosamente.",
        "accuracy": round(metricas_global["accuracy"] * 100, 2),
    })


# ---------------------------------------------------------------------------
# 6. PUNTO DE ENTRADA
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("  DETECCIÓN DE FAKE NEWS - Cargando modelo …")
    print("=" * 60)
    pipeline_global, metricas_global = cargar_modelo()
    print(f"  Accuracy del modelo: {metricas_global['accuracy']*100:.2f}%")
    print("  Iniciando servidor web en http://127.0.0.1:5000")
    print("=" * 60)
    app.run(debug=False, host="127.0.0.1", port=5000)
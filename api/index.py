"""
Serverless entry point for Vercel.
Imports the Flask app from the main module.
"""

import sys
import os

# Add project root to path so we can import the main module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from deteccion_de_fake_news import (
    app, cargar_modelo, pipeline_global, metricas_global
)

# On Vercel the model is loaded at import time via the main module's globals.
# If they are None (serverless cold start), load them here.
import deteccion_de_fake_news as _mod
if _mod.pipeline_global is None:
    _mod.pipeline_global, _mod.metricas_global = cargar_modelo()

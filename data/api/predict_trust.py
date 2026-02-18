from flask import Flask, request, jsonify
import pandas as pd
import joblib
import os

# Crear Flask
app = Flask(__name__)

# Cargar modelo solo una vez por instancia
MODEL_PATH = os.path.join(os.path.dirname(__file__), "../model/horsetrust_model.pkl")
model = joblib.load(MODEL_PATH)

def get_label(score):
    if score < 0.70: return "riesgoso"
    elif score < 0.87: return "confiable"
    else: return "premium"

@app.route("/api/predict_trust", methods=["POST"])
def predict_trust():
    """
    Recibe JSON con datos del caballo y devuelve:
    - trust_score (1-100)
    - trust_label
    - confidence_% (placeholder 100)
    """
    data = request.get_json()
    if not data:
        return jsonify({"error": "No se recibió JSON válido"}), 400

    # Convertir a DataFrame
    df = pd.DataFrame([data])

    # Columnas que el modelo no usa
    drop_cols = ["horse_id","listing_id","seller_id",
                 "s_created_at","s_last_active_at","l_created_at"]
    df_model = df.drop(columns=drop_cols, errors="ignore")

    # Predecir score 0-1
    score = model.predict(df_model)[0]

    # Escalar a 1-100
    trust_score = max(1, min(round(score * 100, 2), 100))
    trust_label = get_label(score)
    confidence = 100  # placeholder

    response = {
        "trust_score": trust_score,
        "trust_label": trust_label,
        "confidence_%": confidence
    }

    return jsonify(response)

# Entrypoint para serverless (Vercel lo requiere)
def handler(request, context=None):
    return app(request.environ, context)

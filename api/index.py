from flask import Flask, request, jsonify
import pandas as pd
import joblib
import os

app = Flask(__name__)

# Cargar modelo
MODEL_PATH = os.path.join(os.path.dirname(__file__), "horsetrust_model.pkl")
model = joblib.load(MODEL_PATH)

def get_label(score):
    if score < 0.70: return "riesgoso"
    elif score < 0.87: return "confiable"
    else: return "premium"

# ✅ Ruta raíz para comprobar que la API funciona
@app.route("/", methods=["GET"])
def root():
    return jsonify({"status": "API funcionando ✅", "message": "Usa /predict_trust para predecir el trust score"}), 200

@app.route("/predict_trust", methods=["POST"])
def predict_trust():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No se recibió JSON válido"}), 400

        df = pd.DataFrame([data])
        drop_cols = ["horse_id","listing_id","seller_id",
                     "s_created_at","s_last_active_at","l_created_at"]
        df_model = df.drop(columns=drop_cols, errors="ignore")

        score = model.predict(df_model)[0]
        trust_score = max(1, min(round(score*100, 2), 100))
        trust_label = get_label(score)
        confidence = 100

        return jsonify({
            "trust_score": trust_score,
            "trust_label": trust_label,
            "confidence_%": confidence
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Exportar la app para Render / Vercel
handler = app

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

# 🔹 GET en /predict_trust para instrucciones
@app.route("/predict_trust", methods=["GET"])
def predict_trust_get():
    html_content = """
    <html>
    <head><title>Predict Trust API</title></head>
    <body>
        <h2>API /predict_trust</h2>
        <p>Usa el método <strong>POST</strong> para enviar datos y obtener el trust score.</p>
        <p>Ejemplo de JSON de prueba:</p>
        <pre>
{
  "horse_name": "Thunder",
  "birth_date": "2018-05-12",
  "h_sex": "M",
  "raza": "Arabian",
  "height_m": 1.65,
  "weight_kg": 520,
  "length_m": 2.3,
  "max_speed_kmh": 60,
  "h_temperament": "Calm",
  "h_category": "Racehorse",
  "h_career_races": 12,
  "h_days_since_last_race": 45,
  "h_linaje": "Purebred",
  "l_listing_status": "active",
  "l_asking_price_usd": 15000,
  "s_first_name": "Juan",
  "s_last_name": "Perez",
  "s_verified": true,
  "s_disputes": 0,
  "s_num_listings": 5,
  "s_flagged_fraud": false,
  "vet_total_exams": 3,
  "vet_major_issues": 0,
  "vet_avg_confidence": 0.95,
  "h_current_country": "USA",
  "h_birth_country": "USA",
  "completeness": 0.98,
  "vet_score": 92,
  "seller_score": 87
}
        </pre>
        <p>Envía este JSON como POST para obtener la predicción.</p>
    </body>
    </html>
    """
    return html_content, 200

# 🔹 POST en /predict_trust para predicción
@app.route("/predict_trust", methods=["POST"])
def predict_trust_post():
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

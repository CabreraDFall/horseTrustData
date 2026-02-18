import json
import pandas as pd
import joblib
import os

MODEL_PATH = os.path.join(os.path.dirname(__file__), "horsetrust_model.pkl")
model = joblib.load(MODEL_PATH)

def get_label(score):
    if score < 0.70: return "riesgoso"
    elif score < 0.87: return "confiable"
    else: return "premium"

def handler(request):
    try:
        data = request.get_json()
        if not data:
            return {"error": "No se recibió JSON válido"}, 400

        df = pd.DataFrame([data])
        drop_cols = ["horse_id","listing_id","seller_id",
                     "s_created_at","s_last_active_at","l_created_at"]
        df_model = df.drop(columns=drop_cols, errors="ignore")

        score = model.predict(df_model)[0]
        trust_score = max(1, min(round(score*100, 2), 100))
        trust_label = get_label(score)
        confidence = 100

        return {
            "trust_score": trust_score,
            "trust_label": trust_label,
            "confidence_%": confidence
        }
    except Exception as e:
        return {"error": str(e)}, 500

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingRegressor

# 1️⃣ Cargar CSV
df = pd.read_csv("data/horsetrust_database_index.csv", index_col=0)

# 2️⃣ Features y target
DROP = [
    "horse_id","listing_id","seller_id",
    "s_created_at","s_last_active_at","l_created_at"
]
X = df.drop(columns=DROP + ["horse_trust_score"], errors="ignore")
y = df["horse_trust_score"]

# 3️⃣ Columnas categóricas y numéricas
cat_cols = X.select_dtypes(include="object").columns.tolist()
num_cols = X.select_dtypes(exclude="object").columns.tolist()

preprocess = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
    ("num", "passthrough", num_cols)
])

# 4️⃣ Modelo de regresión
model = Pipeline([
    ("preprocess", preprocess),
    ("regressor", HistGradientBoostingRegressor(max_iter=300))
])

# 5️⃣ Split train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6️⃣ Entrenar
model.fit(X_train, y_train)
print("R2 score test:", model.score(X_test, y_test))

# 7️⃣ Guardar modelo
joblib.dump(model, "model/horsetrust_model.pkl")
print("Modelo guardado en 'model/horsetrust_model.pkl' ✅")

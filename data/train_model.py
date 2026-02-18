import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingRegressor

# Cargar CSV limpio
df = pd.read_csv("data/horsetrust_database.csv")

# Features y target
DROP = [
    "horse_id","listing_id","seller_id",
    "s_created_at","s_last_active_at","l_created_at"
]
X = df.drop(columns=DROP + ["horse_trust_score"], errors="ignore")
y = df["horse_trust_score"]  # valor 0-1

# Preprocesamiento
cat_cols = X.select_dtypes(include="object").columns
num_cols = X.select_dtypes(exclude="object").columns

preprocess = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ("num", "passthrough", num_cols)
])

# Modelo regresión
model = Pipeline([
    ("prep", preprocess),
    ("reg", HistGradientBoostingRegressor(max_iter=300))
])

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model.fit(X_train, y_train)
print("R2 test:", model.score(X_test, y_test))

# Guardar modelo
joblib.dump(model, "model/horsetrust_model.pkl")
print("Modelo guardado ✔")

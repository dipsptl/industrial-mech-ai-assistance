import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# ── Dataset ──
data = {
    "temperature": [55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 105, 110],
    "vibration":   [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 8.5, 9.0],
    "load":        [20, 25, 30, 35, 40, 50, 55, 60, 65, 70, 75, 80],
    "failure":     [0,  0,  0,  0,  0,  0,  1,  1,  1,  1,  1,  1]
}

df = pd.DataFrame(data)

X = df[["temperature", "vibration", "load"]]
y = df["failure"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

joblib.dump(model, "model.pkl")
print("✅ Model trained and saved as model.pkl!")
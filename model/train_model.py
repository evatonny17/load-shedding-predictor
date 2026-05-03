import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

df = pd.read_csv("../data/load_shedding_dataset.csv")

area_encoder = LabelEncoder()
day_encoder = LabelEncoder()

df["area"] = area_encoder.fit_transform(df["area"])
df["day"] = day_encoder.fit_transform(df["day"])

X = df[["area", "hour", "day", "temperature", "demand", "generation", "previous_outage"]]
y = df["load_shedding"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

pred = model.predict(X_test)
accuracy = accuracy_score(y_test, pred)

print("Model Accuracy:", accuracy)

joblib.dump(model, "../backend/model.pkl")
joblib.dump(area_encoder, "../backend/area_encoder.pkl")
joblib.dump(day_encoder, "../backend/day_encoder.pkl")

print("Model saved successfully!")
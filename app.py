from fastapi import FastAPI
import joblib

app = FastAPI()

# Carga tu modelo
model = joblib.load("model.pkl")

@app.get("/")
def read_root():
    return {"message": "Bienvenido a mi API de predicción"}

@app.post("/predict")
def predict(data: dict):
    # Suponiendo que los datos son un diccionario con las características
    X = [data['feature1'], data['feature2'], data['feature3']]  # Ajusta las características
    prediction = model.predict([X])
    return {"prediction": int(prediction[0])}
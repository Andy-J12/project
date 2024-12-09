import os
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import joblib
import uvicorn

# Crear la app FastAPI
app = FastAPI()

# Permitir el acceso desde cualquier origen
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permite todas las URLs
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cargar el modelo
model = joblib.load("model.pkl")

# Ruta para servir el archivo frontend.html
@app.get("/", response_class=HTMLResponse)
def read_root():
    with open("frontend.html", "r", encoding="utf-8") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)

# Ruta para las predicciones
@app.post("/predict")
def predict(data: dict):
    X = [data['age'], data['duration'], data['balance'], data['pdays']]
    prediction = model.predict([X])
    return {"prediction": int(prediction[0])}

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))  # Usa el puerto proporcionado por el entorno, por defecto 8000
    uvicorn.run(app, host="0.0.0.0", port=port)

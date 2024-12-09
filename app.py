import os
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import joblib
import uvicorn

# Crear la app FastAPI
app = FastAPI()

# Permitir el acceso desde cualquier origen (puedes cambiar esto si necesitas restringirlo)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Cambia esto por tus dominios si quieres restringir el acceso
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cargar el modelo
try:
    model = joblib.load("model.pkl")
except Exception as e:
    print(f"Error cargando el modelo: {e}")
    model = None

# Ruta para servir el archivo frontend.html
@app.get("/", response_class=HTMLResponse)
def read_root():
    try:
        with open("frontend.html", "r", encoding="utf-8") as f:
            html_content = f.read()
        return HTMLResponse(content=html_content)
    except Exception as e:
        return HTMLResponse(content=f"Error al cargar el archivo HTML: {e}", status_code=500)

# Ruta para las predicciones
@app.post("/predict")
def predict(data: dict):
    if model is None:
        return {"error": "Modelo no cargado correctamente"}
    
    try:
        X = [data['age'], data['duration'], data['balance'], data['pdays']]
        prediction = model.predict([X])
        return {"prediction": int(prediction[0])}
    except Exception as e:
        return {"error": f"Error en la predicción: {e}"}

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))  # Usa el puerto proporcionado por el entorno, por defecto 8000
    uvicorn.run(app, host="0.0.0.0", port=port)

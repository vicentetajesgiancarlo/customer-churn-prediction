from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import os

# Configuración de Rutas 
DIR_BASE = os.path.dirname(os.path.abspath(__file__))
CARPETA_MODELOS = os.path.join(DIR_BASE, "..", "models")

# Carga de Inteligencia 
try:
    modelo_entrenado = joblib.load(os.path.join(CARPETA_MODELOS, 'model_xgb.joblib'))
    escalador = joblib.load(os.path.join(CARPETA_MODELOS, 'scaler.joblib'))
    columnas_modelo = joblib.load(os.path.join(CARPETA_MODELOS, 'model_columns.joblib'))
except Exception as error_carga:
    print(f"Error al cargar los archivos del modelo: {error_carga}")

app = FastAPI(title="API de Predicción de Abandono")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def cargar_inicio():
    ruta_index = os.path.join(DIR_BASE, "index.html")
    if os.path.exists(ruta_index):
        return FileResponse(ruta_index)
    return {"error": "No se encontró index.html", "ruta": ruta_index}

# Montar archivos estáticos para CSS y JS
app.mount("/static", StaticFiles(directory=os.path.join(DIR_BASE, "static")), name="static")

class DatosCliente(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float

@app.post("/predict")
async def predecir_abandono(entrada: DatosCliente):
    try:
        # Convertimos la entrada a un DataFrame de Pandas
        df_cliente = pd.DataFrame([entrada.dict()])

        # 1. Mapeo manual para variables binarias 
        mapeo_binario = {'Yes': 1, 'No': 0, 'Male': 1, 'Female': 0}
        columnas_binarias = ['gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
        
        for col in columnas_binarias:
            df_cliente[col] = df_cliente[col].map(mapeo_binario)

        # 2. Creación de variables
        df_codificado = pd.get_dummies(df_cliente)

        # 3. Alineación con las columnas que el modelo espera
        df_final = pd.DataFrame(columns=columnas_modelo)
        
        for col in columnas_modelo:
            if col in df_codificado.columns:
                df_final[col] = df_codificado[col]
            elif any(col.startswith(orig + '_') for orig in df_cliente.columns):
                # Si es una categoría que no está presente en este registro
                df_final[col] = 0
            else:
                # Valores numéricos o directos
                df_final[col] = df_cliente[col] if col in df_cliente.columns else 0

        # Limpieza de valores nulos preventivo
        df_final = df_final.fillna(0)

        # 4. Escalado de variables numéricas
        columnas_a_escalar = ['tenure', 'MonthlyCharges', 'TotalCharges']
        df_final[columnas_a_escalar] = escalador.transform(df_final[columnas_a_escalar])

        # 5. Ejecución de la Predicción
        probabilidad = modelo_entrenado.predict_proba(df_final)[0][1]
        etiqueta = int(probabilidad > 0.5)

        return {
            "churn_probability": round(float(probabilidad), 4),
            "prediction": "Churn" if etiqueta == 1 else "No Churn",
            "risk_level": "High" if probabilidad > 0.7 else "Medium" if probabilidad > 0.4 else "Low"
        }

    except Exception as error_proceso:
        raise HTTPException(status_code=500, detail=str(error_proceso))

@app.get("/health")
async def estado_servidor():
    return {"status": "conectado"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8010)

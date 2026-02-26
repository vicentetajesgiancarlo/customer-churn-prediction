import pandas as pd
import joblib
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

# Configuración y Semilla
SEMILLA = 42

import os

# Determinar rutas relativas
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "WA_Fn-UseC_-Telco-Customer-Churn.csv")
MODELS_DIR = os.path.join(BASE_DIR, "..", "models")

def export_model():
    print("--- Cargando datos y replicando entrenamiento para exportación ---")
    datos = pd.read_csv(DATA_PATH)

    # Preprocesamiento básico (idéntico al notebook)
    datos.drop(columns=['customerID'], inplace=True)
    datos['TotalCharges'] = pd.to_numeric(datos['TotalCharges'], errors='coerce')
    datos['TotalCharges'].fillna(datos['TotalCharges'].median(), inplace=True)
    datos['Churn'] = datos['Churn'].map({'Yes': 1, 'No': 0})

    # Codificación de variables binarias
    cols_binarias = [c for c in datos.select_dtypes(include='object').columns 
                     if datos[c].nunique() == 2]
    le_dict = {}
    for col in cols_binarias:
        le = LabelEncoder()
        datos[col] = le.fit_transform(datos[col])
        le_dict[col] = le

    # One-hot encoding para el resto
    datos = pd.get_dummies(datos, drop_first=True)
    
    # Separar X e y
    X = datos.drop('Churn', axis=1)
    y = datos['Churn']
    
    # Lista de columnas finales para asegurar el orden en la API
    model_columns = list(X.columns)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEMILLA, stratify=y
    )

    # Escalamiento
    cols_escala = ['tenure', 'MonthlyCharges', 'TotalCharges']
    escalador = StandardScaler()
    X_train[cols_escala] = escalador.fit_transform(X_train[cols_escala])

    # Entrenamiento del mejor modelo XGBoost (hiperparámetros del notebook)
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    peso_pos = neg / pos

    print("Entrenando modelo final...")
    modelo_xgb = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.01,
        subsample=0.6,
        colsample_bytree=0.8,
        min_child_weight=3,
        gamma=0.1,
        scale_pos_weight=peso_pos,
        random_state=SEMILLA,
        use_label_encoder=False,
        eval_metric='auc'
    )
    modelo_xgb.fit(X_train, y_train)

    # Exportar objetos
    print(f"Guardando archivos en {MODELS_DIR}...")
    joblib.dump(modelo_xgb, os.path.join(MODELS_DIR, 'model_xgb.joblib'))
    joblib.dump(escalador, os.path.join(MODELS_DIR, 'scaler.joblib'))
    joblib.dump(model_columns, os.path.join(MODELS_DIR, 'model_columns.joblib'))
    # También guardamos el mapping de variables binarias si fuera necesario, 
    # pero para la API usaremos diccionarios directos para simplicidad.
    
    print("¡Exportación completada con éxito!")

if __name__ == "__main__":
    export_model()

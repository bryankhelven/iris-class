from fastapi import FastAPI
import joblib
import numpy as np
import pandas as pd  # Certifique-se de importar o pandas
from pydantic import BaseModel

# Inicializando a aplicação FastAPI
app = FastAPI(title="Iris Flower Classification API", version="1.0")

# Carregando o modelo treinado e o LabelEncoder
rf_model = joblib.load('models/random_forest_model.joblib')
label_encoder = joblib.load('models/label_encoder.joblib')

# Definindo os modelos de dados com Pydantic
class IrisFeatures(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

class Prediction(BaseModel):
    predicted_class: str
    predicted_class_index: int
    probabilities: dict

# Endpoint raiz
@app.get('/')
async def root():
    return {"message": "Bem-vindo à API de Classificação de Flores Íris! Utilize o endpoint /predict para fazer previsões."}

# Endpoint de previsão
@app.post('/predict', response_model=Prediction)
async def predict(iris: IrisFeatures):
    # Define os nomes das features (conforme usado no treinamento)
    feature_names = ['sepal length', 'sepal width', 'petal length', 'petal width']

    # Cria um DataFrame com os dados de entrada
    input_data = pd.DataFrame([[iris.sepal_length, iris.sepal_width, iris.petal_length, iris.petal_width]], columns=feature_names)

    # Faz a previsão
    prediction = rf_model.predict(input_data)
    prediction_proba = rf_model.predict_proba(input_data)

    # Obtém a classe prevista
    predicted_class_index = prediction[0]
    predicted_class = label_encoder.inverse_transform([predicted_class_index])[0]

    # Obtém as probabilidades
    probabilities = dict(zip(label_encoder.classes_, prediction_proba[0]))
    probabilities = {k: float(v) for k, v in probabilities.items()}  # Converter numpy.float para float

    # Cria a resposta
    response = {
        'predicted_class': predicted_class,
        'predicted_class_index': int(predicted_class_index),
        'probabilities': probabilities
    }

    return response

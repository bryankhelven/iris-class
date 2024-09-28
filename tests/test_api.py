from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_root():
    response = client.get('/')
    assert response.status_code == 200
    assert response.json() == {"message": "Bem-vindo à API de Classificação de Flores Íris! Utilize o endpoint /predict para fazer previsões."}

def test_predict_setosa():
    response = client.post('/predict', json={
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    })
    assert response.status_code == 200
    assert response.json()['predicted_class'] == 'Iris-setosa'

def test_predict_versicolor():
    response = client.post('/predict', json={
        "sepal_length": 6.0,
        "sepal_width": 2.2,
        "petal_length": 4.0,
        "petal_width": 1.0
    })
    assert response.status_code == 200
    assert response.json()['predicted_class'] == 'Iris-versicolor'

def test_predict_virginica():
    response = client.post('/predict', json={
        "sepal_length": 7.0,
        "sepal_width": 3.2,
        "petal_length": 6.0,
        "petal_width": 2.0
    })
    assert response.status_code == 200
    assert response.json()['predicted_class'] == 'Iris-virginica'

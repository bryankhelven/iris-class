import os
import joblib
import pandas as pd
#from data_preparation import load_and_explore_data
#from preprocessing import preprocess_data
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV, StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns

def train_and_evaluate_model():
    # Carregando e preparando os dados
    df = load_and_explore_data()
    X_train, X_test, y_train, y_test, label_encoder = preprocess_data(df)

    # Treinando o modelo inicial com parâmetros padrão
    rf_model = RandomForestClassifier(random_state=42)
    rf_model.fit(X_train, y_train)

    # Avaliação inicial do modelo nos dados de teste
    y_pred = rf_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Acurácia do modelo inicial nos dados de teste: {accuracy:.2f}")

    # Matriz de confusão inicial
    conf_matrix = confusion_matrix(y_test, y_pred)
    print("\nMatriz de Confusão - Modelo Inicial:")
    print(conf_matrix)

    # Relatório de classificação inicial
    class_report = classification_report(y_test, y_pred, target_names=label_encoder.classes_)
    print("\nRelatório de Classificação - Modelo Inicial:")
    print(class_report)

    # Plotando a matriz de confusão inicial
    plt.figure(figsize=(6,4))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=label_encoder.classes_,
                yticklabels=label_encoder.classes_)
    plt.xlabel('Previsão')
    plt.ylabel('Verdadeiro')
    plt.title('Matriz de Confusão - Modelo Inicial')
    plt.show()

    # Ajuste de hiperparâmetros com validação cruzada 10-fold no conjunto de treinamento
    param_grid = {
        'n_estimators': [50, 100, 150],
        'max_depth': [None, 2, 4, 6],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'bootstrap': [True, False]
    }

    rf_base = RandomForestClassifier(random_state=42)

    # Configurando a validação cruzada estratificada
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    grid_search = GridSearchCV(
        estimator=rf_base,
        param_grid=param_grid,
        cv=cv,  # Utilizando validação cruzada 10-fold
        scoring='accuracy',
        n_jobs=1  # Definido como 1 para evitar problemas no Windows
    )

    # Executando o Grid Search no conjunto de treinamento
    grid_search.fit(X_train, y_train)
    best_params = grid_search.best_params_
    print("\nMelhores hiperparâmetros encontrados:")
    print(best_params)

    # Treinando o modelo com os melhores hiperparâmetros no conjunto de treinamento completo
    rf_best = RandomForestClassifier(**best_params, random_state=42)
    rf_best.fit(X_train, y_train)

    # Avaliação do modelo otimizado nos dados de teste
    y_pred_best = rf_best.predict(X_test)
    accuracy_best = accuracy_score(y_test, y_pred_best)
    print(f"\nAcurácia do modelo otimizado nos dados de teste: {accuracy_best:.2f}")

    # Matriz de confusão otimizada
    conf_matrix_best = confusion_matrix(y_test, y_pred_best)
    print("\nMatriz de Confusão - Modelo Otimizado:")
    print(conf_matrix_best)

    # Relatório de classificação otimizado
    class_report_best = classification_report(y_test, y_pred_best, target_names=label_encoder.classes_)
    print("\nRelatório de Classificação - Modelo Otimizado:")
    print(class_report_best)

    # Plotando a matriz de confusão otimizada
    plt.figure(figsize=(6,4))
    sns.heatmap(conf_matrix_best, annot=True, fmt='d', cmap='Greens',
                xticklabels=label_encoder.classes_,
                yticklabels=label_encoder.classes_)
    plt.xlabel('Previsão')
    plt.ylabel('Verdadeiro')
    plt.title('Matriz de Confusão - Modelo Otimizado')
    plt.show()

    # Salvando o modelo otimizado e o LabelEncoder
    if not os.path.exists('../models'):
        os.makedirs('../models')

    joblib.dump(rf_best, '../models/random_forest_model.joblib')
    joblib.dump(label_encoder, '../models/label_encoder.joblib')

    print("\nModelo otimizado treinado e salvo com sucesso.")

if __name__ == '__main__':
    train_and_evaluate_model()

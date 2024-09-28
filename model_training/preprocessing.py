import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

def preprocess_data(df):
    # Separando as features (X) e o target (y)
    X = df.drop('class', axis=1)
    y = df['class']

    # Codificando as classes do target
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Dividindo os dados em treinamento (80%) e teste (20%)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

    # Retornando os conjuntos de treino e teste, e o encoder
    return X_train, X_test, y_train, y_test, label_encoder

if __name__ == '__main__':
    # Este bloco é executado apenas quando o script é executado diretamente
    from data_preparation import load_and_explore_data

    df = load_and_explore_data()
    X_train, X_test, y_train, y_test, label_encoder = preprocess_data(df)

    # Exibindo as dimensões dos conjuntos
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_test shape: {y_test.shape}")

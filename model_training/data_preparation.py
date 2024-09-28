from ucimlrepo import fetch_ucirepo
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_and_explore_data():
    # Carregando o dataset Iris do UCI Repository
    iris = fetch_ucirepo(id=53)

    # Exibindo os metadados do dataset
    print("Metadados do dataset:")
    print(iris.metadata)

    # Exibindo informações sobre as variáveis
    print("\nInformações sobre as variáveis:")
    print(iris.variables)

    # Obtendo as features e targets como DataFrames do pandas
    X = iris.data.features
    y = iris.data.targets

    # Visualizando as primeiras linhas das features
    print("\nPrimeiras 5 linhas das features:")
    print(X.head())

    # Visualizando as primeiras linhas dos targets
    print("\nPrimeiras 5 linhas dos targets:")
    print(y.head())

    # Verificando as dimensões das features e targets
    print(f"\nDimensões das features: {X.shape}")
    print(f"Dimensões dos targets: {y.shape}")

    # Verificando os tipos de dados
    print("\nTipos de dados das features:")
    print(X.dtypes)

    print("\nTipos de dados dos targets:")
    print(y.dtypes)

    # Checando valores nulos nas features
    print("\nValores nulos nas features:")
    print(X.isnull().sum())

    # Checando valores nulos nos targets
    print("\nValores nulos nos targets:")
    print(y.isnull().sum())

    # Estatísticas descritivas das features
    print("\nEstatísticas descritivas das features:")
    print(X.describe())

    # Distribuição das classes nos targets
    print("\nDistribuição das classes nos targets:")
    print(y.value_counts())

    # Combinando features e targets em um único DataFrame
    df = pd.concat([X, y], axis=1)

    # Renomeando a coluna target para 'class'
    df = df.rename(columns={df.columns[-1]: 'class'})

    print("\nPrimeiras 5 linhas do DataFrame combinado:")
    print(df.head())

    analyze_outliers(df)

    return df

def analyze_outliers(df):
    features = ['sepal length', 'sepal width', 'petal length', 'petal width']
    outliers_info = {}

    for feature in features:
        # Plotando o boxplot
        #plt.figure(figsize=(8, 4))
        #sns.boxplot(x='class', y=feature, data=df)
        #plt.title(f'Boxplot de {feature} por classe')
        #plt.show()

        # Calculando quartis e IQR para encontrar outliers
        Q1 = df[feature].quantile(0.25)
        Q3 = df[feature].quantile(0.75)
        IQR = Q3 - Q1

        # Defininindo os limites para outliers
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Identificando os outliers
        outliers = df[(df[feature] < lower_bound) | (df[feature] > upper_bound)]
        
        # Salvando as informações dos outliers
        outliers_info[feature] = {
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'outliers': outliers[[feature, 'class']]
        }

        # Exibindo informações sobre outliers
        print(f"\nOutliers detectados para {feature}:")
        print(f"Limite inferior: {lower_bound}, Limite superior: {upper_bound}")
        print(f"Outliers:\n{outliers[[feature, 'class']]}")

    return df, outliers_info



if __name__ == '__main__':
    df = load_and_explore_data()
    df, outliers_info = analyze_outliers(df)

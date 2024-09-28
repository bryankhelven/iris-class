
# Iris Classification API

## Descrição

Bem-vindo ao projeto **Iris Classification API**! Este projeto fornece uma API para classificar flores Iris em suas três espécies: **Iris-setosa**, **Iris-versicolor** e **Iris-virginica**, com base em medidas de comprimento e largura das sépalas e pétalas. A API foi construída usando FastAPI e o modelo de Machine Learning foi treinado utilizando Random Forest Classifier.

---

## Índice

- [Instruções para Iniciar e Usar a API](#instruções-para-iniciar-e-usar-a-api)
  - [Usando Docker](#usando-docker)
  - [Execução Local a partir do Repositório](#execução-local-a-partir-do-repositório)
  - [Execução da API](#execução-da-api)
  - [Parâmetros Esperados](#parâmetros-esperados)
  - [Exemplo 1 de Requisição com `curl`](#exemplo-1-de-requisição-com-curl)
  - [Exemplo 2 de Requisição com `curl`](#exemplo-2-de-requisição-com-curl)
  - [Exemplo de Requisição (JSON)](#exemplo-de-requisição-json)
  - [Exemplo de Requisição Acessando o Swagger UI](#exemplo-de-requisição-acessando-o-swagger-ui)
- [Reproduzindo a Pipeline de Treinamento do Modelo](#reproduzindo-a-pipeline-de-treinamento-do-modelo)
  - [Pré-requisitos](#pré-requisitos)
  - [Configuração do Ambiente](#configuração-do-ambiente)
  - [Executando o Treinamento do Modelo](#executando-o-treinamento-do-modelo)
  - [Análise Exploratória com Jupyter Notebook](#análise-exploratória-com-jupyter-notebook)
- [Estrutura do Projeto](#estrutura-do-projeto)


---

## Instruções para Iniciar e Usar a API

### Usando Docker

#### Pré-requisitos

- [Docker](https://www.docker.com/get-started) instalado em seu sistema.

#### Passos

1. **Clone o repositório**

   ```bash
   git clone https://github.com/seu-usuario/iris_class-api.git
   cd iris_class-api
   ```

2. **Construa a imagem Docker**

   ```bash
   docker build -t iris-classification-api .
   ```

3. **Execute o contêiner Docker**

   ```bash
   docker run -d -p 8000:8000 iris-classification-api
   ```

   - A API estará disponível em `http://localhost:8000`.

4. **Verifique se a API está em execução**

   - Acesse `http://localhost:8000/docs` em seu navegador para visualizar a documentação interativa gerada pelo Swagger UI.

### Execução Local a partir do Repositório

#### Pré-requisitos

- Python 3.8 ou superior
- [Git](https://git-scm.com/downloads)
- [Virtualenv](https://virtualenv.pypa.io/en/latest/installation.html) ou [Conda](https://docs.conda.io/en/latest/)

#### Passos

1. **Clone o repositório**

   ```bash
   git clone https://github.com/seu-usuario/iris_class-api.git
   cd iris_class-api
   ```

2. **Crie e ative um ambiente virtual**

   - Usando `virtualenv`:

     ```bash
     python3 -m venv venv
     source venv/bin/activate  # No Windows: venv\Scripts\activate
     ```

   - Ou usando `conda`:

     ```bash
     conda create -n iris-classification python=3.8
     conda activate iris-classification
     ```

3. **Instale as dependências**

   ```bash
   pip install -r requirements.txt
   ```

4. **Execute a aplicação**

   ```bash
   uvicorn main:app --reload
   ```

   - A aplicação será executada em `http://localhost:8000`.

5. **Verifique se a API está em execução**

   - Acesse `http://localhost:8000/docs` para visualizar a documentação interativa.

### Execução da API

Você pode testar a API usando `curl`, ferramentas como Postman e Insomnia, ou diretamente via o Swagger UI acessando o endpoint `/docs`.

#### Parâmetros Esperados

A API espera receber quatro variáveis numéricas no corpo da requisição JSON:

- `sepal_length`: Comprimento da sépala (formato numérico)
- `sepal_width`: Largura da sépala (formato numérico)
- `petal_length`: Comprimento da pétala (formato numérico)
- `petal_width`: Largura da pétala (formato numérico)

Esses valores devem ser enviados no formato correto (números com o sem decimais, separados por ponto) e não podem conter letras ou pontuações extras, pois isso resultará em erro na requisição.


#### Exemplo 1 de Requisição com `curl`

```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
           "sepal_length": 5.1,
           "sepal_width": 3,
           "petal_length": 1.4,
           "petal_width": 0.2
         }'
```

#### Resposta Esperada do exemplo 1

```json
{
  "predicted_class": "Iris-setosa",
  "predicted_class_index": 0,
  "probabilities": {
    "Iris-setosa": 1.0,
    "Iris-versicolor": 0.0,
    "Iris-virginica": 0.0
  }
}
```

#### Exemplo 2 de Requisição com `curl`
```bash
curl -X POST "http://localhost:8000/predict" -H "Content-Type: application/json" -d '{"sepal_length":2.1,"sepal_width":0.5,"petal_length":3.4,"petal_width":6.2}'
```

#### Resposta Esperada do exemplo 2
```json
{"predicted_class":"Iris-virginica","predicted_class_index":2,"probabilities":{"Iris-setosa":0.006666666666666666,"Iris-versicolor":0.43772222222222223,"Iris-virginica":0.5556111111111111}}
```

#### Exemplo de Requisição (JSON)

Se preferir, aqui está um JSON que pode ser enviado em uma requisição POST para a API:
```json
{
  "sepal_length": 5.1,
  "sepal_width": 3.5,
  "petal_length": 1.4,
  "petal_width": 0.2
}
```

#### Exemplo de Requisição Acessando o Swagger UI
Você também pode acessar a documentação interativa da API através do Swagger UI no endpoint `/docs` acessando o Swagger UI em `http://localhost:8000/docs.`
Nesta interface, você poderá explorar os endpoints disponíveis, preencher os parâmetros necessários e testar a API diretamente no navegador, sem necessidade de ferramentas externas.

Para testar clique em `POST/predict`, depois em `Try it out` e no request body copie um json ou modifique os valores das quatro variáveis para os desejados.
Clique no botão `execute` e verifique a resposta no campo `Response body` que aparecerá abaixo.


---

## Reproduzindo a Pipeline de Treinamento do Modelo

### Pré-requisitos

- Python 3.8 ou superior
- Ambiente virtual configurado (Virtualenv ou Conda)
- Dependências listadas em `requirements.txt`

### Configuração do Ambiente

1. **Clone o repositório e navegue até o diretório**

   ```bash
   git clone https://github.com/seu-usuario/iris_class-api.git
   cd iris_class-api
   ```

2. **Crie e ative um ambiente virtual**

   - Usando `virtualenv`:

     ```bash
     python3 -m venv venv
     source venv/bin/activate  # No Windows: venv\Scripts\activate
     ```

   - Ou usando `conda`:

     ```bash
     conda create -n iris-classification python=3.8
     conda activate iris-classification
     ```

3. **Instale as dependências**

   ```bash
   pip install -r requirements.txt
   ```

### Executando o Treinamento do Modelo

Os scripts relacionados ao treinamento do modelo estão localizados na pasta `model_training`.

1. **Navegue até o diretório `model_training`**

   ```bash
   cd model_training
   ```

2. **Execute o script de treinamento**

   ```bash
   python model_training.py
   ```

   - Este script irá:
     - Carregar e explorar os dados.
     - Pré-processar os dados.
     - Treinar o modelo inicial.
     - Realizar ajuste de hiperparâmetros com validação cruzada.
     - Avaliar o modelo otimizado.
     - Salvar o modelo treinado e o `LabelEncoder` na pasta `models`.

Os arquivos individuais `data_preparation.py` e `preprocessing.py` também podem ser executados de forma independente, e eles exibem informações pertinentes:
     - `data_preparation.py`: Processamento e visualização das features e targets, trazendo detalhes sobre a distribuição dos dados.
     - `preprocessing.py`: Transformações aplicadas aos dados, como normalização e codificação de rótulos.

Esses dois arquivos integram a pipeline completa chamada pelo script model_training.py, estando separados a fim de facilitar a modificação das etapas intermediárias do fluxo de treinamento.

### Análise Exploratória com Jupyter Notebook

Para aqueles que preferem uma abordagem mais visual, há um notebook Jupyter disponível.

1. **Certifique-se de que o Jupyter Notebook está instalado**

   ```bash
   pip install notebook
   ```

2. **Navegue até a pasta `notebooks`**

   ```bash
   cd notebooks
   ```

3. **Inicie o Jupyter Notebook**

   ```bash
   jupyter notebook
   ```

4. **Abra o arquivo `exploratory_analysis.ipynb`**

   - O notebook contém:
     - Análise exploratória dos dados.
     - Visualizações gráficas.
     - Etapas de pré-processamento.
     - Treinamento e avaliação do modelo.

---

## Estrutura do Projeto

```
iris_class-api/
├── main.py
├── models/
│   ├── label_encoder.joblib
│   └── random_forest_model.joblib
├── model_training/
│   ├── __init__.py
│   ├── data_preparation.py
│   ├── preprocessing.py
│   └── model_training.py
├── notebooks/
│   └── exploratory_analysis.ipynb
├── requirements.txt
├── Dockerfile
├── README.md
├── .gitignore
└── tests/
    └── test_api.py
```

- **`main.py`**: Arquivo principal da API.
- **`models/`**: Contém o modelo treinado e o `LabelEncoder`.
- **`model_training/`**: Scripts relacionados ao treinamento do modelo.
- **`notebooks/`**: Contém o notebook Jupyter para análise exploratória.
- **`requirements.txt`**: Lista de dependências do projeto.
- **`Dockerfile`**: Arquivo para construir a imagem Docker.
- **`.gitignore`**: Arquivo que define os padrões de exclusão para o Git.
- **`tests/`**: Contém os testes automatizados para a API.

---

# Instruções Adicionais

Certifique-se de substituir os placeholders como `seu-usuario` com suas informações reais.

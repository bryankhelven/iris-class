FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .

# Instala as dependências e copia tudo do diretório atual para o de trabalho
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

# Expõe a porta de execução
EXPOSE 8000

# Inicia a API
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

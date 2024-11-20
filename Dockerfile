# Usar uma imagem base do Python
FROM python:3.9-slim

# Definir o diretório de trabalho
WORKDIR /app

# Copiar o arquivo requirements.txt para o diretório de trabalho
COPY requirements.txt .

# Instalar as dependências
RUN pip install --no-cache-dir -r requirements.txt

# Copiar o arquivo app.py e o dataset para o diretório de trabalho
COPY app.py .
COPY laptop_prices.csv .

# Expor a porta 3000
EXPOSE 3000

# Comando para executar o aplicativo
CMD ["gunicorn", "--bind", "0.0.0.0:3000", "app:app"]
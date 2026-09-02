# Usa uma imagem oficial leve do Python
FROM python:3.10-slim

# Define o diretório de trabalho dentro do contêiner
WORKDIR /app

# Instala dependências do sistema necessárias para processamento de imagem
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copia e instala as dependências do Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia o restante do código e o modelo para o contêiner
COPY . .

# Expõe a porta que o Render vai usar (geralmente gerenciada internamente)
EXPOSE 5000

# Comando para rodar a aplicação usando Gunicorn em produção
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app", "--timeout", "120"]


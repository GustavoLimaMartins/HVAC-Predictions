FROM python:3.11-slim

WORKDIR /app

# Garante que logs Python apareçam imediatamente no Railway (sem buffering)
ENV PYTHONUNBUFFERED=1

# Instalar dependências do sistema
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements
COPY requirements.txt .

# Atualizar pip e instalar dependências
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Copiar código da aplicação
COPY . .

# Railway injeta $PORT em runtime (geralmente 8080).
# EXPOSE é apenas documentação — não afeta o binding real.
EXPOSE 8000

# Railway ignora HEALTHCHECK do Docker e usa seu próprio sistema
# (configurado via healthcheckPath no railway.json).
# Removido HEALTHCHECK do Dockerfile para evitar confusão.

# exec substitui o shell por uvicorn como PID 1, garantindo que
# SIGTERM do Railway seja recebido diretamente para graceful shutdown.
# ${PORT:-8000} usa a porta do Railway ou fallback 8000 para uso local.
CMD ["sh", "-c", "exec python -m uvicorn tools.api_server:app --host 0.0.0.0 --port ${PORT:-8000}"]

FROM python:3.11-slim

WORKDIR /app

# Dependencias del sistema (xgboost/lightgbm necesitan libgomp1)
RUN apt-get update && apt-get install -y \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Instalar dependencias Python primero (mejor cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el resto del backend (incluye outputs/ ya generados)
COPY . .

EXPOSE 8000

CMD ["sh", "-c", "uvicorn src.api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
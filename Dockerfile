FROM python:3.11-slim

# Установка системных зависимостей для GDAL
RUN apt-get update && apt-get install -y \
    gdal-bin \
    libgdal-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Установка Python-зависимостей
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копирование кода
COPY app.py .

# Команда для запуска
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
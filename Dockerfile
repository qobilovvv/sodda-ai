FROM python:3.10-slim

WORKDIR /app

# Install system dependencies for pymysql and other libs if needed
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Environment variables will be loaded from docker-compose or .env
CMD ["python", "bot.py"]

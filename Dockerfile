FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install Playwright's Chromium browser + its system-level OS dependencies
RUN playwright install --with-deps chromium

COPY . .

EXPOSE 8080

CMD ["sh", "-c", "gunicorn -k uvicorn.workers.UvicornWorker server:app --bind 0.0.0.0:${PORT} --workers 2 --timeout 120"]
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY main.py .
# Set LEDGER_EXPORT_URL, LEDGER_RATES_URL and AIPIPE_TOKEN as secret env vars on the host.
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080}"]

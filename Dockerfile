FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

ENV PORT=8080

WORKDIR /app/app

CMD ["gunicorn", "--bind", "0.0.0.0:8080", "main:app"]
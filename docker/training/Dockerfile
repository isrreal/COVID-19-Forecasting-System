FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update \
 && apt-get install -y gcc \
 && rm -rf /var/lib/apt/lists/*

COPY requirements /tmp/requirements

RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r /tmp/training.txt

COPY . .

RUN groupadd -g 1000 appgroup \
 && useradd -m -u 1000 -g appgroup appuser \
 && chown -R appuser:appgroup /app

USER appuser
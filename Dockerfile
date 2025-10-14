FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt
COPY . .

RUN groupadd -r appgroup && useradd -r -g appgroup appuser

RUN mkdir -p /artifacts && chown -R appuser:appgroup /artifacts

RUN chown -R appuser:appgroup /app

USER appuser

CMD ["python", "main.py"]

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements/api.txt /tmp/api.txt
COPY requirements/base.txt /tmp/base.txt

RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r /tmp/api.txt

COPY . .

RUN groupadd -g 1000 appgroup \
 && useradd -m -u 1000 -g appgroup appuser \
 && chown -R appuser:appgroup /app

USER appuser

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

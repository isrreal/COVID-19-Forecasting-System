FROM python:3.11-slim

# evitar .pyc e buffer
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# instalar dependências do sistema necessárias
RUN apt-get update \
 && apt-get install -y gcc libpq-dev \
 && rm -rf /var/lib/apt/lists/*

# copiar requirements
COPY requirements/api.txt /tmp/api.txt
COPY requirements/base.txt /tmp/base.txt

RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r /tmp/api.txt

# copiar código
COPY . .

# criar usuário não-root
RUN groupadd -g 1000 appgroup \
 && useradd -m -u 1000 -g appgroup appuser \
 && chown -R appuser:appgroup /app

USER appuser

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
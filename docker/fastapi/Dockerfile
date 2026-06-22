# ==============================================================================
# Stage 1 - base
# Common foundation: Python version, env vars, workdir and production deps.
# Both subsequent stages inherit from here so layers are shared and cached.
# ==============================================================================
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements/base.txt /tmp/base.txt
COPY requirements/api.txt /tmp/api.txt

RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r /tmp/api.txt

# ==============================================================================
# Stage 2 - production
# Copies source code, creates a non-root user and sets the startup command.
# This is the image used by docker compose up.
# ==============================================================================
FROM base AS production

COPY . .

RUN groupadd -g 1000 appgroup \
 && useradd -m -u 1000 -g appgroup appuser \
 && chown -R appuser:appgroup /app

USER appuser

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

# ==============================================================================
# Stage 3 - test
# Extends base (not production) to add dev dependencies.
# Runs as root — no need for the non-root user in a test context.
# Docker layer cache means deps are only reinstalled when dev.txt changes.
# ==============================================================================
FROM base AS test

COPY requirements/dev.txt /tmp/dev.txt

RUN pip install --no-cache-dir -r /tmp/dev.txt

COPY . .

CMD ["python", "-m", "pytest", "tests/", "-v", "--tb=short"]

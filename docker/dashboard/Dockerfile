FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements/dashboard.txt /tmp/dashboard.txt
COPY requirements/base.txt /tmp/base.txt

RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r /tmp/dashboard.txt

COPY dashboard/ ./dashboard/

EXPOSE 8501

CMD ["streamlit", "run", "dashboard/Home.py", "--server.port=8501", "--server.address=0.0.0.0"]

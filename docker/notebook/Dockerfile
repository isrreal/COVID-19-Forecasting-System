FROM jupyter/base-notebook:python-3.11

WORKDIR /app

COPY requirements /tmp/requirements

RUN pip install --no-cache-dir -r /tmp/requirements/notebook.txt

RUN chown -R jovyan:users /app
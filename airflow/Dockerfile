FROM apache/airflow:3.2.0

USER root

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

USER airflow

# Path is relative to project root (DA5402_E2E/)
COPY airflow/requirements.txt /tmp/airflow-requirements.txt

RUN pip install --no-cache-dir \
    -r /tmp/airflow-requirements.txt \
    --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-3.2.0/constraints-3.13.txt" \
    --extra-index-url https://download.pytorch.org/whl/cpu

# Same — relative to project root
COPY --chown=airflow:root src/ /opt/airflow/src
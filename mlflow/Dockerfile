# MLflow tracking server
# Using slim to keep image small — no ML libs needed here
FROM python:3.11-slim

# Keeps Python from buffering stdout/stderr
# You want logs to appear immediately in docker logs
ENV PYTHONUNBUFFERED=1

WORKDIR /mlflow

# Copy and install dependencies first (before any other files)
# This layer gets cached — won't re-run unless requirements.txt changes
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Directory where artifacts will be stored inside the container
# This maps to the volume you define in docker-compose
RUN mkdir -p /mlflow/artifacts

# Expose the default MLflow port
EXPOSE 5000

# All connection details come from env vars set in docker-compose
# --host 0.0.0.0 is critical — without it the server only listens
# on localhost inside the container and is unreachable from outside
CMD mlflow server \
    --backend-store-uri postgresql://${MLFLOW_DB_USER}:${MLFLOW_DB_PASSWORD}@${MLFLOW_DB_HOST}:5432/${MLFLOW_DB_NAME} \
    --artifacts-destination /mlflow/artifacts \
    --host 0.0.0.0 \
    --port 5000
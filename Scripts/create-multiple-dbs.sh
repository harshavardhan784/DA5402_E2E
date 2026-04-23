#!/bin/bash
set -e

# Create roles first
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" <<-EOSQL
    CREATE USER airflow WITH PASSWORD 'airflow123';
    CREATE USER mlflow  WITH PASSWORD 'mlflow123';
EOSQL

# Create databases and grant ownership
for db in airflow mlflow; do
    psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" <<-EOSQL
        CREATE DATABASE $db;
        GRANT ALL PRIVILEGES ON DATABASE $db TO $db;
        ALTER DATABASE $db OWNER TO $db;
EOSQL
done
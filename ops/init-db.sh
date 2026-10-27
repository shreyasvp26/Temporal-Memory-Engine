#!/usr/bin/env bash
set -e

echo "Initializing Temporal Memory Engine database..."

# Ensure auth schema stub exists first
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
    CREATE SCHEMA IF NOT EXISTS auth;
    CREATE TABLE IF NOT EXISTS auth.users (id uuid primary key);
EOSQL

# Apply migration files in sorted numerical order
for f in $(ls /docker-entrypoint-initdb.d/migrations/*.sql | sort); do
    echo "Applying migration $f..."
    psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" -f "$f"
done

# Configure passwords and grants for application roles
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
    ALTER ROLE app_api WITH PASSWORD 'api_pass';
    ALTER ROLE app_worker WITH PASSWORD 'worker_pass';
    GRANT USAGE ON SCHEMA auth TO app_api, app_worker;
    GRANT ALL ON TABLE auth.users TO app_api, app_worker;
EOSQL

echo "Temporal Memory Engine database initialized successfully."

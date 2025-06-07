#!/bin/bash
set -e

# Función para logging
log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1"
}

log "Starting Trading Bot container..."

# Verificar variables de entorno críticas
if [ -z "$MT5_LOGIN" ] || [ -z "$MT5_PASSWORD" ] || [ -z "$MT5_SERVER" ]; then
    log "ERROR: MT5 credentials not set. Please check your environment variables."
    exit 1
fi

# Esperar a que PostgreSQL esté listo
if [ ! -z "$DATABASE_URL" ]; then
    log "Waiting for PostgreSQL..."
    while ! pg_isready -h ${DB_HOST:-postgres} -p ${DB_PORT:-5432} -U ${DB_USER:-tradingbot}; do
        sleep 1
    done
    log "PostgreSQL is ready!"
fi

# Esperar a que Redis esté listo
if [ ! -z "$REDIS_URL" ]; then
    log "Waiting for Redis..."
    while ! redis-cli -h ${REDIS_HOST:-redis} -p ${REDIS_PORT:-6379} ping > /dev/null 2>&1; do
        sleep 1
    done
    log "Redis is ready!"
fi

# Ejecutar migraciones si es necesario
if [ "$RUN_MIGRATIONS" = "true" ]; then
    log "Running database migrations..."
    python manage.py migrate
fi

# Crear directorios si no existen
mkdir -p /app/logs /app/data_storage/market_data /app/data_storage/model_artifacts

# Establecer permisos correctos
chown -R tradingbot:tradingbot /app/logs /app/data_storage

# Ejecutar comando
log "Executing command: $@"
exec "$@"
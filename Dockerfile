# ============================================
# ETAPA 1: Base con dependencias del sistema
# ============================================
FROM python:3.10-slim-bullseye AS base

# Variables de entorno
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Herramientas básicas
    curl \
    wget \
    git \
    vim \
    htop \
    # Compiladores y librerías
    build-essential \
    gcc \
    g++ \
    gfortran \
    # Librerías científicas
    libblas-dev \
    liblapack-dev \
    libatlas-base-dev \
    # TA-Lib dependencies
    libta-lib0-dev \
    # PostgreSQL client
    postgresql-client \
    # Redis client
    redis-tools \
    # Otras librerías necesarias
    libffi-dev \
    libssl-dev \
    libxml2-dev \
    libxslt1-dev \
    libjpeg-dev \
    zlib1g-dev \
    # Limpieza
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Instalar TA-Lib desde source
RUN cd /tmp && \
    wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz && \
    tar -xzf ta-lib-0.4.0-src.tar.gz && \
    cd ta-lib && \
    ./configure --prefix=/usr && \
    make && \
    make install && \
    cd .. && \
    rm -rf ta-lib ta-lib-0.4.0-src.tar.gz

# ============================================
# ETAPA 2: Dependencias de Python
# ============================================
FROM base AS dependencies

# Copiar archivos de requirements
COPY requirements.txt requirements-prod.txt* /tmp/

# Actualizar pip y setuptools
RUN pip install --upgrade pip setuptools wheel

# Instalar dependencias de Python
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Instalar dependencias adicionales de producción si existen
RUN if [ -f /tmp/requirements-prod.txt ]; then \
        pip install --no-cache-dir -r /tmp/requirements-prod.txt; \
    fi

# ============================================
# ETAPA 3: Aplicación
# ============================================
FROM dependencies AS application

# Crear usuario no-root
RUN groupadd -r tradingbot && \
    useradd -r -g tradingbot -d /home/tradingbot -s /bin/bash tradingbot && \
    mkdir -p /home/tradingbot && \
    chown -R tradingbot:tradingbot /home/tradingbot

# Directorio de trabajo
WORKDIR /app

# Copiar código fuente
COPY --chown=tradingbot:tradingbot . /app/

# Crear directorios necesarios
RUN mkdir -p \
    /app/logs \
    /app/data_storage/market_data \
    /app/data_storage/model_artifacts \
    /app/data_storage/backtest_results \
    /app/reports \
    /app/temp \
    && chown -R tradingbot:tradingbot /app

# Instalar la aplicación en modo editable
RUN pip install -e .

# ============================================
# ETAPA 4: Configuración de producción
# ============================================
FROM application AS production

# Cambiar a usuario no-root
USER tradingbot

# Variables de entorno de producción
ENV ENVIRONMENT=production \
    PYTHONPATH=/app:$PYTHONPATH

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import sys; sys.path.insert(0, '/app'); from utils.health_check import check_health; exit(0 if check_health() else 1)"

# Exponer puertos
EXPOSE 8000 8501

# Volúmenes
VOLUME ["/app/logs", "/app/data_storage", "/app/reports"]

# Script de entrada
COPY --chown=tradingbot:tradingbot docker-entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

ENTRYPOINT ["docker-entrypoint.sh"]

# Comando por defecto
CMD ["python", "main.py"]

# ============================================
# ETAPA 5: Desarrollo (opcional)
# ============================================
FROM application AS development

# Volver a root para instalar herramientas de desarrollo
USER root

# Instalar herramientas de desarrollo
RUN apt-get update && apt-get install -y --no-install-recommends \
    iputils-ping \
    net-tools \
    strace \
    tcpdump \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Instalar dependencias de desarrollo
COPY requirements-dev.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements-dev.txt

# Configurar Git
RUN git config --global --add safe.directory /app

# Cambiar a usuario tradingbot
USER tradingbot

# Variables de entorno de desarrollo
ENV ENVIRONMENT=development \
    PYTHONPATH=/app:$PYTHONPATH \
    PYTHONDEBUG=1

# Comando para desarrollo
CMD ["python", "-m", "debugpy", "--listen", "0.0.0.0:5678", "--wait-for-client", "main.py"]
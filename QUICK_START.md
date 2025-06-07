# 🚀 INICIO RÁPIDO - TradingBot Cuantitativo MT5

## ✅ Correcciones Realizadas

### 1. **Archivos __init__.py creados** ✓
Se han creado archivos `__init__.py` en todos los directorios de código:
- ✓ config/
- ✓ core/
- ✓ data/
- ✓ models/
- ✓ strategies/
- ✓ risk/
- ✓ analysis/
- ✓ control/
- ✓ visualization/
- ✓ utils/
- ✓ tests/
- ✓ scripts/

### 2. **Archivos críticos creados** ✓
- ✓ `tradingbot/__version__.py` - Información de versión
- ✓ `tradingbot/__init__.py` - Inicialización del paquete principal
- ✓ `.env` - Copiado desde `.env.example`
- ✓ `requirements-prod.txt` - Dependencias de producción
- ✓ `__init__.py` en raíz - Para importaciones
- ✓ `scripts/setup_project.py` - Script de configuración inicial

### 3. **Estructura de directorios mejorada** ✓
- ✓ Creado directorio `scripts/` para utilidades
- ✓ Creado directorio `tradingbot/` para compatibilidad con setup.py

## 📋 Pasos para Configurar el Proyecto

### 1. **Configurar el entorno virtual**
```bash
# Crear entorno virtual
python -m venv venv

# Activar entorno virtual
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate
```

### 2. **Instalar dependencias**
```bash
# Instalar todas las dependencias
pip install -r requirements.txt

# O solo las de producción
pip install -r requirements-prod.txt
```

### 3. **Configurar credenciales**
Edita el archivo `.env` y configura tus credenciales:
```bash
# Credenciales MT5 (REQUERIDO)
MT5_LOGIN=tu_numero_de_cuenta
MT5_PASSWORD=tu_contraseña_segura
MT5_SERVER=nombre_servidor_broker

# Base de datos (Opcional pero recomendado)
DATABASE_URL=postgresql://usuario:contraseña@localhost:5432/tradingbot_db
```

### 4. **Ejecutar setup inicial**
```bash
python scripts/setup_project.py
```

### 5. **Verificar instalación de MT5**
```bash
python -c "import MetaTrader5 as mt5; print('MT5 OK' if mt5.initialize() else 'MT5 Error')"
```

### 6. **Ejecutar en modo Paper Trading**
```bash
python main.py --mode paper
```

### 7. **Acceder al Dashboard**
El dashboard estará disponible en: http://localhost:8501

## ⚠️ Notas Importantes

1. **MetaTrader 5**: Solo funciona en Windows. Para Linux/Mac usar Wine o VM.
2. **TA-Lib**: Requiere instalación de binarios del sistema.
3. **GPU Support**: Para TensorFlow GPU, instalar: `pip install tensorflow[and-cuda]`

## 🔧 Configuración Adicional

### Base de Datos PostgreSQL
```bash
# Crear base de datos
createdb tradingbot_db

# Ejecutar migraciones (cuando estén disponibles)
python scripts/migrate_db.py
```

### Redis (Cache)
```bash
# Instalar Redis
# Windows: Descargar desde https://github.com/microsoftarchive/redis/releases
# Linux: sudo apt-get install redis-server
# Mac: brew install redis

# Iniciar Redis
redis-server
```

## 📞 Soporte

- Documentación completa: `README.md`
- Logs del sistema: `logs/`
- Estado del setup: `.setup_status.json`

---
**¡El proyecto está listo para comenzar!** 🎉
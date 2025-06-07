"""
TradingBot Cuantitative MT5 - Main Package
==========================================

Sistema avanzado de trading algorítmico con Machine Learning para MetaTrader 5
"""

# Hacer que los módulos principales sean accesibles desde la raíz
import sys
from pathlib import Path

# Agregar el directorio actual al path para facilitar importaciones
sys.path.insert(0, str(Path(__file__).parent))

# Información del paquete
__version__ = "0.1.0"
__author__ = "Andres"
__email__ = "andres@example.com"

# Importaciones para acceso conveniente
try:
    from core.trading_bot import TradingBot
    from core.mt5_connector import MT5Connector
    from core.trade_executor import TradeExecutor
    from strategies.ml_strategy import MLStrategy
    from risk.risk_manager import RiskManager
    from data.data_collector import DataCollector
    from models.ml_models import ModelFactory
except ImportError:
    pass  # Permitir importación parcial durante instalación

__all__ = [
    "__version__",
    "TradingBot",
    "MT5Connector", 
    "TradeExecutor",
    "MLStrategy",
    "RiskManager",
    "DataCollector",
    "ModelFactory"
]
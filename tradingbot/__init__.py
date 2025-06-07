"""
TradingBot Cuantitativo MT5
==========================

Sistema avanzado de trading algorítmico con Machine Learning para MetaTrader 5
"""

from tradingbot.__version__ import __version__, __version_info__

# Importar componentes principales para acceso directo
try:
    from core.trading_bot import TradingBot
    from core.mt5_connector import MT5Connector
    from strategies.ml_strategy import MLStrategy
    from risk.risk_manager import RiskManager
except ImportError:
    pass  # Permitir importación parcial durante la instalación

__all__ = [
    "__version__",
    "__version_info__",
    "TradingBot",
    "MT5Connector",
    "MLStrategy",
    "RiskManager"
]
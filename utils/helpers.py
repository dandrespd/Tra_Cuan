import numpy as np, pandas as pd
from typing import Union, List, Dict, Any
from datetime import datetime, timedelta
import MetaTrader5 as mt5

# Conversiones de timeframe
def mt5_timeframe_to_pandas(timeframe: int) -> str:
    """Convierte MT5.TIMEFRAME_X a frecuencia pandas"""
    
def pandas_freq_to_minutes(freq: str) -> int:
    """Convierte frecuencia pandas a minutos"""

# Cálculos de trading
def calculate_pip_value(symbol: str, volume: float, account_currency: str) -> float:
    """Calcula valor monetario de un pip"""
    
def normalize_price(price: float, symbol_info: SymbolInfo) -> float:
    """Normaliza precio según digits del símbolo"""
    
def calculate_commission(symbol: str, volume: float, broker_config: Dict) -> float:
    """Calcula comisión estimada de una operación"""

# Gestión de tiempo
class TradingSession:
    """Maneja sesiones de trading por mercado"""
    ASIAN = ("00:00", "09:00")
    EUROPEAN = ("08:00", "17:00")
    AMERICAN = ("13:00", "22:00")
    
def is_trading_session(timezone: str, session: TradingSession) -> bool:
    """Verifica si estamos en sesión de trading"""
    
def next_trading_day(date: datetime, market: str = "forex") -> datetime:
    """Calcula siguiente día hábil de trading"""

# Decoradores útiles
def retry_on_error(max_attempts: int = 3, delay: float = 1.0, 
                   exceptions: Tuple = (Exception,)):
    """Decorador para reintentar en caso de error"""
    
def measure_time(logger_func=None):
    """Decorador para medir tiempo de ejecución"""
    
def require_connection(connection_attr: str = "connector"):
    """Decorador que verifica conexión antes de ejecutar"""
    
@lru_cache(maxsize=128)
def cached_with_ttl(ttl_seconds: int = 300):
    """Cache con tiempo de vida"""

# Validadores
def validate_symbol(symbol: str) -> bool:
    """Valida formato de símbolo"""
    
def validate_volume(volume: float, symbol_info: SymbolInfo) -> Tuple[bool, str]:
    """Valida volumen según especificaciones"""
    
def validate_price_levels(entry: float, sl: float, tp: float, 
                         order_type: str) -> Tuple[bool, str]:
    """Valida niveles de precio para una orden"""

# Serialización segura
class TradingBotJSONEncoder(json.JSONEncoder):
    """Encoder personalizado para objetos del bot"""
    
def safe_pickle_dump(obj: Any, filepath: Path) -> bool:
    """Guarda objeto con verificación de integridad"""
    
def safe_pickle_load(filepath: Path, expected_type: Type) -> Optional[Any]:
    """Carga objeto con validación de tipo"""
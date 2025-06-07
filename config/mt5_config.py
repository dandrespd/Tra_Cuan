from pathlib import Path
from dataclasses import dataclass
import os
from typing import Optional, Dict, Any

@dataclass
class MT5Symbol:
    """Especificaciones de un símbolo de trading"""
    name: str
    digits: int
    point: float
    tick_size: float
    tick_value: float
    volume_min: float
    volume_max: float
    volume_step: float
    contract_size: float
    margin_required: float
    swap_long: float
    swap_short: float
    
@dataclass
class MT5Config:
    """Configuración principal de MT5"""
    # Conexión
    login: int  # Desde env var MT5_LOGIN
    password: str  # Desde env var MT5_PASSWORD
    server: str  # Desde env var MT5_SERVER
    
    # Timeouts y reintentos
    connection_timeout: int = 60000
    max_reconnect_attempts: int = 5
    reconnect_delay: int = 5
    
    # Trading
    account_type: str = "HEDGE"  # or "NETTING"
    use_real_account: bool = False
    
    # Símbolos disponibles
    symbols: Dict[str, MT5Symbol]
    default_symbol: str = "EURUSD"
    
    # Configuración de órdenes
    order_magic: int = 234000
    order_comment_prefix: str = "TradingBot_"
    max_slippage: int = 20
    
    # Validaciones y límites
    min_free_margin_pct: float = 0.2
    max_orders_per_symbol: int = 3
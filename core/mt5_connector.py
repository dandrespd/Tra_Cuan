# core/mt5_connector.py
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union
import time
from dataclasses import dataclass
from enum import Enum

from config.settings import settings
from utils.log_config import get_logger, log_risk_alert

logger = get_logger('main')


class OrderType(Enum):
    """Tipos de órdenes"""
    BUY = mt5.ORDER_TYPE_BUY
    SELL = mt5.ORDER_TYPE_SELL
    BUY_LIMIT = mt5.ORDER_TYPE_BUY_LIMIT
    SELL_LIMIT = mt5.ORDER_TYPE_SELL_LIMIT
    BUY_STOP = mt5.ORDER_TYPE_BUY_STOP
    SELL_STOP = mt5.ORDER_TYPE_SELL_STOP


@dataclass
class Symbol:
    """Información del símbolo"""
    name: str
    digits: int
    point: float
    tick_size: float
    tick_value: float
    volume_min: float
    volume_max: float
    volume_step: float
    contract_size: float
    spread: float
    
    @classmethod
    def from_mt5(cls, symbol_info) -> 'Symbol':
        """Crear desde información de MT5"""
        return cls(
            name=symbol_info.name,
            digits=symbol_info.digits,
            point=symbol_info.point,
            tick_size=symbol_info.trade_tick_size,
            tick_value=symbol_info.trade_tick_value,
            volume_min=symbol_info.volume_min,
            volume_max=symbol_info.volume_max,
            volume_step=symbol_info.volume_step,
            contract_size=symbol_info.trade_contract_size,
            spread=symbol_info.spread
        )


@dataclass
class Position:
    """Información de posición"""
    ticket: int
    symbol: str
    type: str
    volume: float
    price_open: float
    price_current: float
    sl: float
    tp: float
    profit: float
    swap: float
    commission: float
    time_open: datetime
    magic: int
    comment: str
    
    @property
    def is_buy(self) -> bool:
        return self.type == mt5.POSITION_TYPE_BUY
    
    @property
    def is_profitable(self) -> bool:
        return self.profit > 0
    
    @property
    def pips(self) -> float:
        """Calcular pips de ganancia/pérdida"""
        if self.is_buy:
            return (self.price_current - self.price_open) / 0.0001
        else:
            return (self.price_open - self.price_current) / 0.0001


class MT5Connector:
    """Conector mejorado para MetaTrader 5"""
    
    def __init__(self, symbol: str, magic: int = 234000):
        self.symbol = symbol
        self.magic = magic
        self.connected = False
        self.symbol_info: Optional[Symbol] = None
        self._connection_attempts = 0
        self._max_attempts = 3
        
    def connect(self) -> bool:
        """Conectar a MT5 con reintentos"""
        while self._connection_attempts < self._max_attempts:
            try:
                if mt5.initialize():
                    self.connected = True
                    logger.info(f"✅ Conectado a MT5 - Build: {mt5.version()}")
                    
                    # Verificar símbolo
                    if self._validate_symbol():
                        return True
                    else:
                        logger.error(f"Símbolo {self.symbol} no válido")
                        return False
                        
            except Exception as e:
                logger.error(f"Error conectando a MT5: {e}")
                
            self._connection_attempts += 1
            if self._connection_attempts < self._max_attempts:
                logger.warning(f"Reintentando conexión ({self._connection_attempts}/{self._max_attempts})...")
                time.sleep(2)
        
        logger.error("No se pudo conectar a MT5")
        return False
    
    def disconnect(self):
        """Desconectar de MT5"""
        if self.connected:
            mt5.shutdown()
            self.connected = False
            logger.info("Desconectado de MT5")
    
    def _validate_symbol(self) -> bool:
        """Validar y configurar símbolo"""
        symbols = mt5.symbols_get(self.symbol)
        
        if not symbols:
            return False
        
        # Seleccionar símbolo
        if not mt5.symbol_select(self.symbol, True):
            return False
        
        # Obtener información
        symbol_info = mt5.symbol_info(self.symbol)
        if symbol_info is None:
            return False
        
        self.symbol_info = Symbol.from_mt5(symbol_info)
        logger.info(f"Símbolo configurado: {self.symbol} - Spread: {self.symbol_info.spread}")
        
        return True
    
    def get_account_info(self) -> Dict:
        """Obtener información de la cuenta"""
        if not self.connected:
            return {}
        
        account = mt5.account_info()
        if account is None:
            return {}
        
        return {
            'balance': account.balance,
            'equity': account.equity,
            'margin': account.margin,
            'free_margin': account.margin_free,
            'margin_level': account.margin_level,
            'profit': account.profit,
            'currency': account.currency,
            'leverage': account.leverage,
            'trade_allowed': account.trade_allowed,
            'trade_expert': account.trade_expert,
            'limit_orders': account.limit_orders
        }
    
    def get_market_data(self, timeframe: int, count: int, 
                       start_pos: int = 0) -> Optional[pd.DataFrame]:
        """Obtener datos históricos del mercado"""
        if not self.connected:
            return None
        
        try:
            rates = mt5.copy_rates_from_pos(self.symbol, timeframe, start_pos, count)
            
            if rates is None or len(rates) == 0:
                logger.error(f"No se pudieron obtener datos para {self.symbol}")
                return None
            
            # Convertir a DataFrame
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df.set_index('time', inplace=True)
            
            # Agregar columnas adicionales
            df['spread'] = df['ask'] - df['bid']
            df['mid'] = (df['ask'] + df['bid']) / 2
            df['range'] = df['high'] - df['low']
            
            return df
            
        except Exception as e:
            logger.error(f"Error obteniendo datos de mercado: {e}")
            return None
    
    def get_current_price(self) -> Optional[Dict]:
        """Obtener precio actual"""
        if not self.connected:
            return None
        
        tick = mt5.symbol_info_tick(self.symbol)
        if tick is None:
            return None
        
        return {
            'bid': tick.bid,
            'ask': tick.ask,
            'last': tick.last,
            'volume': tick.volume,
            'time': datetime.fromtimestamp(tick.time),
            'spread': tick.ask - tick.bid,
            'mid': (tick.ask + tick.bid) / 2
        }
    
    def get_positions(self, symbol: str = None) -> List[Position]:
        """Obtener posiciones abiertas"""
        if not self.connected:
            return []
        
        if symbol:
            positions = mt5.positions_get(symbol=symbol)
        else:
            positions = mt5.positions_get()
        
        if positions is None:
            return []
        
        return [
            Position(
                ticket=pos.ticket,
                symbol=pos.symbol,
                type='BUY' if pos.type == mt5.POSITION_TYPE_BUY else 'SELL',
                volume=pos.volume,
                price_open=pos.price_open,
                price_current=pos.price_current,
                sl=pos.sl,
                tp=pos.tp,
                profit=pos.profit,
                swap=pos.swap,
                commission=pos.commission,
                time_open=datetime.fromtimestamp(pos.time),
                magic=pos.magic,
                comment=pos.comment
            )
            for pos in positions
        ]
    
    def check_trading_allowed(self) -> Tuple[bool, str]:
        """Verificar si se permite trading"""
        if not self.connected:
            return False, "No conectado a MT5"
        
        # Verificar cuenta
        account = mt5.account_info()
        if not account.trade_allowed:
            return False, "Trading no permitido en la cuenta"
        
        if not account.trade_expert:
            return False, "Trading automático no permitido"
        
        # Verificar símbolo
        symbol_info = mt5.symbol_info(self.symbol)
        if not symbol_info.trade_mode == mt5.SYMBOL_TRADE_MODE_FULL:
            return False, f"Trading no permitido para {self.symbol}"
        
        # Verificar horario de mercado
        if not symbol_info.session_deals:
            return False, "Mercado cerrado"
        
        return True, "Trading permitido"
    
    def normalize_volume(self, volume: float) -> float:
        """Normalizar volumen según especificaciones del símbolo"""
        if not self.symbol_info:
            return volume
        
        # Redondear al step más cercano
        volume_step = self.symbol_info.volume_step
        normalized = round(volume / volume_step) * volume_step
        
        # Aplicar límites
        normalized = max(self.symbol_info.volume_min, 
                        min(normalized, self.symbol_info.volume_max))
        
        # Redondear a la precisión correcta
        decimals = str(volume_step)[::-1].find('.')
        if decimals > 0:
            normalized = round(normalized, decimals)
        else:
            normalized = int(normalized)
        
        return normalized
    
    def normalize_price(self, price: float) -> float:
        """Normalizar precio según especificaciones del símbolo"""
        if not self.symbol_info:
            return price
        
        return round(price, self.symbol_info.digits)
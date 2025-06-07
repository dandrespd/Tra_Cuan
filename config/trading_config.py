# 1. config/trading_config.py - Configuración de Trading
# config/trading_config.py
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import time
import json
from pathlib import Path
import MetaTrader5 as mt5

from utils.log_config import get_logger

logger = get_logger('main')


@dataclass
class TradingConfig:
    """Configuración completa para el sistema de trading"""
    
    # Símbolo y timeframe
    symbol: str = "EURUSD"
    timeframe: int = mt5.TIMEFRAME_M15
    
    # Parámetros de trading
    magic_number: int = 234000
    max_positions: int = 3
    max_spread: float = 2.0  # pips
    slippage: int = 20  # points
    
    # Gestión de riesgo
    risk_per_trade: float = 0.02  # 2% por operación
    max_daily_loss: float = 0.05  # 5% pérdida máxima diaria
    max_daily_trades: int = 10
    max_drawdown: float = 0.15  # 15% drawdown máximo
    
    # Tamaños de posición
    min_lot_size: float = 0.01
    max_lot_size: float = 1.0
    lot_step: float = 0.01
    
    # Stop Loss y Take Profit
    default_sl_pips: float = 30.0
    default_tp_pips: float = 50.0
    use_atr_stops: bool = True
    atr_multiplier_sl: float = 2.0
    atr_multiplier_tp: float = 3.0
    
    # Trailing stop
    use_trailing_stop: bool = True
    trailing_stop_pips: float = 20.0
    trailing_step_pips: float = 5.0
    
    # Horarios de trading
    trading_hours: Dict[str, time] = field(default_factory=lambda: {
        'start': time(8, 0),  # 8:00 AM
        'end': time(22, 0)    # 10:00 PM
    })
    
    # Días de trading (0=Lunes, 6=Domingo)
    trading_days: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])  # Lun-Vie
    
    # Configuración de señales
    min_confidence: float = 0.65  # Confianza mínima para operar
    signal_expiry_bars: int = 3   # Barras antes de que expire la señal
    
    # Filtros adicionales
    use_spread_filter: bool = True
    use_time_filter: bool = True
    use_news_filter: bool = True
    avoid_news_minutes: int = 30  # Minutos antes/después de noticias
    
    # Configuración de modelos
    model_update_frequency: str = "daily"  # "hourly", "daily", "weekly"
    retrain_after_trades: int = 100
    min_training_samples: int = 1000
    
    # Backtesting
    lookback_period: int = 200  # Barras para análisis
    
    # Sistema
    update_interval: int = 5  # Segundos entre actualizaciones
    close_on_stop: bool = False  # Cerrar posiciones al detener
    
    # Timeouts
    max_position_time: Optional[int] = None  # Horas máximas por posición
    order_timeout: int = 60  # Segundos para timeout de órdenes
    
    def __post_init__(self):
        """Validar configuración después de inicialización"""
        self._validate()
        self._log_configuration()
    
    def _validate(self):
        """Validar parámetros de configuración"""
        validations = []
        
        # Validar símbolo
        if not self.symbol:
            validations.append("Símbolo no puede estar vacío")
        
        # Validar riesgos
        if not 0 < self.risk_per_trade <= 0.1:
            validations.append("Riesgo por trade debe estar entre 0 y 10%")
        
        if not 0 < self.max_daily_loss <= 0.2:
            validations.append("Pérdida diaria máxima debe estar entre 0 y 20%")
        
        # Validar tamaños
        if self.min_lot_size <= 0:
            validations.append("Tamaño mínimo de lote debe ser mayor a 0")
        
        if self.max_lot_size < self.min_lot_size:
            validations.append("Tamaño máximo debe ser mayor al mínimo")
        
        # Validar stops
        if self.default_sl_pips <= 0:
            validations.append("Stop loss debe ser mayor a 0")
        
        if self.default_tp_pips <= 0:
            validations.append("Take profit debe ser mayor a 0")
        
        # Validar confianza
        if not 0 < self.min_confidence <= 1:
            validations.append("Confianza mínima debe estar entre 0 y 1")
        
        if validations:
            for error in validations:
                logger.error(f"Error de configuración: {error}")
            raise ValueError(f"Configuración inválida: {'; '.join(validations)}")
    
    def _log_configuration(self):
        """Registrar configuración en logs"""
        logger.info("="*60)
        logger.info("CONFIGURACIÓN DE TRADING")
        logger.info("="*60)
        
        config_sections = {
            "Trading": {
                "Símbolo": self.symbol,
                "Timeframe": self._timeframe_to_string(self.timeframe),
                "Magic Number": self.magic_number,
                "Max Posiciones": self.max_positions,
                "Max Spread": f"{self.max_spread} pips"
            },
            "Gestión de Riesgo": {
                "Riesgo por Trade": f"{self.risk_per_trade:.1%}",
                "Pérdida Diaria Max": f"{self.max_daily_loss:.1%}",
                "Max Trades Diarios": self.max_daily_trades,
                "Max Drawdown": f"{self.max_drawdown:.1%}"
            },
            "Stop Loss / Take Profit": {
                "SL Default": f"{self.default_sl_pips} pips",
                "TP Default": f"{self.default_tp_pips} pips",
                "Usar ATR": self.use_atr_stops,
                "Trailing Stop": self.use_trailing_stop
            },
            "Horarios": {
                "Hora Inicio": self.trading_hours['start'].strftime("%H:%M"),
                "Hora Fin": self.trading_hours['end'].strftime("%H:%M"),
                "Días": self._days_to_string(self.trading_days)
            },
            "Señales y Modelos": {
                "Confianza Mínima": f"{self.min_confidence:.1%}",
                "Actualización Modelo": self.model_update_frequency,
                "Retrain después de": f"{self.retrain_after_trades} trades"
            }
        }
        
        for section, params in config_sections.items():
            logger.info(f"\n{section}:")
            for key, value in params.items():
                logger.info(f"  {key}: {value}")
        
        logger.info("="*60)
    
    def _timeframe_to_string(self, timeframe: int) -> str:
        """Convertir timeframe a string legible"""
        timeframes = {
            mt5.TIMEFRAME_M1: "M1 (1 minuto)",
            mt5.TIMEFRAME_M5: "M5 (5 minutos)",
            mt5.TIMEFRAME_M15: "M15 (15 minutos)",
            mt5.TIMEFRAME_M30: "M30 (30 minutos)",
            mt5.TIMEFRAME_H1: "H1 (1 hora)",
            mt5.TIMEFRAME_H4: "H4 (4 horas)",
            mt5.TIMEFRAME_D1: "D1 (1 día)",
            mt5.TIMEFRAME_W1: "W1 (1 semana)",
            mt5.TIMEFRAME_MN1: "MN1 (1 mes)"
        }
        return timeframes.get(timeframe, f"Desconocido ({timeframe})")
    
    def _days_to_string(self, days: List[int]) -> str:
        """Convertir lista de días a string"""
        day_names = ['Lun', 'Mar', 'Mié', 'Jue', 'Vie', 'Sáb', 'Dom']
        return ', '.join([day_names[d] for d in sorted(days)])
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir configuración a diccionario"""
        return {
            'symbol': self.symbol,
            'timeframe': self.timeframe,
            'magic_number': self.magic_number,
            'max_positions': self.max_positions,
            'max_spread': self.max_spread,
            'slippage': self.slippage,
            'risk_per_trade': self.risk_per_trade,
            'max_daily_loss': self.max_daily_loss,
            'max_daily_trades': self.max_daily_trades,
            'max_drawdown': self.max_drawdown,
            'min_lot_size': self.min_lot_size,
            'max_lot_size': self.max_lot_size,
            'lot_step': self.lot_step,
            'default_sl_pips': self.default_sl_pips,
            'default_tp_pips': self.default_tp_pips,
            'use_atr_stops': self.use_atr_stops,
            'atr_multiplier_sl': self.atr_multiplier_sl,
            'atr_multiplier_tp': self.atr_multiplier_tp,
            'use_trailing_stop': self.use_trailing_stop,
            'trailing_stop_pips': self.trailing_stop_pips,
            'trailing_step_pips': self.trailing_step_pips,
            'trading_hours': {
                'start': self.trading_hours['start'].strftime("%H:%M"),
                'end': self.trading_hours['end'].strftime("%H:%M")
            },
            'trading_days': self.trading_days,
            'min_confidence': self.min_confidence,
            'signal_expiry_bars': self.signal_expiry_bars,
            'use_spread_filter': self.use_spread_filter,
            'use_time_filter': self.use_time_filter,
            'use_news_filter': self.use_news_filter,
            'avoid_news_minutes': self.avoid_news_minutes,
            'model_update_frequency': self.model_update_frequency,
            'retrain_after_trades': self.retrain_after_trades,
            'min_training_samples': self.min_training_samples,
            'lookback_period': self.lookback_period,
            'update_interval': self.update_interval,
            'close_on_stop': self.close_on_stop,
            'max_position_time': self.max_position_time,
            'order_timeout': self.order_timeout
        }
    
    def save(self, filepath: Path):
        """Guardar configuración en archivo"""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=4)
        
        logger.info(f"Configuración guardada en: {filepath}")
    
    @classmethod
    def load(cls, filepath: Path) -> 'TradingConfig':
        """Cargar configuración desde archivo"""
        filepath = Path(filepath)
        
        if not filepath.exists():
            logger.warning(f"Archivo de configuración no encontrado: {filepath}")
            logger.info("Usando configuración por defecto")
            return cls()
        
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            # Convertir horarios de string a time
            if 'trading_hours' in data:
                hours = data['trading_hours']
                data['trading_hours'] = {
                    'start': time.fromisoformat(hours['start']),
                    'end': time.fromisoformat(hours['end'])
                }
            
            config = cls(**data)
            logger.info(f"Configuración cargada desde: {filepath}")
            return config
            
        except Exception as e:
            logger.error(f"Error cargando configuración: {e}")
            logger.info("Usando configuración por defecto")
            return cls()
    
    def update(self, **kwargs):
        """Actualizar parámetros de configuración"""
        old_values = {}
        
        for key, value in kwargs.items():
            if hasattr(self, key):
                old_values[key] = getattr(self, key)
                setattr(self, key, value)
                logger.info(f"Configuración actualizada: {key} = {value} (anterior: {old_values[key]})")
            else:
                logger.warning(f"Parámetro de configuración desconocido: {key}")
        
        # Revalidar
        self._validate()
        
        return old_values
    
    def get_risk_parameters(self) -> Dict[str, float]:
        """Obtener parámetros de riesgo"""
        return {
            'risk_per_trade': self.risk_per_trade,
            'max_daily_loss': self.max_daily_loss,
            'max_drawdown': self.max_drawdown,
            'max_positions': self.max_positions,
            'max_daily_trades': self.max_daily_trades
        }
    
    def get_position_parameters(self) -> Dict[str, float]:
        """Obtener parámetros de posición"""
        return {
            'min_lot_size': self.min_lot_size,
            'max_lot_size': self.max_lot_size,
            'lot_step': self.lot_step,
            'default_sl_pips': self.default_sl_pips,
            'default_tp_pips': self.default_tp_pips,
            'use_atr_stops': self.use_atr_stops
        }


# Configuraciones predefinidas para diferentes estilos
class TradingProfiles:
    """Perfiles de trading predefinidos"""
    
    @staticmethod
    def conservative() -> TradingConfig:
        """Perfil conservador"""
        return TradingConfig(
            risk_per_trade=0.01,
            max_daily_loss=0.03,
            max_positions=1,
            default_sl_pips=20,
            default_tp_pips=30,
            min_confidence=0.75
        )
    
    @staticmethod
    def moderate() -> TradingConfig:
        """Perfil moderado (default)"""
        return TradingConfig()
    
    @staticmethod
    def aggressive() -> TradingConfig:
        """Perfil agresivo"""
        return TradingConfig(
            risk_per_trade=0.03,
            max_daily_loss=0.08,
            max_positions=5,
            default_sl_pips=40,
            default_tp_pips=80,
            min_confidence=0.60
        )
    
    @staticmethod
    def scalping() -> TradingConfig:
        """Perfil para scalping"""
        return TradingConfig(
            timeframe=mt5.TIMEFRAME_M1,
            risk_per_trade=0.005,
            max_daily_trades=50,
            default_sl_pips=5,
            default_tp_pips=10,
            use_trailing_stop=True,
            trailing_stop_pips=5,
            max_position_time=1  # 1 hora máximo
        )
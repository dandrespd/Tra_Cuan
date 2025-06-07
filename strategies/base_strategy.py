'''
14. strategies/base_strategy.py
Ruta: TradingBot_Cuantitative_MT5/strategies/base_strategy.py
Resumen:

Clase abstracta BaseStrategy que define la interfaz estándar para todas las estrategias de trading
Proporciona estructura común para generación de señales, gestión de estado, validación de condiciones
Incluye métodos para backtesting, optimización de parámetros y métricas de performance específicas
Framework extensible que permite crear estrategias simples o complejas manteniendo consistencia
'''
# strategies/base_strategy.py
import abc
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union, Tuple, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

from utils.log_config import get_logger, log_trade, log_performance
from models.model_hub import ModelHub

logger = get_logger('strategies')


class SignalType(Enum):
    """Tipos de señales de trading"""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    CLOSE_LONG = "CLOSE_LONG"
    CLOSE_SHORT = "CLOSE_SHORT"
    NO_SIGNAL = "NO_SIGNAL"


class SignalStrength(Enum):
    """Fuerza de la señal"""
    WEAK = 1
    MODERATE = 2
    STRONG = 3
    VERY_STRONG = 4


@dataclass
class TradingSignal:
    """Señal de trading completa"""
    signal_type: SignalType
    confidence: float  # 0.0 a 1.0
    strength: SignalStrength
    price: float
    timestamp: datetime
    
    # Niveles de precio
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    entry_price: Optional[float] = None
    
    # Metadatos
    strategy_name: str = ""
    symbol: str = ""
    timeframe: str = ""
    reasoning: str = ""
    features_used: List[str] = field(default_factory=list)
    risk_reward_ratio: Optional[float] = None
    
    # Validez temporal
    valid_until: Optional[datetime] = None
    max_hold_time: Optional[timedelta] = None
    
    def __post_init__(self):
        """Validación y cálculos automáticos post-inicialización"""
        # Validar confianza
        self.confidence = max(0.0, min(1.0, self.confidence))
        
        # Calcular ratio riesgo/beneficio si es posible
        if self.stop_loss and self.take_profit and self.entry_price:
            if self.signal_type in [SignalType.BUY]:
                risk = abs(self.entry_price - self.stop_loss)
                reward = abs(self.take_profit - self.entry_price)
            elif self.signal_type in [SignalType.SELL]:
                risk = abs(self.stop_loss - self.entry_price)
                reward = abs(self.entry_price - self.take_profit)
            else:
                risk = reward = 0
            
            if risk > 0:
                self.risk_reward_ratio = reward / risk
    
    @property
    def is_valid(self) -> bool:
        """Verificar si la señal sigue siendo válida"""
        if self.valid_until and datetime.now() > self.valid_until:
            return False
        return True
    
    @property
    def is_actionable(self) -> bool:
        """Verificar si la señal es accionable"""
        return (
            self.is_valid and 
            self.signal_type != SignalType.NO_SIGNAL and
            self.confidence > 0.5
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir señal a diccionario"""
        return {
            'signal_type': self.signal_type.value,
            'confidence': self.confidence,
            'strength': self.strength.value,
            'price': self.price,
            'timestamp': self.timestamp.isoformat(),
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'entry_price': self.entry_price,
            'strategy_name': self.strategy_name,
            'symbol': self.symbol,
            'timeframe': self.timeframe,
            'reasoning': self.reasoning,
            'features_used': self.features_used,
            'risk_reward_ratio': self.risk_reward_ratio,
            'valid_until': self.valid_until.isoformat() if self.valid_until else None,
            'max_hold_time': str(self.max_hold_time) if self.max_hold_time else None,
            'is_valid': self.is_valid,
            'is_actionable': self.is_actionable
        }


@dataclass
class StrategyState:
    """Estado interno de la estrategia"""
    last_signal: Optional[TradingSignal] = None
    last_update: Optional[datetime] = None
    active_positions: List[Dict[str, Any]] = field(default_factory=list)
    signal_history: List[TradingSignal] = field(default_factory=list)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    
    # Contadores
    total_signals: int = 0
    successful_signals: int = 0
    failed_signals: int = 0
    
    # Estado de mercado
    market_conditions: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def success_rate(self) -> float:
        """Calcular tasa de éxito de señales"""
        if self.total_signals == 0:
            return 0.0
        return self.successful_signals / self.total_signals
    
    def update_signal_result(self, success: bool):
        """Actualizar resultado de señal"""
        self.total_signals += 1
        if success:
            self.successful_signals += 1
        else:
            self.failed_signals += 1


@dataclass
class StrategyConfig:
    """Configuración de estrategia"""
    # Identificación
    name: str
    description: str = ""
    version: str = "1.0.0"
    
    # Parámetros generales
    min_confidence_threshold: float = 0.6
    max_positions: int = 3
    signal_expiry_minutes: int = 60
    
    # Risk management
    max_risk_per_trade: float = 0.02  # 2%
    stop_loss_pips: Optional[float] = None
    take_profit_pips: Optional[float] = None
    use_trailing_stop: bool = False
    
    # Filtros
    trading_hours: Optional[Tuple[int, int]] = None  # (start_hour, end_hour)
    trading_days: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])  # Lun-Vie
    min_spread: Optional[float] = None
    max_spread: Optional[float] = None
    
    # Features y datos
    required_features: List[str] = field(default_factory=list)
    lookback_periods: List[int] = field(default_factory=lambda: [20, 50, 100])
    
    # Optimización
    optimize_parameters: bool = True
    backtest_period_days: int = 30
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir configuración a diccionario"""
        return {
            'name': self.name,
            'description': self.description,
            'version': self.version,
            'min_confidence_threshold': self.min_confidence_threshold,
            'max_positions': self.max_positions,
            'signal_expiry_minutes': self.signal_expiry_minutes,
            'max_risk_per_trade': self.max_risk_per_trade,
            'stop_loss_pips': self.stop_loss_pips,
            'take_profit_pips': self.take_profit_pips,
            'use_trailing_stop': self.use_trailing_stop,
            'trading_hours': self.trading_hours,
            'trading_days': self.trading_days,
            'min_spread': self.min_spread,
            'max_spread': self.max_spread,
            'required_features': self.required_features,
            'lookback_periods': self.lookback_periods,
            'optimize_parameters': self.optimize_parameters,
            'backtest_period_days': self.backtest_period_days
        }


class BaseStrategy(abc.ABC):
    """
    Clase base abstracta para todas las estrategias de trading
    
    Proporciona:
    - Interfaz estándar para generación de señales
    - Gestión de estado y configuración
    - Validación de condiciones de mercado
    - Integración con risk management
    - Métricas de performance
    """
    
    def __init__(self, config: StrategyConfig, model_hub: Optional[ModelHub] = None):
        """
        Inicializar estrategia base
        
        Args:
            config: Configuración de la estrategia
            model_hub: Hub de modelos ML (opcional)
        """
        self.config = config
        self.model_hub = model_hub
        self.state = StrategyState()
        
        # Datos y features
        self.current_data: Optional[pd.DataFrame] = None
        self.feature_columns: List[str] = []
        
        # Callbacks opcionales
        self.on_signal_generated: Optional[Callable] = None
        self.on_position_opened: Optional[Callable] = None
        self.on_position_closed: Optional[Callable] = None
        
        # Inicialización
        self._initialize_strategy()
        
        logger.info(f"Estrategia inicializada: {config.name} v{config.version}")
    
    # ==================== MÉTODOS ABSTRACTOS ====================
    
    @abc.abstractmethod
    def _generate_signal_logic(self, data: pd.DataFrame, 
                              market_conditions: Dict[str, Any]) -> TradingSignal:
        """
        Lógica específica para generar señales
        
        Args:
            data: DataFrame con datos de mercado y features
            market_conditions: Condiciones actuales del mercado
            
        Returns:
            Señal de trading generada
        """
        pass
    
    @abc.abstractmethod
    def _validate_signal(self, signal: TradingSignal, 
                        data: pd.DataFrame) -> bool:
        """
        Validar señal antes de emitirla
        
        Args:
            signal: Señal a validar
            data: Datos actuales
            
        Returns:
            True si la señal es válida
        """
        pass
    
    @abc.abstractmethod
    def get_required_features(self) -> List[str]:
        """
        Obtener lista de features requeridas por la estrategia
        
        Returns:
            Lista de nombres de features necesarias
        """
        pass
    
    # ==================== MÉTODOS PÚBLICOS ====================
    
    def generate_signal(self, data: pd.DataFrame, 
                       market_conditions: Dict[str, Any] = None) -> Optional[TradingSignal]:
        """
        Generar señal de trading
        
        Args:
            data: Datos de mercado con features
            market_conditions: Condiciones del mercado
            
        Returns:
            Señal de trading o None si no hay señal
        """
        try:
            # Validar datos de entrada
            if not self._validate_input_data(data):
                return None
            
            # Actualizar datos actuales
            self.current_data = data
            
            # Verificar condiciones de trading
            if not self._check_trading_conditions(data, market_conditions):
                return self._create_no_signal("Condiciones de trading no cumplidas")
            
            # Actualizar estado del mercado
            if market_conditions:
                self.state.market_conditions.update(market_conditions)
            
            # Generar señal usando lógica específica
            signal = self._generate_signal_logic(data, market_conditions or {})
            
            # Enriquecer señal con metadatos
            signal = self._enrich_signal(signal, data)
            
            # Validar señal
            if not self._validate_signal(signal, data):
                logger.debug(f"Señal no válida: {signal.reasoning}")
                return None
            
            # Verificar confianza mínima
            if signal.confidence < self.config.min_confidence_threshold:
                logger.debug(f"Confianza insuficiente: {signal.confidence:.3f} < {self.config.min_confidence_threshold}")
                return None
            
            # Actualizar estado
            self._update_state_with_signal(signal)
            
            # Callback opcional
            if self.on_signal_generated:
                self.on_signal_generated(signal)
            
            # Log señal
            if signal.signal_type != SignalType.NO_SIGNAL:
                logger.info(f"🎯 Señal generada: {signal.signal_type.value} "
                           f"(confianza: {signal.confidence:.2%}, precio: {signal.price})")
                
                log_trade({
                    'action': 'SIGNAL_GENERATED',
                    'signal_type': signal.signal_type.value,
                    'confidence': signal.confidence,
                    'price': signal.price,
                    'strategy': self.config.name
                })
            
            return signal
            
        except Exception as e:
            logger.error(f"Error generando señal: {e}")
            return None
    
    def update_position_status(self, position_id: str, status: str, 
                             profit: float = None, **kwargs):
        """
        Actualizar estado de posición
        
        Args:
            position_id: ID de la posición
            status: 'opened', 'closed', 'modified'
            profit: Profit/Loss de la posición
            **kwargs: Información adicional
        """
        if status == 'opened':
            self.state.active_positions.append({
                'id': position_id,
                'opened_at': datetime.now(),
                'signal': self.state.last_signal.to_dict() if self.state.last_signal else None,
                **kwargs
            })
            
            if self.on_position_opened:
                self.on_position_opened(position_id, kwargs)
        
        elif status == 'closed':
            # Remover de posiciones activas
            self.state.active_positions = [
                pos for pos in self.state.active_positions if pos['id'] != position_id
            ]
            
            # Actualizar estadísticas
            if profit is not None:
                success = profit > 0
                self.state.update_signal_result(success)
                
                # Actualizar métricas de performance
                self._update_performance_metrics(profit, **kwargs)
            
            if self.on_position_closed:
                self.on_position_closed(position_id, profit, kwargs)
    
    def get_strategy_statistics(self) -> Dict[str, Any]:
        """Obtener estadísticas de la estrategia"""
        stats = {
            'config': self.config.to_dict(),
            'state': {
                'total_signals': self.state.total_signals,
                'successful_signals': self.state.successful_signals,
                'failed_signals': self.state.failed_signals,
                'success_rate': self.state.success_rate,
                'active_positions': len(self.state.active_positions),
                'last_signal_time': self.state.last_signal.timestamp.isoformat() if self.state.last_signal else None
            },
            'performance': self.state.performance_metrics.copy()
        }
        
        # Estadísticas de señales por tipo
        signal_types = {}
        for signal in self.state.signal_history[-100:]:  # Últimas 100
            signal_type = signal.signal_type.value
            signal_types[signal_type] = signal_types.get(signal_type, 0) + 1
        
        stats['signal_distribution'] = signal_types
        
        # Estadísticas de confianza
        if self.state.signal_history:
            confidences = [s.confidence for s in self.state.signal_history[-50:]]
            stats['confidence_stats'] = {
                'mean': np.mean(confidences),
                'std': np.std(confidences),
                'min': np.min(confidences),
                'max': np.max(confidences)
            }
        
        return stats
    
    def optimize_parameters(self, data: pd.DataFrame, 
                          optimization_metric: str = 'sharpe_ratio',
                          parameter_ranges: Dict[str, List] = None) -> Dict[str, Any]:
        """
        Optimizar parámetros de la estrategia
        
        Args:
            data: Datos históricos para optimización
            optimization_metric: Métrica a optimizar
            parameter_ranges: Rangos de parámetros a probar
            
        Returns:
            Mejores parámetros encontrados
        """
        if not self.config.optimize_parameters:
            logger.info("Optimización de parámetros deshabilitada")
            return {}
        
        logger.info(f"Optimizando parámetros para {self.config.name}")
        
        # Implementación básica - las subclases pueden sobrescribir
        best_params = {}
        best_score = -float('inf')
        
        # Ejemplo: optimizar threshold de confianza
        if parameter_ranges is None:
            parameter_ranges = {
                'min_confidence_threshold': [0.5, 0.6, 0.7, 0.8]
            }
        
        for param_name, param_values in parameter_ranges.items():
            for value in param_values:
                # Crear configuración temporal
                temp_config = self.config
                setattr(temp_config, param_name, value)
                
                # Simular backtest rápido
                score = self._quick_backtest(data, optimization_metric)
                
                if score > best_score:
                    best_score = score
                    best_params[param_name] = value
        
        # Aplicar mejores parámetros
        for param_name, value in best_params.items():
            setattr(self.config, param_name, value)
            logger.info(f"Parámetro optimizado: {param_name} = {value}")
        
        return best_params
    
    def backtest(self, data: pd.DataFrame, 
                start_date: datetime = None, 
                end_date: datetime = None) -> Dict[str, Any]:
        """
        Ejecutar backtest de la estrategia
        
        Args:
            data: Datos históricos
            start_date: Fecha de inicio
            end_date: Fecha de fin
            
        Returns:
            Resultados del backtest
        """
        logger.info(f"Ejecutando backtest: {self.config.name}")
        
        # Filtrar datos por fechas
        if start_date or end_date:
            mask = pd.Series(True, index=data.index)
            if start_date:
                mask &= data.index >= start_date
            if end_date:
                mask &= data.index <= end_date
            data = data[mask]
        
        # Resultados del backtest
        trades = []
        equity_curve = []
        signals_generated = []
        
        initial_balance = 10000  # Balance inicial para simulación
        current_balance = initial_balance
        position = None
        
        # Simular trading histórico
        for i in range(len(data)):
            current_row = data.iloc[i:i+1]
            current_time = current_row.index[0]
            current_price = current_row['close'].iloc[0]
            
            # Obtener datos hasta el momento actual
            historical_data = data.iloc[:i+1]
            
            if len(historical_data) < max(self.config.lookback_periods, default=20):
                continue
            
            # Generar señal
            signal = self.generate_signal(historical_data)
            
            if signal and signal.is_actionable:
                signals_generated.append({
                    'timestamp': current_time,
                    'signal': signal.to_dict()
                })
                
                # Simular ejecución de trade
                if position is None and signal.signal_type in [SignalType.BUY, SignalType.SELL]:
                    # Abrir posición
                    position = {
                        'type': signal.signal_type.value,
                        'entry_price': current_price,
                        'entry_time': current_time,
                        'stop_loss': signal.stop_loss,
                        'take_profit': signal.take_profit,
                        'size': current_balance * 0.1  # 10% del balance
                    }
                
                elif position and signal.signal_type in [SignalType.CLOSE_LONG, SignalType.CLOSE_SHORT]:
                    # Cerrar posición
                    if position['type'] == 'BUY':
                        pnl = (current_price - position['entry_price']) * position['size'] / position['entry_price']
                    else:
                        pnl = (position['entry_price'] - current_price) * position['size'] / position['entry_price']
                    
                    current_balance += pnl
                    
                    trades.append({
                        'entry_time': position['entry_time'],
                        'exit_time': current_time,
                        'entry_price': position['entry_price'],
                        'exit_price': current_price,
                        'type': position['type'],
                        'pnl': pnl,
                        'size': position['size']
                    })
                    
                    position = None
            
            # Verificar stop loss / take profit
            if position:
                should_close = False
                
                if position['stop_loss'] and ((position['type'] == 'BUY' and current_price <= position['stop_loss']) or
                                            (position['type'] == 'SELL' and current_price >= position['stop_loss'])):
                    should_close = True
                
                if position['take_profit'] and ((position['type'] == 'BUY' and current_price >= position['take_profit']) or
                                              (position['type'] == 'SELL' and current_price <= position['take_profit'])):
                    should_close = True
                
                if should_close:
                    # Cerrar por SL/TP
                    if position['type'] == 'BUY':
                        pnl = (current_price - position['entry_price']) * position['size'] / position['entry_price']
                    else:
                        pnl = (position['entry_price'] - current_price) * position['size'] / position['entry_price']
                    
                    current_balance += pnl
                    
                    trades.append({
                        'entry_time': position['entry_time'],
                        'exit_time': current_time,
                        'entry_price': position['entry_price'],
                        'exit_price': current_price,
                        'type': position['type'],
                        'pnl': pnl,
                        'size': position['size'],
                        'exit_reason': 'SL/TP'
                    })
                    
                    position = None
            
            # Registrar equity
            equity_curve.append({
                'timestamp': current_time,
                'balance': current_balance
            })
        
        # Calcular métricas del backtest
        results = self._calculate_backtest_metrics(
            trades, equity_curve, initial_balance, current_balance
        )
        
        results['trades'] = trades
        results['signals'] = signals_generated
        results['equity_curve'] = equity_curve
        
        logger.info(f"Backtest completado: {len(trades)} trades, "
                   f"Return: {results['total_return']:.2%}")
        
        return results
    
    # ==================== MÉTODOS PROTEGIDOS ====================
    
    def _initialize_strategy(self):
        """Inicialización específica de la estrategia"""
        # Hook para inicialización personalizada
        pass
    
    def _validate_input_data(self, data: pd.DataFrame) -> bool:
        """Validar datos de entrada"""
        if data.empty:
            logger.warning("Datos de entrada vacíos")
            return False
        
        # Verificar features requeridas
        required_features = self.get_required_features()
        missing_features = [f for f in required_features if f not in data.columns]
        
        if missing_features:
            logger.warning(f"Features faltantes: {missing_features}")
            return False
        
        # Verificar suficientes datos históricos
        min_history = max(self.config.lookback_periods, default=20)
        if len(data) < min_history:
            logger.debug(f"Historial insuficiente: {len(data)} < {min_history}")
            return False
        
        return True
    
    def _check_trading_conditions(self, data: pd.DataFrame, 
                                market_conditions: Dict[str, Any] = None) -> bool:
        """Verificar condiciones generales de trading"""
        current_time = datetime.now()
        
        # Verificar horario de trading
        if self.config.trading_hours:
            start_hour, end_hour = self.config.trading_hours
            current_hour = current_time.hour
            
            if not (start_hour <= current_hour < end_hour):
                return False
        
        # Verificar día de trading
        if current_time.weekday() not in self.config.trading_days:
            return False
        
        # Verificar spread si está configurado
        if market_conditions and 'spread' in market_conditions:
            spread = market_conditions['spread']
            
            if self.config.min_spread and spread < self.config.min_spread:
                return False
            
            if self.config.max_spread and spread > self.config.max_spread:
                return False
        
        # Verificar número máximo de posiciones
        if len(self.state.active_positions) >= self.config.max_positions:
            return False
        
        return True
    
    def _create_no_signal(self, reason: str = "") -> TradingSignal:
        """Crear señal de 'no acción'"""
        current_price = self.current_data['close'].iloc[-1] if self.current_data is not None else 0.0
        
        return TradingSignal(
            signal_type=SignalType.NO_SIGNAL,
            confidence=0.0,
            strength=SignalStrength.WEAK,
            price=current_price,
            timestamp=datetime.now(),
            strategy_name=self.config.name,
            reasoning=reason
        )
    
    def _enrich_signal(self, signal: TradingSignal, data: pd.DataFrame) -> TradingSignal:
        """Enriquecer señal con metadatos adicionales"""
        # Información básica
        signal.strategy_name = self.config.name
        signal.timestamp = datetime.now()
        
        # Establecer validez temporal
        if self.config.signal_expiry_minutes:
            signal.valid_until = signal.timestamp + timedelta(minutes=self.config.signal_expiry_minutes)
        
        # Calcular SL/TP si no están establecidos
        if not signal.stop_loss and self.config.stop_loss_pips:
            if signal.signal_type == SignalType.BUY:
                signal.stop_loss = signal.price - (self.config.stop_loss_pips * 0.0001)
            elif signal.signal_type == SignalType.SELL:
                signal.stop_loss = signal.price + (self.config.stop_loss_pips * 0.0001)
        
        if not signal.take_profit and self.config.take_profit_pips:
            if signal.signal_type == SignalType.BUY:
                signal.take_profit = signal.price + (self.config.take_profit_pips * 0.0001)
            elif signal.signal_type == SignalType.SELL:
                signal.take_profit = signal.price - (self.config.take_profit_pips * 0.0001)
        
        # Establecer precio de entrada
        if not signal.entry_price:
            signal.entry_price = signal.price
        
        return signal
    
    def _update_state_with_signal(self, signal: TradingSignal):
        """Actualizar estado con nueva señal"""
        self.state.last_signal = signal
        self.state.last_update = datetime.now()
        
        # Agregar al historial
        self.state.signal_history.append(signal)
        
        # Mantener historial limitado
        if len(self.state.signal_history) > 1000:
            self.state.signal_history = self.state.signal_history[-500:]
    
    def _update_performance_metrics(self, profit: float, **kwargs):
        """Actualizar métricas de performance"""
        # Profit total
        current_profit = self.state.performance_metrics.get('total_profit', 0)
        self.state.performance_metrics['total_profit'] = current_profit + profit
        
        # Número de trades
        total_trades = self.state.performance_metrics.get('total_trades', 0)
        self.state.performance_metrics['total_trades'] = total_trades + 1
        
        # Profit promedio
        self.state.performance_metrics['avg_profit'] = \
            self.state.performance_metrics['total_profit'] / self.state.performance_metrics['total_trades']
        
        # Trades ganadores/perdedores
        if profit > 0:
            winning_trades = self.state.performance_metrics.get('winning_trades', 0)
            self.state.performance_metrics['winning_trades'] = winning_trades + 1
        else:
            losing_trades = self.state.performance_metrics.get('losing_trades', 0)
            self.state.performance_metrics['losing_trades'] = losing_trades + 1
        
        # Win rate
        winning_trades = self.state.performance_metrics.get('winning_trades', 0)
        total_trades = self.state.performance_metrics['total_trades']
        self.state.performance_metrics['win_rate'] = winning_trades / total_trades
    
    def _quick_backtest(self, data: pd.DataFrame, metric: str) -> float:
        """Backtest rápido para optimización"""
        # Implementación simplificada
        signals_count = 0
        correct_signals = 0
        
        for i in range(100, len(data), 10):  # Muestrear cada 10 puntos
            historical_data = data.iloc[:i]
            signal = self.generate_signal(historical_data)
            
            if signal and signal.is_actionable:
                signals_count += 1
                
                # Verificar si la predicción fue correcta (simplificado)
                future_price = data.iloc[min(i+5, len(data)-1)]['close']
                current_price = data.iloc[i]['close']
                
                if signal.signal_type == SignalType.BUY and future_price > current_price:
                    correct_signals += 1
                elif signal.signal_type == SignalType.SELL and future_price < current_price:
                    correct_signals += 1
        
        if signals_count == 0:
            return 0.0
        
        accuracy = correct_signals / signals_count
        
        # Mapear métrica solicitada
        if metric == 'accuracy':
            return accuracy
        elif metric == 'sharpe_ratio':
            return accuracy * 2 - 1  # Conversión simplificada
        else:
            return accuracy
    
    def _calculate_backtest_metrics(self, trades: List[Dict], 
                                  equity_curve: List[Dict],
                                  initial_balance: float,
                                  final_balance: float) -> Dict[str, Any]:
        """Calcular métricas del backtest"""
        if not trades:
            return {
                'total_return': 0.0,
                'total_trades': 0,
                'win_rate': 0.0,
                'avg_profit': 0.0,
                'max_drawdown': 0.0,
                'sharpe_ratio': 0.0
            }
        
        # Métricas básicas
        total_return = (final_balance - initial_balance) / initial_balance
        total_trades = len(trades)
        
        # Análisis de trades
        profits = [trade['pnl'] for trade in trades]
        winning_trades = [p for p in profits if p > 0]
        losing_trades = [p for p in profits if p <= 0]
        
        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
        avg_profit = np.mean(profits) if profits else 0
        
        # Drawdown
        balances = [eq['balance'] for eq in equity_curve]
        peak = balances[0]
        max_drawdown = 0
        
        for balance in balances:
            if balance > peak:
                peak = balance
            
            drawdown = (peak - balance) / peak
            if drawdown > max_drawdown:
                max_drawdown = drawdown
        
        # Sharpe ratio simplificado
        returns = np.diff(balances) / balances[:-1]
        sharpe_ratio = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252) if len(returns) > 1 else 0
        
        return {
            'total_return': total_return,
            'total_trades': total_trades,
            'win_rate': win_rate,
            'avg_profit': avg_profit,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'avg_winner': np.mean(winning_trades) if winning_trades else 0,
            'avg_loser': np.mean(losing_trades) if losing_trades else 0,
            'profit_factor': abs(sum(winning_trades) / sum(losing_trades)) if losing_trades and sum(losing_trades) != 0 else 0
        }


# Funciones de utilidad
def create_strategy_config(name: str, **kwargs) -> StrategyConfig:
    """Crear configuración de estrategia con valores por defecto"""
    return StrategyConfig(name=name, **kwargs)


def validate_strategy_config(config: StrategyConfig) -> List[str]:
    """Validar configuración de estrategia"""
    errors = []
    
    if not config.name:
        errors.append("Nombre de estrategia requerido")
    
    if config.min_confidence_threshold < 0 or config.min_confidence_threshold > 1:
        errors.append("Threshold de confianza debe estar entre 0 y 1")
    
    if config.max_positions <= 0:
        errors.append("Máximo de posiciones debe ser positivo")
    
    if config.max_risk_per_trade <= 0 or config.max_risk_per_trade > 1:
        errors.append("Riesgo por trade debe estar entre 0 y 1")
    
    return errors
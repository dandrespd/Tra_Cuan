'''
16. risk/risk_manager.py
Ruta: TradingBot_Cuantitative_MT5/risk/risk_manager.py
Resumen:

Sistema avanzado de gestión de riesgo que monitorea y controla la exposición del portfolio
Calcula tamaños de posición óptimos, aplica límites dinámicos de drawdown y gestiona concentración
Incluye métricas sofisticadas como VaR, CVaR, correlaciones rolling y límites adaptativos
Integra completamente con el sistema de logging para tracking y alertas automáticas
'''
# risk/risk_manager.py
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Union, Tuple, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

from utils.log_config import get_logger, log_risk_alert, log_performance
from config.trading_config import TradingConfig

# Imports opcionales
try:
    from scipy import stats
    from sklearn.covariance import LedoitWolf
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

logger = get_logger('risk')


class RiskLevel(Enum):
    """Niveles de riesgo"""
    LOW = "LOW"
    MODERATE = "MODERATE"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class RiskViolationType(Enum):
    """Tipos de violaciones de riesgo"""
    MAX_POSITION_SIZE = "MAX_POSITION_SIZE"
    MAX_DRAWDOWN = "MAX_DRAWDOWN"
    MAX_CORRELATION = "MAX_CORRELATION"
    MAX_EXPOSURE = "MAX_EXPOSURE"
    MAX_DAILY_LOSS = "MAX_DAILY_LOSS"
    VAR_LIMIT = "VAR_LIMIT"
    CONCENTRATION_LIMIT = "CONCENTRATION_LIMIT"
    LEVERAGE_LIMIT = "LEVERAGE_LIMIT"


@dataclass
class RiskViolation:
    """Registro de violación de riesgo"""
    violation_type: RiskViolationType
    severity: RiskLevel
    current_value: float
    limit_value: float
    timestamp: datetime
    symbol: Optional[str] = None
    strategy: Optional[str] = None
    description: str = ""
    
    @property
    def excess_percentage(self) -> float:
        """Porcentaje de exceso sobre el límite"""
        if self.limit_value == 0:
            return 0.0
        return (self.current_value - self.limit_value) / self.limit_value
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario"""
        return {
            'violation_type': self.violation_type.value,
            'severity': self.severity.value,
            'current_value': self.current_value,
            'limit_value': self.limit_value,
            'excess_percentage': self.excess_percentage,
            'timestamp': self.timestamp.isoformat(),
            'symbol': self.symbol,
            'strategy': self.strategy,
            'description': self.description
        }


@dataclass
class Position:
    """Información de posición para gestión de riesgo"""
    symbol: str
    size: float
    entry_price: float
    current_price: float
    side: str  # 'LONG' or 'SHORT'
    strategy: str
    entry_time: datetime
    unrealized_pnl: float = 0.0
    
    @property
    def market_value(self) -> float:
        """Valor de mercado de la posición"""
        return abs(self.size * self.current_price)
    
    @property
    def pnl_percentage(self) -> float:
        """P&L como porcentaje"""
        if self.entry_price == 0:
            return 0.0
        
        if self.side == 'LONG':
            return (self.current_price - self.entry_price) / self.entry_price
        else:
            return (self.entry_price - self.current_price) / self.entry_price


@dataclass
class RiskMetrics:
    """Métricas de riesgo del portfolio"""
    total_exposure: float = 0.0
    net_exposure: float = 0.0
    gross_exposure: float = 0.0
    current_drawdown: float = 0.0
    max_drawdown: float = 0.0
    daily_pnl: float = 0.0
    var_95: float = 0.0
    cvar_95: float = 0.0
    sharpe_ratio: float = 0.0
    leverage: float = 0.0
    concentration_risk: float = 0.0
    correlation_risk: float = 0.0
    volatility: float = 0.0
    
    def to_dict(self) -> Dict[str, float]:
        """Convertir a diccionario"""
        return {
            'total_exposure': self.total_exposure,
            'net_exposure': self.net_exposure,
            'gross_exposure': self.gross_exposure,
            'current_drawdown': self.current_drawdown,
            'max_drawdown': self.max_drawdown,
            'daily_pnl': self.daily_pnl,
            'var_95': self.var_95,
            'cvar_95': self.cvar_95,
            'sharpe_ratio': self.sharpe_ratio,
            'leverage': self.leverage,
            'concentration_risk': self.concentration_risk,
            'correlation_risk': self.correlation_risk,
            'volatility': self.volatility
        }


@dataclass
class RiskLimits:
    """Límites de riesgo configurables"""
    max_position_size_pct: float = 0.1  # 10% máximo por posición
    max_drawdown_pct: float = 0.15  # 15% drawdown máximo
    max_daily_loss_pct: float = 0.05  # 5% pérdida diaria máxima
    max_correlation: float = 0.8  # Correlación máxima entre posiciones
    max_exposure_pct: float = 0.95  # 95% exposición máxima
    max_leverage: float = 3.0  # Apalancamiento máximo
    max_concentration_pct: float = 0.3  # 30% concentración máxima por símbolo/estrategia
    var_limit_pct: float = 0.1  # 10% VaR límite
    
    # Límites dinámicos basados en volatilidad
    volatility_adjustment: bool = True
    low_vol_multiplier: float = 1.5  # Aumentar límites en baja volatilidad
    high_vol_multiplier: float = 0.7  # Reducir límites en alta volatilidad
    vol_threshold_low: float = 0.01  # Umbral volatilidad baja
    vol_threshold_high: float = 0.03  # Umbral volatilidad alta


class RiskManager:
    """
    Gestor de riesgo principal para el sistema de trading
    
    Funcionalidades:
    - Cálculo de tamaños de posición basados en riesgo
    - Monitoreo de drawdown y exposición
    - Gestión de correlaciones y concentración
    - Cálculo de VaR y métricas avanzadas
    - Alertas automáticas de violaciones
    """
    
    def __init__(self, config: TradingConfig, risk_limits: Optional[RiskLimits] = None):
        """
        Inicializar gestor de riesgo
        
        Args:
            config: Configuración de trading
            risk_limits: Límites de riesgo personalizados
        """
        self.config = config
        self.risk_limits = risk_limits or RiskLimits()
        
        # Estado del portfolio
        self.positions: Dict[str, Position] = {}
        self.account_balance: float = 10000.0  # Balance inicial
        self.account_equity: float = 10000.0
        self.peak_equity: float = 10000.0
        
        # Historial de P&L y equity
        self.pnl_history: List[Dict[str, Any]] = []
        self.equity_history: List[Dict[str, Any]] = []
        
        # Métricas de riesgo
        self.current_metrics = RiskMetrics()
        self.risk_violations: List[RiskViolation] = []
        
        # Datos de mercado para cálculos
        self.price_history: Dict[str, pd.Series] = {}
        self.correlation_matrix: Optional[pd.DataFrame] = None
        self.volatility_estimates: Dict[str, float] = {}
        
        # Configuración de cálculos
        self.var_confidence_level = 0.95
        self.lookback_periods = {
            'volatility': 20,
            'correlation': 60,
            'var': 252
        }
        
        # Callbacks opcionales
        self.on_risk_violation: Optional[Callable] = None
        self.on_limit_breach: Optional[Callable] = None
        
        logger.info("RiskManager inicializado")
        logger.info(f"Límites configurados: max_drawdown={self.risk_limits.max_drawdown_pct:.1%}")
    
    # ==================== CÁLCULO DE TAMAÑO DE POSICIÓN ====================
    
    def calculate_position_size(self, symbol: str, signal_confidence: float,
                              entry_price: float, stop_loss: Optional[float] = None,
                              strategy: str = "default") -> float:
        """
        Calcular tamaño óptimo de posición basado en riesgo
        
        Args:
            symbol: Símbolo a operar
            signal_confidence: Confianza de la señal (0-1)
            entry_price: Precio de entrada
            stop_loss: Nivel de stop loss
            strategy: Nombre de la estrategia
            
        Returns:
            Tamaño de posición en unidades base
        """
        try:
            # Verificar si podemos operar
            if not self._can_open_position(symbol, strategy):
                logger.debug(f"No se puede abrir posición en {symbol}: límites excedidos")
                return 0.0
            
            # Calcular riesgo base por trade
            base_risk_amount = self.account_equity * self.config.risk_per_trade
            
            # Ajustar por confianza de la señal
            confidence_adjusted_risk = base_risk_amount * signal_confidence
            
            # Calcular tamaño basado en stop loss
            if stop_loss and entry_price:
                risk_per_unit = abs(entry_price - stop_loss)
                size_by_stop = confidence_adjusted_risk / risk_per_unit
            else:
                # Usar volatilidad histórica como proxy
                volatility = self._get_symbol_volatility(symbol)
                estimated_risk_per_unit = entry_price * volatility * 2  # 2 desviaciones estándar
                size_by_stop = confidence_adjusted_risk / estimated_risk_per_unit
            
            # Aplicar límites de concentración
            max_position_value = self.account_equity * self.risk_limits.max_position_size_pct
            max_size_by_concentration = max_position_value / entry_price
            
            # Verificar límites de correlación
            correlation_adjustment = self._calculate_correlation_adjustment(symbol, strategy)
            
            # Tomar el mínimo de todos los límites
            proposed_size = min(
                size_by_stop,
                max_size_by_concentration,
                size_by_stop * correlation_adjustment
            )
            
            # Ajustar por volatilidad de mercado
            volatility_adjustment = self._get_volatility_adjustment()
            final_size = proposed_size * volatility_adjustment
            
            # Redondear al tamaño mínimo permitido
            final_size = self._round_to_min_size(final_size)
            
            logger.info(f"Tamaño calculado para {symbol}: {final_size:.4f}")
            logger.info(f"  Risk base: {base_risk_amount:.2f}")
            logger.info(f"  Confianza: {signal_confidence:.2%}")
            logger.info(f"  Ajuste correlación: {correlation_adjustment:.2f}")
            logger.info(f"  Ajuste volatilidad: {volatility_adjustment:.2f}")
            
            return final_size
            
        except Exception as e:
            logger.error(f"Error calculando tamaño de posición: {e}")
            return 0.0
    
    def _can_open_position(self, symbol: str, strategy: str) -> bool:
        """Verificar si se puede abrir nueva posición"""
        
        # Verificar número máximo de posiciones
        if len(self.positions) >= self.config.max_positions:
            return False
        
        # Verificar drawdown actual
        if self.current_metrics.current_drawdown >= self.risk_limits.max_drawdown_pct:
            return False
        
        # Verificar pérdida diaria
        if self.current_metrics.daily_pnl < -self.risk_limits.max_daily_loss_pct * self.account_equity:
            return False
        
        # Verificar exposición total
        if self.current_metrics.gross_exposure >= self.risk_limits.max_exposure_pct * self.account_equity:
            return False
        
        # Verificar concentración por símbolo
        symbol_exposure = self._calculate_symbol_exposure(symbol)
        if symbol_exposure >= self.risk_limits.max_concentration_pct * self.account_equity:
            return False
        
        # Verificar concentración por estrategia
        strategy_exposure = self._calculate_strategy_exposure(strategy)
        if strategy_exposure >= self.risk_limits.max_concentration_pct * self.account_equity:
            return False
        
        return True
    
    def _calculate_correlation_adjustment(self, symbol: str, strategy: str) -> float:
        """Calcular ajuste por correlación con posiciones existentes"""
        if not self.positions or not SCIPY_AVAILABLE:
            return 1.0
        
        try:
            # Obtener símbolos de posiciones existentes
            existing_symbols = [pos.symbol for pos in self.positions.values()]
            
            if symbol not in existing_symbols:
                # Calcular correlación promedio con posiciones existentes
                correlations = []
                
                for existing_symbol in existing_symbols:
                    corr = self._get_correlation(symbol, existing_symbol)
                    if corr is not None:
                        correlations.append(abs(corr))
                
                if correlations:
                    avg_correlation = np.mean(correlations)
                    
                    # Reducir tamaño si alta correlación
                    if avg_correlation > self.risk_limits.max_correlation:
                        adjustment = 1.0 - (avg_correlation - self.risk_limits.max_correlation)
                        return max(0.1, adjustment)  # Mínimo 10%
            
            return 1.0
            
        except Exception as e:
            logger.warning(f"Error calculando ajuste de correlación: {e}")
            return 1.0
    
    def _get_volatility_adjustment(self) -> float:
        """Calcular ajuste basado en volatilidad de mercado"""
        if not self.risk_limits.volatility_adjustment:
            return 1.0
        
        try:
            # Calcular volatilidad promedio del portfolio
            volatilities = list(self.volatility_estimates.values())
            
            if not volatilities:
                return 1.0
            
            avg_volatility = np.mean(volatilities)
            
            # Ajustar límites según volatilidad
            if avg_volatility < self.risk_limits.vol_threshold_low:
                return self.risk_limits.low_vol_multiplier
            elif avg_volatility > self.risk_limits.vol_threshold_high:
                return self.risk_limits.high_vol_multiplier
            else:
                # Interpolación lineal entre umbrales
                vol_range = self.risk_limits.vol_threshold_high - self.risk_limits.vol_threshold_low
                vol_position = (avg_volatility - self.risk_limits.vol_threshold_low) / vol_range
                
                multiplier_range = self.risk_limits.low_vol_multiplier - self.risk_limits.high_vol_multiplier
                adjustment = self.risk_limits.low_vol_multiplier - (vol_position * multiplier_range)
                
                return adjustment
                
        except Exception as e:
            logger.warning(f"Error calculando ajuste de volatilidad: {e}")
            return 1.0
    
    def _round_to_min_size(self, size: float) -> float:
        """Redondear al tamaño mínimo permitido"""
        min_size = self.config.min_lot_size
        lot_step = self.config.lot_step
        
        # Redondear al step más cercano
        rounded_size = round(size / lot_step) * lot_step
        
        # Aplicar límites
        rounded_size = max(min_size, min(rounded_size, self.config.max_lot_size))
        
        return rounded_size
    
    # ==================== GESTIÓN DE POSICIONES ====================
    
    def add_position(self, position: Position) -> bool:
        """
        Agregar nueva posición al portfolio
        
        Args:
            position: Posición a agregar
            
        Returns:
            True si se agregó exitosamente
        """
        try:
            # Verificar límites antes de agregar
            if not self._can_open_position(position.symbol, position.strategy):
                return False
            
            # Generar ID único para la posición
            position_id = f"{position.symbol}_{position.strategy}_{position.entry_time.timestamp()}"
            
            # Agregar al registro
            self.positions[position_id] = position
            
            # Actualizar métricas
            self._update_risk_metrics()
            
            logger.info(f"Posición agregada: {position_id}")
            logger.info(f"  Símbolo: {position.symbol}")
            logger.info(f"  Tamaño: {position.size}")
            logger.info(f"  Lado: {position.side}")
            logger.info(f"  Estrategia: {position.strategy}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error agregando posición: {e}")
            return False
    
    def remove_position(self, position_id: str, exit_price: float) -> Optional[float]:
        """
        Remover posición del portfolio
        
        Args:
            position_id: ID de la posición
            exit_price: Precio de salida
            
        Returns:
            P&L realizado de la posición
        """
        if position_id not in self.positions:
            logger.warning(f"Posición no encontrada: {position_id}")
            return None
        
        try:
            position = self.positions[position_id]
            
            # Calcular P&L realizado
            if position.side == 'LONG':
                pnl = (exit_price - position.entry_price) * position.size
            else:
                pnl = (position.entry_price - exit_price) * position.size
            
            # Actualizar balance
            self.account_balance += pnl
            self.account_equity = self.account_balance + self._calculate_unrealized_pnl()
            
            # Registrar en historial
            self._record_trade_pnl(position, pnl, exit_price)
            
            # Remover posición
            del self.positions[position_id]
            
            # Actualizar métricas
            self._update_risk_metrics()
            
            logger.info(f"Posición cerrada: {position_id}")
            logger.info(f"  P&L: {pnl:.2f}")
            logger.info(f"  Nuevo balance: {self.account_balance:.2f}")
            
            return pnl
            
        except Exception as e:
            logger.error(f"Error removiendo posición: {e}")
            return None
    
    def update_position_prices(self, price_updates: Dict[str, float]):
        """
        Actualizar precios actuales de posiciones
        
        Args:
            price_updates: Diccionario {símbolo: precio}
        """
        try:
            for position in self.positions.values():
                if position.symbol in price_updates:
                    position.current_price = price_updates[position.symbol]
                    
                    # Recalcular P&L no realizado
                    if position.side == 'LONG':
                        position.unrealized_pnl = (position.current_price - position.entry_price) * position.size
                    else:
                        position.unrealized_pnl = (position.entry_price - position.current_price) * position.size
            
            # Actualizar equity
            self.account_equity = self.account_balance + self._calculate_unrealized_pnl()
            
            # Actualizar peak equity para drawdown
            if self.account_equity > self.peak_equity:
                self.peak_equity = self.account_equity
            
            # Actualizar métricas
            self._update_risk_metrics()
            
            # Verificar violaciones
            self._check_risk_violations()
            
        except Exception as e:
            logger.error(f"Error actualizando precios: {e}")
    
    # ==================== CÁLCULO DE MÉTRICAS DE RIESGO ====================
    
    def _update_risk_metrics(self):
        """Actualizar todas las métricas de riesgo"""
        try:
            # Exposición
            self._calculate_exposure_metrics()
            
            # Drawdown
            self._calculate_drawdown_metrics()
            
            # P&L diario
            self._calculate_daily_pnl()
            
            # VaR y CVaR
            if SCIPY_AVAILABLE:
                self._calculate_var_metrics()
            
            # Concentración y correlación
            self._calculate_concentration_metrics()
            self._calculate_correlation_metrics()
            
            # Leverage
            self._calculate_leverage()
            
            # Volatilidad del portfolio
            self._calculate_portfolio_volatility()
            
            # Sharpe ratio
            self._calculate_sharpe_ratio()
            
        except Exception as e:
            logger.error(f"Error actualizando métricas de riesgo: {e}")
    
    def _calculate_exposure_metrics(self):
        """Calcular métricas de exposición"""
        long_exposure = 0.0
        short_exposure = 0.0
        
        for position in self.positions.values():
            market_value = position.market_value
            
            if position.side == 'LONG':
                long_exposure += market_value
            else:
                short_exposure += market_value
        
        self.current_metrics.gross_exposure = long_exposure + short_exposure
        self.current_metrics.net_exposure = long_exposure - short_exposure
        self.current_metrics.total_exposure = self.current_metrics.gross_exposure
    
    def _calculate_drawdown_metrics(self):
        """Calcular métricas de drawdown"""
        if self.peak_equity > 0:
            self.current_metrics.current_drawdown = (self.peak_equity - self.account_equity) / self.peak_equity
            self.current_metrics.max_drawdown = max(
                self.current_metrics.max_drawdown,
                self.current_metrics.current_drawdown
            )
        else:
            self.current_metrics.current_drawdown = 0.0
    
    def _calculate_daily_pnl(self):
        """Calcular P&L diario"""
        today = datetime.now().date()
        
        # Buscar P&L del día actual
        daily_pnl = 0.0
        
        for record in self.pnl_history:
            if record['date'].date() == today:
                daily_pnl += record['pnl']
        
        # Agregar P&L no realizado
        daily_pnl += self._calculate_unrealized_pnl()
        
        self.current_metrics.daily_pnl = daily_pnl
    
    def _calculate_var_metrics(self):
        """Calcular Value at Risk y Conditional VaR"""
        if not SCIPY_AVAILABLE or len(self.pnl_history) < 30:
            return
        
        try:
            # Obtener retornos recientes
            recent_pnl = [record['pnl'] for record in self.pnl_history[-252:]]  # Último año
            
            if len(recent_pnl) < 10:
                return
            
            returns = np.array(recent_pnl) / self.account_balance
            
            # VaR paramétrico (asumiendo distribución normal)
            var_95 = np.percentile(returns, (1 - self.var_confidence_level) * 100)
            self.current_metrics.var_95 = abs(var_95 * self.account_equity)
            
            # CVaR (Expected Shortfall)
            var_threshold = returns <= var_95
            if np.any(var_threshold):
                cvar_95 = np.mean(returns[var_threshold])
                self.current_metrics.cvar_95 = abs(cvar_95 * self.account_equity)
            
        except Exception as e:
            logger.warning(f"Error calculando VaR: {e}")
    
    def _calculate_concentration_metrics(self):
        """Calcular métricas de concentración"""
        if not self.positions:
            self.current_metrics.concentration_risk = 0.0
            return
        
        # Concentración por símbolo
        symbol_exposures = {}
        strategy_exposures = {}
        
        for position in self.positions.values():
            market_value = position.market_value
            
            # Por símbolo
            if position.symbol not in symbol_exposures:
                symbol_exposures[position.symbol] = 0.0
            symbol_exposures[position.symbol] += market_value
            
            # Por estrategia
            if position.strategy not in strategy_exposures:
                strategy_exposures[position.strategy] = 0.0
            strategy_exposures[position.strategy] += market_value
        
        # Calcular concentración máxima
        max_symbol_concentration = max(symbol_exposures.values()) / self.account_equity if symbol_exposures else 0.0
        max_strategy_concentration = max(strategy_exposures.values()) / self.account_equity if strategy_exposures else 0.0
        
        self.current_metrics.concentration_risk = max(max_symbol_concentration, max_strategy_concentration)
    
    def _calculate_correlation_metrics(self):
        """Calcular métricas de correlación"""
        if len(self.positions) < 2:
            self.current_metrics.correlation_risk = 0.0
            return
        
        try:
            symbols = list(set(pos.symbol for pos in self.positions.values()))
            
            if len(symbols) < 2:
                self.current_metrics.correlation_risk = 0.0
                return
            
            # Actualizar matriz de correlación
            self._update_correlation_matrix(symbols)
            
            if self.correlation_matrix is not None:
                # Correlación promedio ponderada por exposición
                weighted_correlations = []
                
                for i, symbol1 in enumerate(symbols):
                    for j, symbol2 in enumerate(symbols):
                        if i < j:  # Evitar duplicados
                            corr = self.correlation_matrix.loc[symbol1, symbol2]
                            
                            if not np.isnan(corr):
                                # Ponderar por exposición de ambos símbolos
                                weight1 = self._calculate_symbol_exposure(symbol1) / self.account_equity
                                weight2 = self._calculate_symbol_exposure(symbol2) / self.account_equity
                                weighted_corr = abs(corr) * weight1 * weight2
                                weighted_correlations.append(weighted_corr)
                
                if weighted_correlations:
                    self.current_metrics.correlation_risk = np.mean(weighted_correlations)
                else:
                    self.current_metrics.correlation_risk = 0.0
            
        except Exception as e:
            logger.warning(f"Error calculando correlación: {e}")
            self.current_metrics.correlation_risk = 0.0
    
    def _calculate_leverage(self):
        """Calcular apalancamiento actual"""
        if self.account_equity > 0:
            self.current_metrics.leverage = self.current_metrics.gross_exposure / self.account_equity
        else:
            self.current_metrics.leverage = 0.0
    
    def _calculate_portfolio_volatility(self):
        """Calcular volatilidad del portfolio"""
        if len(self.pnl_history) < 20:
            self.current_metrics.volatility = 0.0
            return
        
        try:
            recent_returns = []
            
            for i in range(1, min(21, len(self.pnl_history))):
                pnl1 = self.pnl_history[-i]['pnl']
                pnl2 = self.pnl_history[-i-1]['pnl'] if -i-1 < len(self.pnl_history) else 0
                
                return_pct = (pnl1 - pnl2) / self.account_balance if self.account_balance > 0 else 0
                recent_returns.append(return_pct)
            
            if recent_returns:
                self.current_metrics.volatility = np.std(recent_returns) * np.sqrt(252)  # Anualizada
            
        except Exception as e:
            logger.warning(f"Error calculando volatilidad: {e}")
    
    def _calculate_sharpe_ratio(self):
        """Calcular Sharpe ratio del portfolio"""
        if len(self.pnl_history) < 20:
            self.current_metrics.sharpe_ratio = 0.0
            return
        
        try:
            recent_returns = []
            
            for record in self.pnl_history[-252:]:  # Último año
                return_pct = record['pnl'] / self.account_balance
                recent_returns.append(return_pct)
            
            if len(recent_returns) > 10:
                mean_return = np.mean(recent_returns)
                std_return = np.std(recent_returns)
                
                if std_return > 0:
                    self.current_metrics.sharpe_ratio = (mean_return / std_return) * np.sqrt(252)
            
        except Exception as e:
            logger.warning(f"Error calculando Sharpe ratio: {e}")
    
    # ==================== VERIFICACIÓN DE VIOLACIONES ====================
    
    def _check_risk_violations(self):
        """Verificar violaciones de límites de riesgo"""
        violations = []
        
        # Verificar drawdown
        if self.current_metrics.current_drawdown > self.risk_limits.max_drawdown_pct:
            violations.append(RiskViolation(
                violation_type=RiskViolationType.MAX_DRAWDOWN,
                severity=RiskLevel.CRITICAL,
                current_value=self.current_metrics.current_drawdown,
                limit_value=self.risk_limits.max_drawdown_pct,
                timestamp=datetime.now(),
                description=f"Drawdown excede límite: {self.current_metrics.current_drawdown:.2%}"
            ))
        
        # Verificar pérdida diaria
        daily_loss_pct = abs(self.current_metrics.daily_pnl) / self.account_equity
        if self.current_metrics.daily_pnl < 0 and daily_loss_pct > self.risk_limits.max_daily_loss_pct:
            violations.append(RiskViolation(
                violation_type=RiskViolationType.MAX_DAILY_LOSS,
                severity=RiskLevel.HIGH,
                current_value=daily_loss_pct,
                limit_value=self.risk_limits.max_daily_loss_pct,
                timestamp=datetime.now(),
                description=f"Pérdida diaria excede límite: {daily_loss_pct:.2%}"
            ))
        
        # Verificar exposición
        exposure_pct = self.current_metrics.gross_exposure / self.account_equity
        if exposure_pct > self.risk_limits.max_exposure_pct:
            violations.append(RiskViolation(
                violation_type=RiskViolationType.MAX_EXPOSURE,
                severity=RiskLevel.MODERATE,
                current_value=exposure_pct,
                limit_value=self.risk_limits.max_exposure_pct,
                timestamp=datetime.now(),
                description=f"Exposición excede límite: {exposure_pct:.2%}"
            ))
        
        # Verificar apalancamiento
        if self.current_metrics.leverage > self.risk_limits.max_leverage:
            violations.append(RiskViolation(
                violation_type=RiskViolationType.LEVERAGE_LIMIT,
                severity=RiskLevel.HIGH,
                current_value=self.current_metrics.leverage,
                limit_value=self.risk_limits.max_leverage,
                timestamp=datetime.now(),
                description=f"Apalancamiento excede límite: {self.current_metrics.leverage:.2f}x"
            ))
        
        # Verificar concentración
        if self.current_metrics.concentration_risk > self.risk_limits.max_concentration_pct:
            violations.append(RiskViolation(
                violation_type=RiskViolationType.CONCENTRATION_LIMIT,
                severity=RiskLevel.MODERATE,
                current_value=self.current_metrics.concentration_risk,
                limit_value=self.risk_limits.max_concentration_pct,
                timestamp=datetime.now(),
                description=f"Concentración excede límite: {self.current_metrics.concentration_risk:.2%}"
            ))
        
        # Verificar VaR
        if self.current_metrics.var_95 > 0:
            var_pct = self.current_metrics.var_95 / self.account_equity
            if var_pct > self.risk_limits.var_limit_pct:
                violations.append(RiskViolation(
                    violation_type=RiskViolationType.VAR_LIMIT,
                    severity=RiskLevel.MODERATE,
                    current_value=var_pct,
                    limit_value=self.risk_limits.var_limit_pct,
                    timestamp=datetime.now(),
                    description=f"VaR excede límite: {var_pct:.2%}"
                ))
        
        # Procesar violaciones
        for violation in violations:
            self._handle_risk_violation(violation)
    
    def _handle_risk_violation(self, violation: RiskViolation):
        """Manejar violación de riesgo"""
        # Agregar al registro
        self.risk_violations.append(violation)
        
        # Mantener historial limitado
        if len(self.risk_violations) > 100:
            self.risk_violations = self.risk_violations[-50:]
        
        # Log según severidad
        if violation.severity == RiskLevel.CRITICAL:
            log_risk_alert(
                f"VIOLACIÓN CRÍTICA: {violation.violation_type.value}",
                violation.description,
                violation.to_dict()
            )
        elif violation.severity == RiskLevel.HIGH:
            logger.error(f"🚨 Violación HIGH: {violation.description}")
        else:
            logger.warning(f"⚠️ Violación {violation.severity.value}: {violation.description}")
        
        # Callback opcional
        if self.on_risk_violation:
            self.on_risk_violation(violation)
        
        # Log performance para tracking
        log_performance({
            'risk_violation': True,
            'violation_type': violation.violation_type.value,
            'severity': violation.severity.value,
            'excess_percentage': violation.excess_percentage
        })
    
    # ==================== MÉTODOS AUXILIARES ====================
    
    def _calculate_unrealized_pnl(self) -> float:
        """Calcular P&L no realizado total"""
        return sum(pos.unrealized_pnl for pos in self.positions.values())
    
    def _calculate_symbol_exposure(self, symbol: str) -> float:
        """Calcular exposición total para un símbolo"""
        return sum(
            pos.market_value for pos in self.positions.values() 
            if pos.symbol == symbol
        )
    
    def _calculate_strategy_exposure(self, strategy: str) -> float:
        """Calcular exposición total para una estrategia"""
        return sum(
            pos.market_value for pos in self.positions.values() 
            if pos.strategy == strategy
        )
    
    def _get_symbol_volatility(self, symbol: str) -> float:
        """Obtener volatilidad estimada para un símbolo"""
        if symbol in self.volatility_estimates:
            return self.volatility_estimates[symbol]
        
        # Volatilidad por defecto
        return 0.02  # 2% diario
    
    def _get_correlation(self, symbol1: str, symbol2: str) -> Optional[float]:
        """Obtener correlación entre dos símbolos"""
        if (self.correlation_matrix is not None and 
            symbol1 in self.correlation_matrix.index and 
            symbol2 in self.correlation_matrix.columns):
            return self.correlation_matrix.loc[symbol1, symbol2]
        
        return None
    
    def _update_correlation_matrix(self, symbols: List[str]):
        """Actualizar matriz de correlación"""
        try:
            if len(symbols) < 2:
                return
            
            # Construir matriz de retornos
            returns_data = {}
            
            for symbol in symbols:
                if symbol in self.price_history:
                    prices = self.price_history[symbol]
                    if len(prices) > self.lookback_periods['correlation']:
                        returns = prices.pct_change().dropna()
                        returns_data[symbol] = returns.tail(self.lookback_periods['correlation'])
            
            if len(returns_data) >= 2:
                # Crear DataFrame de retornos
                returns_df = pd.DataFrame(returns_data)
                
                # Calcular matriz de correlación
                self.correlation_matrix = returns_df.corr()
                
        except Exception as e:
            logger.warning(f"Error actualizando matriz de correlación: {e}")
    
    def _record_trade_pnl(self, position: Position, pnl: float, exit_price: float):
        """Registrar P&L de trade cerrado"""
        trade_record = {
            'date': datetime.now(),
            'symbol': position.symbol,
            'strategy': position.strategy,
            'side': position.side,
            'size': position.size,
            'entry_price': position.entry_price,
            'exit_price': exit_price,
            'pnl': pnl,
            'pnl_pct': pnl / (position.entry_price * position.size) if position.entry_price > 0 else 0,
            'hold_time': datetime.now() - position.entry_time
        }
        
        self.pnl_history.append(trade_record)
        
        # Mantener historial limitado
        if len(self.pnl_history) > 1000:
            self.pnl_history = self.pnl_history[-500:]
    
    # ==================== MÉTODOS PÚBLICOS ====================
    
    def update_market_data(self, symbol: str, prices: pd.Series):
        """
        Actualizar datos de mercado para cálculos de riesgo
        
        Args:
            symbol: Símbolo
            prices: Serie de precios históricos
        """
        self.price_history[symbol] = prices
        
        # Actualizar volatilidad
        if len(prices) > self.lookback_periods['volatility']:
            returns = prices.pct_change().dropna()
            volatility = returns.tail(self.lookback_periods['volatility']).std() * np.sqrt(252)
            self.volatility_estimates[symbol] = volatility
    
    def get_risk_metrics(self) -> RiskMetrics:
        """Obtener métricas actuales de riesgo"""
        return self.current_metrics
    
    def get_position_summary(self) -> Dict[str, Any]:
        """Obtener resumen de posiciones"""
        if not self.positions:
            return {
                'total_positions': 0,
                'total_exposure': 0.0,
                'unrealized_pnl': 0.0
            }
        
        summary = {
            'total_positions': len(self.positions),
            'total_exposure': self.current_metrics.gross_exposure,
            'net_exposure': self.current_metrics.net_exposure,
            'unrealized_pnl': self._calculate_unrealized_pnl(),
            'by_symbol': {},
            'by_strategy': {},
            'by_side': {'LONG': 0, 'SHORT': 0}
        }
        
        # Agrupar por símbolo
        for pos in self.positions.values():
            if pos.symbol not in summary['by_symbol']:
                summary['by_symbol'][pos.symbol] = {
                    'count': 0,
                    'exposure': 0.0,
                    'unrealized_pnl': 0.0
                }
            
            summary['by_symbol'][pos.symbol]['count'] += 1
            summary['by_symbol'][pos.symbol]['exposure'] += pos.market_value
            summary['by_symbol'][pos.symbol]['unrealized_pnl'] += pos.unrealized_pnl
            
            # Por estrategia
            if pos.strategy not in summary['by_strategy']:
                summary['by_strategy'][pos.strategy] = {
                    'count': 0,
                    'exposure': 0.0,
                    'unrealized_pnl': 0.0
                }
            
            summary['by_strategy'][pos.strategy]['count'] += 1
            summary['by_strategy'][pos.strategy]['exposure'] += pos.market_value
            summary['by_strategy'][pos.strategy]['unrealized_pnl'] += pos.unrealized_pnl
            
            # Por lado
            summary['by_side'][pos.side] += 1
        
        return summary
    
    def get_recent_violations(self, hours: int = 24) -> List[RiskViolation]:
        """Obtener violaciones recientes"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [
            v for v in self.risk_violations 
            if v.timestamp >= cutoff_time
        ]
    
    def export_risk_report(self) -> Dict[str, Any]:
        """Exportar reporte completo de riesgo"""
        return {
            'timestamp': datetime.now().isoformat(),
            'account_info': {
                'balance': self.account_balance,
                'equity': self.account_equity,
                'peak_equity': self.peak_equity
            },
            'risk_metrics': self.current_metrics.to_dict(),
            'risk_limits': self.risk_limits.__dict__,
            'positions': self.get_position_summary(),
            'recent_violations': [v.to_dict() for v in self.get_recent_violations()],
            'performance_stats': {
                'total_trades': len(self.pnl_history),
                'avg_daily_pnl': np.mean([r['pnl'] for r in self.pnl_history[-30:]]) if self.pnl_history else 0,
                'win_rate': len([r for r in self.pnl_history if r['pnl'] > 0]) / len(self.pnl_history) if self.pnl_history else 0
            }
        }
    
    def reset_daily_metrics(self):
        """Resetear métricas diarias (llamar al inicio de cada día)"""
        self.current_metrics.daily_pnl = 0.0
        
        # Limpiar violaciones antiguas
        cutoff_time = datetime.now() - timedelta(days=7)
        self.risk_violations = [
            v for v in self.risk_violations 
            if v.timestamp >= cutoff_time
        ]
        
        logger.info("Métricas diarias reseteadas")


# Funciones de utilidad
def create_default_risk_manager(config: TradingConfig) -> RiskManager:
    """Crear gestor de riesgo con configuración por defecto"""
    return RiskManager(config)


def create_conservative_risk_manager(config: TradingConfig) -> RiskManager:
    """Crear gestor de riesgo conservador"""
    conservative_limits = RiskLimits(
        max_position_size_pct=0.05,  # 5% máximo por posición
        max_drawdown_pct=0.10,       # 10% drawdown máximo
        max_daily_loss_pct=0.03,     # 3% pérdida diaria máxima
        max_exposure_pct=0.80,       # 80% exposición máxima
        max_leverage=2.0,            # Apalancamiento máximo 2x
        max_concentration_pct=0.20   # 20% concentración máxima
    )
    
    return RiskManager(config, conservative_limits)


def create_aggressive_risk_manager(config: TradingConfig) -> RiskManager:
    """Crear gestor de riesgo agresivo"""
    aggressive_limits = RiskLimits(
        max_position_size_pct=0.20,  # 20% máximo por posición
        max_drawdown_pct=0.25,       # 25% drawdown máximo
        max_daily_loss_pct=0.10,     # 10% pérdida diaria máxima
        max_exposure_pct=1.50,       # 150% exposición máxima
        max_leverage=5.0,            # Apalancamiento máximo 5x
        max_concentration_pct=0.50   # 50% concentración máxima
    )
    
    return RiskManager(config, aggressive_limits)
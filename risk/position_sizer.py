import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple, List, Any
from dataclasses import dataclass, field
from scipy import optimize
from scipy.stats import norm
import MetaTrader5 as mt5

from config.trading_config import TradingConfig
from config.mt5_config import MT5Symbol
from utils.log_config import get_logger
from utils.helpers import calculate_pip_value, normalize_price
from analysis.market_analyzer import MarketConditions
from core.mt5_connector import MT5Connector

@dataclass
class PositionSizeResult:
    """Resultado del cálculo de tamaño de posición"""
    volume: float  # Tamaño final en lotes
    risk_amount: float  # Riesgo monetario de la posición
    risk_percentage: float  # Porcentaje del capital en riesgo
    method_used: str  # Método de sizing utilizado
    adjustments_applied: List[str]  # Ajustes aplicados
    confidence_level: float  # Nivel de confianza en el cálculo
    max_allowed_volume: float  # Volumen máximo permitido
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class RiskParameters:
    """Parámetros de riesgo para el cálculo"""
    account_balance: float
    account_equity: float
    account_free_margin: float
    max_risk_per_trade: float  # Porcentaje
    max_risk_total: float  # Porcentaje total en riesgo
    current_positions: List[Dict[str, Any]]
    correlation_matrix: Optional[pd.DataFrame] = None
    historical_performance: Optional[Dict[str, float]] = None

class PositionSizer:
    """Calculador principal de tamaño de posiciones"""
    
    def __init__(self, config: TradingConfig, mt5_connector: MT5Connector):
        self.config = config
        self.mt5_connector = mt5_connector
        self.logger = get_logger(__name__)
        
        # Métodos de sizing disponibles
        self.sizing_methods = {
            'fixed_risk': self._fixed_risk_sizing,
            'kelly_criterion': self._kelly_criterion_sizing,
            'optimal_f': self._optimal_f_sizing,
            'volatility_based': self._volatility_based_sizing,
            'correlation_adjusted': self._correlation_adjusted_sizing,
            'dynamic_risk': self._dynamic_risk_sizing
        }
        
        # Configuración de métodos
        self.method_priority = ['dynamic_risk', 'correlation_adjusted', 'kelly_criterion']
        self.min_position_size = 0.01  # Tamaño mínimo en lotes
        
        # Cache de cálculos
        self.calculation_cache = {}
        
        # Validadores
        self.validators = PositionSizeValidators()
        
    def calculate_position_size(self, symbol: str, 
                              entry_price: float,
                              stop_loss_price: float,
                              strategy_confidence: float = 1.0,
                              market_conditions: Optional[MarketConditions] = None,
                              method: str = 'auto') -> PositionSizeResult:
        """Calcula el tamaño óptimo de la posición"""
        
        # Obtener información del símbolo
        symbol_info = self._get_symbol_info(symbol)
        if not symbol_info:
            return self._create_error_result("Symbol info not available")
        
        # Obtener parámetros de riesgo actuales
        risk_params = self._get_risk_parameters()
        
        # Calcular distancia del stop loss
        sl_distance = abs(entry_price - stop_loss_price)
        sl_pips = self._calculate_pips(sl_distance, symbol_info)
        
        # Seleccionar método de sizing
        if method == 'auto':
            selected_method = self._select_best_method(
                risk_params, market_conditions, strategy_confidence
            )
        else:
            selected_method = method
        
        # Calcular tamaño base
        if selected_method in self.sizing_methods:
            base_result = self.sizing_methods[selected_method](
                symbol_info, risk_params, sl_pips, strategy_confidence
            )
        else:
            base_result = self._fixed_risk_sizing(
                symbol_info, risk_params, sl_pips, strategy_confidence
            )
        
        # Aplicar ajustes
        adjusted_result = self._apply_adjustments(
            base_result, symbol_info, risk_params, 
            market_conditions, strategy_confidence
        )
        
        # Validar resultado final
        final_result = self._validate_and_finalize(
            adjusted_result, symbol_info, risk_params
        )
        
        # Registrar cálculo
        self._log_calculation(symbol, final_result)
        
        return final_result
    
    def _fixed_risk_sizing(self, symbol_info: MT5Symbol, 
                          risk_params: RiskParameters,
                          sl_pips: float,
                          confidence: float) -> PositionSizeResult:
        """Método de riesgo fijo por operación"""
        
        # Calcular riesgo monetario
        risk_percentage = risk_params.max_risk_per_trade * confidence
        risk_amount = risk_params.account_balance * (risk_percentage / 100)
        
        # Calcular valor del pip
        pip_value = calculate_pip_value(
            symbol_info.name, 
            1.0,  # 1 lote estándar
            risk_params.account_balance
        )
        
        # Calcular volumen
        if pip_value > 0 and sl_pips > 0:
            volume = risk_amount / (pip_value * sl_pips)
        else:
            volume = self.min_position_size
        
        # Normalizar al step del símbolo
        volume = self._normalize_volume(volume, symbol_info)
        
        return PositionSizeResult(
            volume=volume,
            risk_amount=risk_amount,
            risk_percentage=risk_percentage,
            method_used='fixed_risk',
            adjustments_applied=[],
            confidence_level=confidence,
            max_allowed_volume=symbol_info.volume_max,
            metadata={
                'pip_value': pip_value,
                'sl_pips': sl_pips
            }
        )
    
    def _kelly_criterion_sizing(self, symbol_info: MT5Symbol,
                               risk_params: RiskParameters,
                               sl_pips: float,
                               confidence: float) -> PositionSizeResult:
        """Sizing usando Kelly Criterion"""
        
        # Necesitamos estadísticas históricas
        if not risk_params.historical_performance:
            # Fallback a fixed risk
            return self._fixed_risk_sizing(
                symbol_info, risk_params, sl_pips, confidence
            )
        
        perf = risk_params.historical_performance
        
        # Calcular Kelly percentage
        win_rate = perf.get('win_rate', 0.5)
        avg_win = perf.get('avg_win_pips', 50)
        avg_loss = perf.get('avg_loss_pips', 30)
        
        if avg_loss == 0:
            kelly_pct = 0
        else:
            # Kelly formula: f = (p*b - q) / b
            # donde p = probabilidad de ganar, q = 1-p, b = ratio win/loss
            b = avg_win / avg_loss
            p = win_rate
            q = 1 - p
            
            kelly_pct = (p * b - q) / b
        
        # Aplicar fracción de Kelly (típicamente 25%)
        kelly_fraction = 0.25
        adjusted_kelly = kelly_pct * kelly_fraction * confidence
        
        # Limitar Kelly
        max_kelly = 0.02  # Máximo 2% por trade
        final_kelly = min(max(adjusted_kelly, 0), max_kelly)
        
        # Calcular volumen
        risk_amount = risk_params.account_balance * final_kelly
        pip_value = calculate_pip_value(
            symbol_info.name, 1.0, risk_params.account_balance
        )
        
        if pip_value > 0 and sl_pips > 0:
            volume = risk_amount / (pip_value * sl_pips)
        else:
            volume = self.min_position_size
        
        volume = self._normalize_volume(volume, symbol_info)
        
        return PositionSizeResult(
            volume=volume,
            risk_amount=risk_amount,
            risk_percentage=final_kelly * 100,
            method_used='kelly_criterion',
            adjustments_applied=['kelly_fraction'],
            confidence_level=confidence,
            max_allowed_volume=symbol_info.volume_max,
            metadata={
                'kelly_percentage': kelly_pct,
                'adjusted_kelly': final_kelly,
                'win_rate': win_rate,
                'win_loss_ratio': b if avg_loss > 0 else 0
            }
        )
    
    def _volatility_based_sizing(self, symbol_info: MT5Symbol,
                                risk_params: RiskParameters,
                                sl_pips: float,
                                confidence: float) -> PositionSizeResult:
        """Sizing basado en volatilidad del mercado"""
        
        # Obtener datos históricos para calcular volatilidad
        rates = self.mt5_connector.get_historical_data(
            symbol_info.name, mt5.TIMEFRAME_H1, 100
        )
        
        if rates is None or len(rates) < 20:
            return self._fixed_risk_sizing(
                symbol_info, risk_params, sl_pips, confidence
            )
        
        # Calcular volatilidad (ATR)
        df = pd.DataFrame(rates)
        df['atr'] = self._calculate_atr(df, period=14)
        current_atr = df['atr'].iloc[-1]
        avg_atr = df['atr'].mean()
        
        # Ratio de volatilidad
        vol_ratio = current_atr / avg_atr if avg_atr > 0 else 1.0
        
        # Ajustar riesgo según volatilidad
        # Mayor volatilidad = menor posición
        if vol_ratio > 1.5:
            risk_multiplier = 0.5
        elif vol_ratio > 1.2:
            risk_multiplier = 0.7
        elif vol_ratio < 0.8:
            risk_multiplier = 1.3
        else:
            risk_multiplier = 1.0
        
        # Calcular tamaño ajustado
        adjusted_risk_pct = risk_params.max_risk_per_trade * risk_multiplier * confidence
        risk_amount = risk_params.account_balance * (adjusted_risk_pct / 100)
        
        pip_value = calculate_pip_value(
            symbol_info.name, 1.0, risk_params.account_balance
        )
        
        if pip_value > 0 and sl_pips > 0:
            volume = risk_amount / (pip_value * sl_pips)
        else:
            volume = self.min_position_size
        
        volume = self._normalize_volume(volume, symbol_info)
        
        return PositionSizeResult(
            volume=volume,
            risk_amount=risk_amount,
            risk_percentage=adjusted_risk_pct,
            method_used='volatility_based',
            adjustments_applied=['volatility_adjustment'],
            confidence_level=confidence,
            max_allowed_volume=symbol_info.volume_max,
            metadata={
                'volatility_ratio': vol_ratio,
                'risk_multiplier': risk_multiplier,
                'current_atr': current_atr
            }
        )
    
    def _correlation_adjusted_sizing(self, symbol_info: MT5Symbol,
                                   risk_params: RiskParameters,
                                   sl_pips: float,
                                   confidence: float) -> PositionSizeResult:
        """Sizing ajustado por correlaciones con posiciones existentes"""
        
        # Si no hay posiciones abiertas, usar método base
        if not risk_params.current_positions:
            return self._volatility_based_sizing(
                symbol_info, risk_params, sl_pips, confidence
            )
        
        # Calcular exposición actual
        current_exposure = self._calculate_current_exposure(
            risk_params.current_positions, risk_params.account_balance
        )
        
        # Calcular correlación con posiciones existentes
        avg_correlation = self._calculate_average_correlation(
            symbol_info.name, risk_params.current_positions,
            risk_params.correlation_matrix
        )
        
        # Factor de ajuste por correlación
        # Alta correlación = reducir tamaño
        if avg_correlation > 0.7:
            correlation_factor = 0.5
        elif avg_correlation > 0.5:
            correlation_factor = 0.7
        elif avg_correlation < -0.5:
            correlation_factor = 1.2  # Correlación negativa puede aumentar
        else:
            correlation_factor = 1.0
        
        # Factor de ajuste por exposición total
        remaining_risk = risk_params.max_risk_total - current_exposure
        if remaining_risk <= 0:
            return self._create_error_result("Maximum total risk reached")
        
        exposure_factor = min(1.0, remaining_risk / risk_params.max_risk_per_trade)
        
        # Combinar factores
        total_adjustment = correlation_factor * exposure_factor * confidence
        
        # Calcular tamaño
        adjusted_risk_pct = risk_params.max_risk_per_trade * total_adjustment
        risk_amount = risk_params.account_balance * (adjusted_risk_pct / 100)
        
        pip_value = calculate_pip_value(
            symbol_info.name, 1.0, risk_params.account_balance
        )
        
        if pip_value > 0 and sl_pips > 0:
            volume = risk_amount / (pip_value * sl_pips)
        else:
            volume = self.min_position_size
        
        volume = self._normalize_volume(volume, symbol_info)
        
        return PositionSizeResult(
            volume=volume,
            risk_amount=risk_amount,
            risk_percentage=adjusted_risk_pct,
            method_used='correlation_adjusted',
            adjustments_applied=['correlation', 'exposure_limit'],
            confidence_level=confidence,
            max_allowed_volume=symbol_info.volume_max,
            metadata={
                'avg_correlation': avg_correlation,
                'correlation_factor': correlation_factor,
                'exposure_factor': exposure_factor,
                'current_total_exposure': current_exposure
            }
        )
    
    def _dynamic_risk_sizing(self, symbol_info: MT5Symbol,
                           risk_params: RiskParameters,
                           sl_pips: float,
                           confidence: float) -> PositionSizeResult:
        """Sizing dinámico que combina múltiples factores"""
        
        # Factores a considerar
        factors = {
            'base_risk': risk_params.max_risk_per_trade,
            'confidence': confidence,
            'drawdown': self._get_drawdown_factor(risk_params),
            'win_streak': self._get_win_streak_factor(risk_params),
            'time_of_day': self._get_time_factor(),
            'margin_usage': self._get_margin_factor(risk_params)
        }
        
        # Calcular factor combinado
        combined_factor = 1.0
        for factor_name, factor_value in factors.items():
            combined_factor *= factor_value
        
        # Limitar factor combinado
        combined_factor = max(0.2, min(combined_factor, 2.0))
        
        # Calcular riesgo ajustado
        dynamic_risk_pct = risk_params.max_risk_per_trade * combined_factor
        
        # Aplicar límites absolutos
        min_risk = 0.1  # 0.1%
        max_risk = 3.0  # 3%
        dynamic_risk_pct = max(min_risk, min(dynamic_risk_pct, max_risk))
        
        # Calcular volumen
        risk_amount = risk_params.account_balance * (dynamic_risk_pct / 100)
        pip_value = calculate_pip_value(
            symbol_info.name, 1.0, risk_params.account_balance
        )
        
        if pip_value > 0 and sl_pips > 0:
            volume = risk_amount / (pip_value * sl_pips)
        else:
            volume = self.min_position_size
        
        volume = self._normalize_volume(volume, symbol_info)
        
        return PositionSizeResult(
            volume=volume,
            risk_amount=risk_amount,
            risk_percentage=dynamic_risk_pct,
            method_used='dynamic_risk',
            adjustments_applied=list(factors.keys()),
            confidence_level=confidence,
            max_allowed_volume=symbol_info.volume_max,
            metadata={
                'factors': factors,
                'combined_factor': combined_factor
            }
        )
    
    def _apply_adjustments(self, base_result: PositionSizeResult,
                          symbol_info: MT5Symbol,
                          risk_params: RiskParameters,
                          market_conditions: Optional[MarketConditions],
                          confidence: float) -> PositionSizeResult:
        """Aplica ajustes adicionales al tamaño calculado"""
        
        adjusted_volume = base_result.volume
        adjustments = base_result.adjustments_applied.copy()
        
        # Ajuste por condiciones de mercado
        if market_conditions:
            if market_conditions.volatility == 'extreme':
                adjusted_volume *= 0.5
                adjustments.append('extreme_volatility')
            elif market_conditions.volatility == 'high':
                adjusted_volume *= 0.7
                adjustments.append('high_volatility')
            
            if market_conditions.liquidity == 'low':
                adjusted_volume *= 0.6
                adjustments.append('low_liquidity')
        
        # Ajuste por horario
        current_hour = pd.Timestamp.now().hour
        if current_hour < 8 or current_hour > 20:
            adjusted_volume *= 0.8
            adjustments.append('off_peak_hours')
        
        # Ajuste por número de posiciones abiertas
        num_positions = len(risk_params.current_positions)
        if num_positions >= 5:
            adjusted_volume *= 0.5
            adjustments.append('many_positions')
        elif num_positions >= 3:
            adjusted_volume *= 0.7
            adjustments.append('multiple_positions')
        
        # Normalizar volumen final
        adjusted_volume = self._normalize_volume(adjusted_volume, symbol_info)
        
        # Actualizar resultado
        base_result.volume = adjusted_volume
        base_result.adjustments_applied = adjustments
        
        # Recalcular riesgo con volumen ajustado
        pip_value = calculate_pip_value(
            symbol_info.name, adjusted_volume, risk_params.account_balance
        )
        base_result.risk_amount = pip_value * base_result.metadata.get('sl_pips', 0)
        base_result.risk_percentage = (
            base_result.risk_amount / risk_params.account_balance * 100
        )
        
        return base_result

class PositionSizeOptimizer:
    """Optimizador avanzado de tamaño de posiciones"""
    
    def __init__(self):
        self.optimization_methods = {
            'mean_variance': self._mean_variance_optimization,
            'cvar': self._cvar_optimization,
            'kelly_multi': self._multi_kelly_optimization
        }
        
    def optimize_portfolio_sizes(self, opportunities: List[Dict[str, Any]],
                               risk_params: RiskParameters,
                               constraints: Dict[str, Any]) -> Dict[str, float]:
        """Optimiza tamaños para múltiples oportunidades simultáneas"""
        
        if len(opportunities) == 1:
            # Solo una oportunidad, no hay que optimizar portfolio
            return {opportunities[0]['symbol']: 1.0}
        
        # Preparar datos para optimización
        expected_returns = np.array([opp['expected_return'] for opp in opportunities])
        covariance_matrix = self._build_covariance_matrix(opportunities)
        
        # Optimizar según método seleccionado
        method = constraints.get('optimization_method', 'mean_variance')
        
        if method in self.optimization_methods:
            weights = self.optimization_methods[method](
                expected_returns, covariance_matrix, constraints
            )
        else:
            # Distribución uniforme como fallback
            weights = np.ones(len(opportunities)) / len(opportunities)
        
        # Convertir a diccionario
        size_allocation = {}
        for i, opp in enumerate(opportunities):
            size_allocation[opp['symbol']] = weights[i]
        
        return size_allocation
    
    def _mean_variance_optimization(self, returns: np.ndarray,
                                  cov_matrix: np.ndarray,
                                  constraints: Dict[str, Any]) -> np.ndarray:
        """Optimización media-varianza (Markowitz)"""
        
        n_assets = len(returns)
        
        # Función objetivo: minimizar varianza del portfolio
        def portfolio_variance(weights):
            return weights.T @ cov_matrix @ weights
        
        # Restricciones
        constraints_list = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}  # Suma = 1
        ]
        
        # Límites para cada peso
        bounds = tuple((0, constraints.get('max_weight', 0.5)) for _ in range(n_assets))
        
        # Peso inicial
        x0 = np.ones(n_assets) / n_assets
        
        # Optimizar
        result = optimize.minimize(
            portfolio_variance,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints_list
        )
        
        if result.success:
            return result.x
        else:
            return x0

class RiskAllocationSystem:
    """Sistema de asignación de riesgo entre múltiples estrategias"""
    
    def __init__(self, total_risk_budget: float):
        self.total_risk_budget = total_risk_budget
        self.strategy_allocations = {}
        self.performance_tracker = {}
        
    def allocate_risk(self, active_strategies: List[str],
                     performance_data: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """Asigna presupuesto de riesgo a cada estrategia"""
        
        if not active_strategies:
            return {}
        
        # Calcular scores para cada estrategia
        strategy_scores = {}
        
        for strategy in active_strategies:
            perf = performance_data.get(strategy, {})
            
            # Score basado en Sharpe ratio y consistencia
            sharpe = perf.get('sharpe_ratio', 0)
            consistency = perf.get('consistency_score', 0.5)
            drawdown = perf.get('max_drawdown', 0.2)
            
            # Penalizar por drawdown alto
            drawdown_penalty = max(0, 1 - drawdown / 0.2)
            
            score = sharpe * consistency * drawdown_penalty
            strategy_scores[strategy] = max(score, 0.1)  # Score mínimo
        
        # Normalizar scores a allocations
        total_score = sum(strategy_scores.values())
        
        allocations = {}
        for strategy, score in strategy_scores.items():
            allocation = (score / total_score) * self.total_risk_budget
            allocations[strategy] = allocation
        
        # Aplicar límites
        for strategy in allocations:
            allocations[strategy] = min(
                allocations[strategy],
                self.total_risk_budget * 0.4  # Máximo 40% por estrategia
            )
        
        return allocations

class PositionSizeValidators:
    """Validadores para tamaño de posiciones"""
    
    def validate_against_account(self, volume: float, symbol_info: MT5Symbol,
                               account_info: Dict[str, float]) -> Tuple[bool, List[str]]:
        """Valida el tamaño contra restricciones de la cuenta"""
        
        warnings = []
        
        # Validar margen requerido
        margin_required = volume * symbol_info.margin_required
        free_margin = account_info.get('free_margin', 0)
        
        if margin_required > free_margin * 0.8:
            warnings.append(f"Position uses {margin_required/free_margin*100:.1f}% of free margin")
        
        if margin_required > free_margin:
            return False, ["Insufficient margin"]
        
        # Validar contra límites del símbolo
        if volume < symbol_info.volume_min:
            warnings.append(f"Volume below minimum: {symbol_info.volume_min}")
            return False, warnings
        
        if volume > symbol_info.volume_max:
            warnings.append(f"Volume above maximum: {symbol_info.volume_max}")
            return False, warnings
        
        # Validar step
        if volume % symbol_info.volume_step != 0:
            warnings.append("Volume adjusted to match step size")
        
        return True, warnings
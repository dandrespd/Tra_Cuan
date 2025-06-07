import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from scipy import optimize
from scipy.stats import norm, multivariate_normal
import cvxpy as cp
from sklearn.covariance import LedoitWolf
import warnings

from utils.log_config import get_logger
from analysis.performance_analyzer import PerformanceMetrics
from risk.position_sizer import RiskParameters

@dataclass
class PortfolioAsset:
    """Activo individual en el portfolio"""
    symbol: str
    expected_return: float
    volatility: float
    current_weight: float = 0.0
    min_weight: float = 0.0
    max_weight: float = 1.0
    transaction_cost: float = 0.001
    
@dataclass
class OptimizationConstraints:
    """Restricciones para la optimización"""
    max_total_risk: float  # Volatilidad máxima del portfolio
    max_concentration: float  # Peso máximo por activo
    min_diversification: int  # Número mínimo de activos
    max_turnover: float  # Rotación máxima permitida
    sector_limits: Dict[str, float] = field(default_factory=dict)
    correlation_limit: float = 0.8  # Correlación máxima entre activos
    
@dataclass
class OptimizationResult:
    """Resultado de la optimización"""
    optimal_weights: Dict[str, float]
    expected_return: float
    expected_volatility: float
    sharpe_ratio: float
    diversification_ratio: float
    effective_assets: float  # Número efectivo de activos
    turnover: float
    transaction_costs: float
    optimization_method: str
    convergence_info: Dict[str, Any]

class PortfolioOptimizer:
    """Optimizador principal de portfolio"""
    
    def __init__(self, risk_free_rate: float = 0.02):
        self.risk_free_rate = risk_free_rate
        self.logger = get_logger(__name__)
        
        # Métodos de optimización disponibles
        self.optimization_methods = {
            'mean_variance': self._mean_variance_optimization,
            'minimum_variance': self._minimum_variance_optimization,
            'maximum_sharpe': self._maximum_sharpe_optimization,
            'risk_parity': self._risk_parity_optimization,
            'hierarchical_risk_parity': self._hierarchical_risk_parity,
            'black_litterman': self._black_litterman_optimization,
            'cvar_optimization': self._cvar_optimization,
            'robust_optimization': self._robust_optimization
        }
        
        # Estimadores de covarianza
        self.covariance_estimators = {
            'sample': self._sample_covariance,
            'ledoit_wolf': self._ledoit_wolf_covariance,
            'mcd': self._minimum_covariance_determinant,
            'factor_model': self._factor_model_covariance
        }
        
        # Cache de cálculos
        self.calculation_cache = {}
        
    def optimize_portfolio(self, assets: List[PortfolioAsset],
                         returns_data: pd.DataFrame,
                         constraints: OptimizationConstraints,
                         method: str = 'maximum_sharpe',
                         covariance_method: str = 'ledoit_wolf') -> OptimizationResult:
        """Optimiza el portfolio según el método especificado"""
        
        # Validar inputs
        if len(assets) < 2:
            raise ValueError("Se necesitan al menos 2 activos para optimizar")
        
        # Preparar datos
        symbols = [asset.symbol for asset in assets]
        returns = returns_data[symbols].dropna()
        
        if len(returns) < 30:
            self.logger.warning("Pocos datos históricos para optimización robusta")
        
        # Calcular matriz de covarianza
        cov_matrix = self.covariance_estimators[covariance_method](returns)
        
        # Calcular retornos esperados
        expected_returns = self._calculate_expected_returns(assets, returns)
        
        # Optimizar según método seleccionado
        if method in self.optimization_methods:
            result = self.optimization_methods[method](
                expected_returns, cov_matrix, assets, constraints
            )
        else:
            raise ValueError(f"Método de optimización no soportado: {method}")
        
        # Post-procesar resultado
        final_result = self._post_process_result(result, assets, cov_matrix)
        
        # Validar resultado
        self._validate_result(final_result, constraints)
        
        return final_result
    
    def _mean_variance_optimization(self, expected_returns: np.ndarray,
                                  cov_matrix: np.ndarray,
                                  assets: List[PortfolioAsset],
                                  constraints: OptimizationConstraints) -> Dict[str, Any]:
        """Optimización clásica de Markowitz"""
        
        n_assets = len(assets)
        
        # Variables de decisión
        weights = cp.Variable(n_assets)
        
        # Retorno esperado del portfolio
        portfolio_return = expected_returns @ weights
        
        # Varianza del portfolio
        portfolio_variance = cp.quad_form(weights, cov_matrix)
        
        # Función objetivo: maximizar utilidad (retorno - lambda * varianza)
        risk_aversion = 2.0  # Parámetro de aversión al riesgo
        objective = cp.Maximize(portfolio_return - risk_aversion * portfolio_variance)
        
        # Restricciones
        constraints_list = [
            cp.sum(weights) == 1,  # Pesos suman 1
            weights >= 0,  # No short selling (puede modificarse)
        ]
        
        # Restricción de concentración
        constraints_list.append(weights <= constraints.max_concentration)
        
        # Restricción de volatilidad máxima
        constraints_list.append(
            portfolio_variance <= constraints.max_total_risk ** 2
        )
        
        # Resolver
        problem = cp.Problem(objective, constraints_list)
        problem.solve(solver=cp.OSQP)
        
        if problem.status != 'optimal':
            self.logger.warning(f"Optimización no convergió: {problem.status}")
        
        return {
            'weights': weights.value,
            'expected_return': portfolio_return.value,
            'variance': portfolio_variance.value,
            'status': problem.status
        }
    
    def _maximum_sharpe_optimization(self, expected_returns: np.ndarray,
                                   cov_matrix: np.ndarray,
                                   assets: List[PortfolioAsset],
                                   constraints: OptimizationConstraints) -> Dict[str, Any]:
        """Maximiza el ratio de Sharpe"""
        
        n_assets = len(assets)
        
        # Para maximizar Sharpe, usamos un truco: 
        # max(mu - rf) / sigma es equivalente a resolver un problema convexo
        
        def negative_sharpe(weights):
            """Negativo del Sharpe ratio para minimizar"""
            portfolio_return = np.dot(expected_returns, weights)
            portfolio_vol = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))
            
            if portfolio_vol == 0:
                return 999999
            
            sharpe = (portfolio_return - self.risk_free_rate) / portfolio_vol
            return -sharpe
        
        # Restricciones
        constraints_scipy = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},  # Suma = 1
        ]
        
        # Agregar restricción de volatilidad
        constraints_scipy.append({
            'type': 'ineq',
            'fun': lambda w: constraints.max_total_risk - np.sqrt(
                np.dot(w, np.dot(cov_matrix, w))
            )
        })
        
        # Límites para cada peso
        bounds = []
        for asset in assets:
            bounds.append((asset.min_weight, min(asset.max_weight, constraints.max_concentration)))
        
        # Peso inicial
        x0 = np.ones(n_assets) / n_assets
        
        # Optimizar
        result = optimize.minimize(
            negative_sharpe,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints_scipy,
            options={'ftol': 1e-9}
        )
        
        # Calcular métricas finales
        final_weights = result.x
        portfolio_return = np.dot(expected_returns, final_weights)
        portfolio_vol = np.sqrt(np.dot(final_weights, np.dot(cov_matrix, final_weights)))
        
        return {
            'weights': final_weights,
            'expected_return': portfolio_return,
            'variance': portfolio_vol ** 2,
            'sharpe_ratio': (portfolio_return - self.risk_free_rate) / portfolio_vol,
            'status': 'optimal' if result.success else 'failed',
            'convergence_info': {
                'iterations': result.nit,
                'success': result.success,
                'message': result.message
            }
        }
    
    def _risk_parity_optimization(self, expected_returns: np.ndarray,
                                cov_matrix: np.ndarray,
                                assets: List[PortfolioAsset],
                                constraints: OptimizationConstraints) -> Dict[str, Any]:
        """Optimización de paridad de riesgo"""
        
        n_assets = len(assets)
        
        def risk_parity_objective(weights):
            """Minimiza la diferencia en contribución de riesgo"""
            portfolio_vol = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))
            
            # Contribución marginal al riesgo
            marginal_contrib = np.dot(cov_matrix, weights) / portfolio_vol
            
            # Contribución al riesgo de cada activo
            contrib = weights * marginal_contrib
            
            # Objetivo: minimizar la varianza de las contribuciones
            # (todas deberían ser iguales)
            target_contrib = portfolio_vol / n_assets
            
            return np.sum((contrib - target_contrib) ** 2)
        
        # Restricciones
        constraints_scipy = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
        ]
        
        # Límites
        bounds = [(0.01, constraints.max_concentration) for _ in range(n_assets)]
        
        # Optimizar
        x0 = np.ones(n_assets) / n_assets
        result = optimize.minimize(
            risk_parity_objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints_scipy
        )
        
        final_weights = result.x
        portfolio_return = np.dot(expected_returns, final_weights)
        portfolio_vol = np.sqrt(np.dot(final_weights, np.dot(cov_matrix, final_weights)))
        
        return {
            'weights': final_weights,
            'expected_return': portfolio_return,
            'variance': portfolio_vol ** 2,
            'status': 'optimal' if result.success else 'failed'
        }
    
    def _hierarchical_risk_parity(self, expected_returns: np.ndarray,
                                cov_matrix: np.ndarray,
                                assets: List[PortfolioAsset],
                                constraints: OptimizationConstraints) -> Dict[str, Any]:
        """Hierarchical Risk Parity (HRP) - López de Prado"""
        
        # 1. Calcular matriz de distancia
        corr_matrix = self._cov_to_corr(cov_matrix)
        distance_matrix = np.sqrt(0.5 * (1 - corr_matrix))
        
        # 2. Clustering jerárquico
        clusters = self._hierarchical_clustering(distance_matrix)
        
        # 3. Quasi-diagonalización
        sorted_indices = self._quasi_diagonalization(clusters)
        
        # 4. Asignación recursiva de pesos
        weights = self._recursive_bisection(
            cov_matrix[sorted_indices][:, sorted_indices],
            sorted_indices
        )
        
        # Reordenar pesos
        final_weights = np.zeros(len(assets))
        for i, idx in enumerate(sorted_indices):
            final_weights[idx] = weights[i]
        
        # Normalizar y aplicar restricciones
        final_weights = self._apply_weight_constraints(
            final_weights, assets, constraints
        )
        
        portfolio_return = np.dot(expected_returns, final_weights)
        portfolio_vol = np.sqrt(np.dot(final_weights, np.dot(cov_matrix, final_weights)))
        
        return {
            'weights': final_weights,
            'expected_return': portfolio_return,
            'variance': portfolio_vol ** 2,
            'status': 'optimal'
        }
    
    def _black_litterman_optimization(self, expected_returns: np.ndarray,
                                    cov_matrix: np.ndarray,
                                    assets: List[PortfolioAsset],
                                    constraints: OptimizationConstraints,
                                    views: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Optimización Black-Litterman con views del inversor"""
        
        n_assets = len(assets)
        
        # Parámetros Black-Litterman
        tau = 0.05  # Incertidumbre en los retornos de equilibrio
        
        # Calcular retornos de equilibrio (CAPM)
        market_weights = self._calculate_market_weights(assets)
        risk_aversion = self._estimate_risk_aversion(market_weights, cov_matrix)
        equilibrium_returns = risk_aversion * np.dot(cov_matrix, market_weights)
        
        if views and 'P' in views and 'Q' in views:
            # Incorporar views del inversor
            P = views['P']  # Matriz de views
            Q = views['Q']  # Vector de views
            omega = views.get('omega', np.eye(len(Q)) * 0.01)  # Incertidumbre de views
            
            # Actualizar retornos esperados
            tau_cov = tau * cov_matrix
            
            # Fórmula Black-Litterman
            term1 = np.linalg.inv(tau_cov)
            term2 = np.dot(P.T, np.dot(np.linalg.inv(omega), P))
            posterior_precision = term1 + term2
            
            posterior_mean_term1 = np.dot(term1, equilibrium_returns)
            posterior_mean_term2 = np.dot(P.T, np.dot(np.linalg.inv(omega), Q))
            posterior_mean = np.dot(
                np.linalg.inv(posterior_precision),
                posterior_mean_term1 + posterior_mean_term2
            )
            
            # Covarianza posterior
            posterior_cov = np.linalg.inv(posterior_precision)
            
        else:
            # Sin views, usar retornos de equilibrio
            posterior_mean = equilibrium_returns
            posterior_cov = cov_matrix * (1 + tau)
        
        # Optimizar con retornos Black-Litterman
        return self._mean_variance_optimization(
            posterior_mean, posterior_cov, assets, constraints
        )
    
    def _cvar_optimization(self, expected_returns: np.ndarray,
                         cov_matrix: np.ndarray,
                         assets: List[PortfolioAsset],
                         constraints: OptimizationConstraints,
                         confidence_level: float = 0.95) -> Dict[str, Any]:
        """Optimización minimizando Conditional Value at Risk (CVaR)"""
        
        # Generar escenarios usando simulación Monte Carlo
        n_scenarios = 5000
        scenarios = self._generate_return_scenarios(
            expected_returns, cov_matrix, n_scenarios
        )
        
        n_assets = len(assets)
        
        # Variables
        weights = cp.Variable(n_assets)
        z = cp.Variable()  # VaR
        u = cp.Variable(n_scenarios)  # Variables auxiliares para CVaR
        
        # Retornos del portfolio en cada escenario
        portfolio_returns = scenarios @ weights
        
        # Restricciones para CVaR
        constraints_list = [
            u >= 0,
            u >= z - portfolio_returns,
            cp.sum(weights) == 1,
            weights >= 0,
            weights <= constraints.max_concentration
        ]
        
        # Función objetivo: minimizar CVaR
        alpha = 1 - confidence_level
        cvar = z - (1 / (n_scenarios * alpha)) * cp.sum(u)
        
        # Agregar término de retorno esperado
        lambda_return = 0.1  # Trade-off entre riesgo y retorno
        objective = cp.Minimize(-cvar - lambda_return * (expected_returns @ weights))
        
        # Resolver
        problem = cp.Problem(objective, constraints_list)
        problem.solve(solver=cp.OSQP)
        
        final_weights = weights.value
        portfolio_return = np.dot(expected_returns, final_weights)
        portfolio_vol = np.sqrt(np.dot(final_weights, np.dot(cov_matrix, final_weights)))
        
        return {
            'weights': final_weights,
            'expected_return': portfolio_return,
            'variance': portfolio_vol ** 2,
            'cvar': -cvar.value,
            'var': z.value,
            'status': problem.status
        }

class DynamicPortfolioRebalancer:
    """Rebalanceador dinámico de portfolio"""
    
    def __init__(self, optimizer: PortfolioOptimizer):
        self.optimizer = optimizer
        self.rebalance_history = []
        self.transaction_cost_model = TransactionCostModel()
        
    def should_rebalance(self, current_weights: Dict[str, float],
                        target_weights: Dict[str, float],
                        market_conditions: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        """Determina si se debe rebalancear el portfolio"""
        
        # Calcular desviación de pesos objetivo
        total_deviation = sum(
            abs(current_weights.get(symbol, 0) - target_weights.get(symbol, 0))
            for symbol in set(current_weights) | set(target_weights)
        )
        
        # Factores para decisión de rebalanceo
        factors = {
            'weight_deviation': total_deviation,
            'time_since_last': self._time_since_last_rebalance(),
            'market_volatility': market_conditions.get('volatility', 'normal'),
            'transaction_costs': self._estimate_rebalance_cost(
                current_weights, target_weights
            )
        }
        
        # Reglas de rebalanceo
        should_rebalance = False
        reasons = []
        
        # Rebalanceo por desviación
        if factors['weight_deviation'] > 0.10:  # 10% desviación total
            should_rebalance = True
            reasons.append('weight_deviation_high')
        
        # Rebalanceo periódico
        if factors['time_since_last'] > 30:  # 30 días
            should_rebalance = True
            reasons.append('periodic_rebalance')
        
        # No rebalancear en alta volatilidad
        if factors['market_volatility'] == 'extreme':
            should_rebalance = False
            reasons.append('high_volatility_override')
        
        # Verificar costo-beneficio
        if should_rebalance:
            benefit = self._estimate_rebalance_benefit(
                current_weights, target_weights
            )
            if benefit < factors['transaction_costs'] * 2:
                should_rebalance = False
                reasons.append('cost_benefit_negative')
        
        return should_rebalance, {
            'factors': factors,
            'reasons': reasons
        }
    
    def calculate_rebalance_trades(self, current_positions: Dict[str, float],
                                 target_weights: Dict[str, float],
                                 total_value: float) -> List[Dict[str, Any]]:
        """Calcula las operaciones necesarias para rebalancear"""
        
        trades = []
        
        # Calcular valores objetivo
        target_values = {
            symbol: weight * total_value 
            for symbol, weight in target_weights.items()
        }
        
        # Calcular diferencias
        for symbol in set(current_positions) | set(target_weights):
            current_value = current_positions.get(symbol, 0)
            target_value = target_values.get(symbol, 0)
            difference = target_value - current_value
            
            if abs(difference) > total_value * 0.001:  # Umbral mínimo
                trades.append({
                    'symbol': symbol,
                    'action': 'buy' if difference > 0 else 'sell',
                    'amount': abs(difference),
                    'current_value': current_value,
                    'target_value': target_value
                })
        
        # Optimizar orden de trades para minimizar riesgo
        optimized_trades = self._optimize_trade_sequence(trades)
        
        return optimized_trades

class PortfolioRiskAnalyzer:
    """Analizador de riesgos del portfolio"""
    
    def __init__(self):
        self.risk_metrics = {}
        
    def analyze_portfolio_risk(self, weights: Dict[str, float],
                             returns_data: pd.DataFrame,
                             cov_matrix: np.ndarray) -> Dict[str, Any]:
        """Análisis completo de riesgos del portfolio"""
        
        # Convertir a arrays
        symbols = list(weights.keys())
        w = np.array([weights[s] for s in symbols])
        
        # Métricas básicas
        portfolio_return = returns_data[symbols].mean().values @ w
        portfolio_vol = np.sqrt(w @ cov_matrix @ w)
        
        # VaR y CVaR
        portfolio_returns = returns_data[symbols] @ w
        var_95 = np.percentile(portfolio_returns, 5)
        cvar_95 = portfolio_returns[portfolio_returns <= var_95].mean()
        
        # Métricas avanzadas
        risk_metrics = {
            'volatility': portfolio_vol,
            'var_95': var_95,
            'cvar_95': cvar_95,
            'sharpe_ratio': portfolio_return / portfolio_vol * np.sqrt(252),
            'max_drawdown': self._calculate_max_drawdown(portfolio_returns),
            'downside_deviation': self._calculate_downside_deviation(portfolio_returns),
            'beta': self._calculate_portfolio_beta(portfolio_returns, returns_data),
            'tracking_error': self._calculate_tracking_error(portfolio_returns, returns_data)
        }
        
        # Análisis de contribución al riesgo
        risk_contributions = self._calculate_risk_contributions(w, cov_matrix, symbols)
        
        # Análisis de diversificación
        diversification_metrics = self._analyze_diversification(w, cov_matrix)
        
        return {
            'basic_metrics': risk_metrics,
            'risk_contributions': risk_contributions,
            'diversification': diversification_metrics,
            'stress_test_results': self._perform_stress_tests(w, returns_data[symbols])
        }
    
    def _calculate_risk_contributions(self, weights: np.ndarray,
                                    cov_matrix: np.ndarray,
                                    symbols: List[str]) -> Dict[str, float]:
        """Calcula la contribución de cada activo al riesgo total"""
        
        portfolio_vol = np.sqrt(weights @ cov_matrix @ weights)
        
        # Contribución marginal
        marginal_contrib = cov_matrix @ weights / portfolio_vol
        
        # Contribución al riesgo
        risk_contrib = weights * marginal_contrib
        
        # Porcentaje de contribución
        risk_contrib_pct = risk_contrib / risk_contrib.sum()
        
        return {
            symbols[i]: {
                'contribution': risk_contrib[i],
                'percentage': risk_contrib_pct[i],
                'marginal': marginal_contrib[i]
            }
            for i in range(len(symbols))
        }
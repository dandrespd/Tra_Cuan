import pandas as pd, numpy as np
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
from utils.log_config import get_logger

@dataclass
class PerformanceMetrics:
    """Métricas completas de performance"""
    # Returns
    total_return: float
    annualized_return: float
    monthly_returns: pd.Series
    
    # Risk metrics
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    information_ratio: float
    
    # Drawdown
    max_drawdown: float
    max_drawdown_duration: int
    recovery_time: Optional[int]
    underwater_curve: pd.Series
    
    # Win/Loss
    win_rate: float
    profit_factor: float
    expectancy: float
    avg_win: float
    avg_loss: float
    
    # Risk-adjusted
    omega_ratio: float
    kappa_ratio: float
    upside_potential_ratio: float
    
    # Attribution
    strategy_attribution: Dict[str, float]
    timeframe_attribution: Dict[str, float]
    
class PerformanceAnalyzer:
    """Analizador principal de performance"""
    
    def __init__(self, benchmark: Optional[pd.Series] = None):
        self.benchmark = benchmark
        self.risk_free_rate = 0.02  # 2% anual
        
    def analyze_performance(self, trades: pd.DataFrame, 
                          equity_curve: pd.Series) -> PerformanceMetrics:
        """Análisis completo de performance"""
        
        # Calcular returns
        returns = equity_curve.pct_change().dropna()
        
        # Métricas básicas
        total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1
        annualized_return = self._annualize_return(total_return, equity_curve.index)
        
        # Risk metrics
        sharpe = self._calculate_sharpe_ratio(returns)
        sortino = self._calculate_sortino_ratio(returns)
        calmar = self._calculate_calmar_ratio(annualized_return, equity_curve)
        
        # Drawdown analysis
        drawdown_analysis = self._analyze_drawdowns(equity_curve)
        
        # Win/Loss analysis
        win_loss_analysis = self._analyze_trades(trades)
        
        # Advanced metrics
        omega = self._calculate_omega_ratio(returns)
        
        # Attribution analysis
        attribution = self._performance_attribution(trades, returns)
        
        return PerformanceMetrics(
            total_return=total_return,
            annualized_return=annualized_return,
            monthly_returns=self._calculate_monthly_returns(returns),
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            calmar_ratio=calmar,
            information_ratio=self._calculate_information_ratio(returns),
            max_drawdown=drawdown_analysis['max_drawdown'],
            max_drawdown_duration=drawdown_analysis['max_duration'],
            recovery_time=drawdown_analysis['recovery_time'],
            underwater_curve=drawdown_analysis['underwater_curve'],
            win_rate=win_loss_analysis['win_rate'],
            profit_factor=win_loss_analysis['profit_factor'],
            expectancy=win_loss_analysis['expectancy'],
            avg_win=win_loss_analysis['avg_win'],
            avg_loss=win_loss_analysis['avg_loss'],
            omega_ratio=omega,
            kappa_ratio=self._calculate_kappa_ratio(returns, 3),
            upside_potential_ratio=self._calculate_upside_potential_ratio(returns),
            strategy_attribution=attribution['by_strategy'],
            timeframe_attribution=attribution['by_timeframe']
        )
    
    def _analyze_drawdowns(self, equity_curve: pd.Series) -> Dict[str, Any]:
        """Análisis detallado de drawdowns"""
        # Calculate running maximum
        running_max = equity_curve.expanding().max()
        
        # Drawdown series
        drawdown = (equity_curve - running_max) / running_max
        
        # Find all drawdown periods
        drawdown_periods = []
        in_drawdown = False
        start_idx = None
        
        for i, dd in enumerate(drawdown):
            if dd < 0 and not in_drawdown:
                in_drawdown = True
                start_idx = i
            elif dd == 0 and in_drawdown:
                in_drawdown = False
                end_idx = i
                
                period_data = {
                    'start': equity_curve.index[start_idx],
                    'end': equity_curve.index[end_idx],
                    'depth': drawdown[start_idx:end_idx].min(),
                    'duration': end_idx - start_idx,
                    'recovery_time': end_idx - drawdown[start_idx:end_idx].idxmin()
                }
                drawdown_periods.append(period_data)
        
        # Max drawdown stats
        max_dd = drawdown.min()
        max_dd_idx = drawdown.idxmin()
        
        return {
            'max_drawdown': abs(max_dd),
            'max_drawdown_date': max_dd_idx,
            'max_duration': max(p['duration'] for p in drawdown_periods) if drawdown_periods else 0,
            'recovery_time': self._calculate_recovery_time(drawdown, max_dd_idx),
            'underwater_curve': drawdown,
            'all_drawdowns': drawdown_periods,
            'current_drawdown': drawdown.iloc[-1] if drawdown.iloc[-1] < 0 else 0
        }
    
    def _performance_attribution(self, trades: pd.DataFrame, 
                               returns: pd.Series) -> Dict[str, Dict[str, float]]:
        """Attribution analysis - qué contribuye a los returns"""
        attribution = {
            'by_strategy': {},
            'by_timeframe': {},
            'by_symbol': {},
            'by_direction': {},
            'by_day_of_week': {},
            'by_hour': {},
            'by_market_condition': {}
        }
        
        # Por estrategia
        for strategy in trades['strategy'].unique():
            strategy_trades = trades[trades['strategy'] == strategy]
            strategy_pnl = strategy_trades['pnl'].sum()
            attribution['by_strategy'][strategy] = strategy_pnl
        
        # Por timeframe
        trades['hour'] = pd.to_datetime(trades['entry_time']).dt.hour
        for hour in range(24):
            hour_trades = trades[trades['hour'] == hour]
            if not hour_trades.empty:
                attribution['by_hour'][f'hour_{hour}'] = hour_trades['pnl'].sum()
        
        # Normalizar attribution
        total_pnl = trades['pnl'].sum()
        for key in attribution:
            if attribution[key]:
                total = sum(attribution[key].values())
                attribution[key] = {
                    k: v/total if total != 0 else 0 
                    for k, v in attribution[key].items()
                }
        
        return attribution

class MonteCarloSimulator:
    """Simulación Monte Carlo para confidence intervals"""
    
    def simulate_performance(self, returns: pd.Series, 
                           n_simulations: int = 1000,
                           n_periods: int = 252) -> Dict[str, np.ndarray]:
        """Simula múltiples paths de performance"""
        results = {
            'terminal_wealth': np.zeros(n_simulations),
            'max_drawdown': np.zeros(n_simulations),
            'sharpe_ratio': np.zeros(n_simulations)
        }
        
        # Bootstrap returns
        for i in range(n_simulations):
            # Resample with replacement
            simulated_returns = np.random.choice(returns, size=n_periods, replace=True)
            
            # Calculate metrics
            equity_curve = (1 + pd.Series(simulated_returns)).cumprod()
            results['terminal_wealth'][i] = equity_curve.iloc[-1]
            
            # Drawdown
            running_max = equity_curve.expanding().max()
            drawdown = (equity_curve - running_max) / running_max
            results['max_drawdown'][i] = abs(drawdown.min())
            
            # Sharpe
            results['sharpe_ratio'][i] = (
                simulated_returns.mean() / simulated_returns.std() * np.sqrt(252)
            )
        
        return results
    
    def calculate_var_cvar(self, simulation_results: Dict[str, np.ndarray],
                          confidence_level: float = 0.95) -> Dict[str, float]:
        """Calcula VaR y CVaR de los resultados simulados"""
        terminal_wealth = simulation_results['terminal_wealth']
        
        # VaR
        var_percentile = (1 - confidence_level) * 100
        var = np.percentile(terminal_wealth - 1, var_percentile)  # Returns
        
        # CVaR (Expected Shortfall)
        losses = terminal_wealth - 1
        cvar = losses[losses <= var].mean()
        
        return {
            'var': abs(var),
            'cvar': abs(cvar),
            'var_percentile': var_percentile
        }

class TradeAnalyzer:
    """Análisis detallado de trades individuales"""
    
    def analyze_trade_quality(self, trades: pd.DataFrame) -> Dict[str, Any]:
        """Analiza calidad de entrada y salida de trades"""
        analysis = {}
        
        # Entry efficiency (qué tan cerca del mejor precio)
        trades['entry_efficiency'] = self._calculate_entry_efficiency(trades)
        
        # Exit efficiency
        trades['exit_efficiency'] = self._calculate_exit_efficiency(trades)
        
        # MAE/MFE analysis
        mae_mfe = self._calculate_mae_mfe(trades)
        
        # Trade duration analysis
        trades['duration'] = (
            pd.to_datetime(trades['exit_time']) - pd.to_datetime(trades['entry_time'])
        )
        
        analysis['avg_entry_efficiency'] = trades['entry_efficiency'].mean()
        analysis['avg_exit_efficiency'] = trades['exit_efficiency'].mean()
        analysis['mae_mfe_ratio'] = mae_mfe['avg_mae'] / mae_mfe['avg_mfe']
        analysis['avg_trade_duration'] = trades['duration'].mean()
        
        # Edge analysis
        analysis['edge'] = self._calculate_trading_edge(trades)
        
        return analysis

class RollingPerformanceAnalyzer:
    """Análisis de performance rolling para detectar cambios"""
    
    def analyze_rolling_metrics(self, returns: pd.Series, 
                              window: int = 252) -> pd.DataFrame:
        """Calcula métricas rolling"""
        rolling_metrics = pd.DataFrame(index=returns.index)
        
        # Rolling returns
        rolling_metrics['returns_1m'] = returns.rolling(21).sum()
        rolling_metrics['returns_3m'] = returns.rolling(63).sum()
        rolling_metrics['returns_6m'] = returns.rolling(126).sum()
        
        # Rolling volatility
        rolling_metrics['volatility'] = returns.rolling(window).std() * np.sqrt(252)
        
        # Rolling Sharpe
        rolling_metrics['sharpe'] = (
            returns.rolling(window).mean() / returns.rolling(window).std() * np.sqrt(252)
        )
        
        # Rolling max drawdown
        rolling_metrics['max_dd'] = returns.rolling(window).apply(
            lambda x: self._calculate_max_drawdown(x)
        )
        
        # Regime detection
        rolling_metrics['regime'] = self._detect_performance_regime(rolling_metrics)
        
        return rolling_metrics
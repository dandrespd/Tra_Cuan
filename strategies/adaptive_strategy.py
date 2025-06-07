from strategies.base_strategy import BaseStrategy, TradingSignal
from control.adaptive_controller import AdaptiveController
from analysis.market_analyzer import MarketAnalyzer
from typing import Dict, List, Optional

@dataclass
class AdaptiveParameters:
    """Parámetros que se adaptan dinámicamente"""
    # Agresividad
    position_size_multiplier: float = 1.0
    max_positions: int = 3
    
    # Umbrales de entrada
    entry_threshold: float = 0.65
    confirmation_required: bool = False
    
    # Gestión de riesgo
    stop_loss_multiplier: float = 1.0
    take_profit_multiplier: float = 1.0
    use_trailing_stop: bool = True
    
    # Filtros temporales
    trading_hours_restriction: Optional[Tuple[int, int]] = None
    avoid_news_trading: bool = True
    
    # Indicadores activos
    active_indicators: List[str] = field(default_factory=list)
    indicator_weights: Dict[str, float] = field(default_factory=dict)
    
    def adapt_to_volatility(self, volatility_ratio: float):
        """Adapta parámetros según volatilidad"""
        if volatility_ratio > 1.5:  # Alta volatilidad
            self.position_size_multiplier *= 0.7
            self.stop_loss_multiplier *= 1.3
            self.entry_threshold *= 1.1
        elif volatility_ratio < 0.7:  # Baja volatilidad
            self.position_size_multiplier *= 1.2
            self.stop_loss_multiplier *= 0.8
            self.entry_threshold *= 0.9

class AdaptiveStrategy(BaseStrategy):
    """Estrategia adaptativa que ajusta su comportamiento según el mercado"""
    
    def __init__(self, config: StrategyConfig, model_hub: ModelHub,
                adaptive_controller: AdaptiveController):
        super().__init__(config, model_hub)
        
        self.adaptive_controller = adaptive_controller
        self.adaptive_params = AdaptiveParameters()
        
        # Sub-estrategias disponibles
        self.sub_strategies = {
            'trend_following': TrendFollowingStrategy(),
            'mean_reversion': MeanReversionStrategy(),
            'breakout': BreakoutStrategy(),
            'range_trading': RangeTradingStrategy()
        }
        
        # Estado de adaptación
        self.current_regime = "unknown"
        self.adaptation_history = []
        self.performance_window = deque(maxlen=100)
        
        # Analizadores
        self.regime_analyzer = MarketRegimeAnalyzer()
        self.performance_analyzer = StrategyPerformanceAnalyzer()
        
    def _generate_signal_logic(self, data: pd.DataFrame, 
                              market_conditions: Dict[str, Any]) -> TradingSignal:
        """Genera señal adaptándose a condiciones del mercado"""
        
        # 1. Detectar régimen de mercado actual
        self.current_regime = self.regime_analyzer.detect_regime(data)
        
        # 2. Adaptar parámetros según condiciones
        self._adapt_parameters(data, market_conditions)
        
        # 3. Seleccionar sub-estrategia apropiada
        selected_strategy = self._select_strategy(market_conditions)
        
        # 4. Generar señal base con sub-estrategia
        base_signal = selected_strategy.generate_signal(data, market_conditions)
        
        # 5. Aplicar filtros adaptativos
        filtered_signal = self._apply_adaptive_filters(base_signal, data)
        
        # 6. Ajustar niveles de precio según volatilidad
        adjusted_signal = self._adjust_price_levels(filtered_signal, data)
        
        # 7. Confirmar con modelos ML si están disponibles
        if self.model_hub and base_signal.signal_type != SignalType.NO_SIGNAL:
            ml_confirmation = self._get_ml_confirmation(data)
            if ml_confirmation < self.adaptive_params.entry_threshold:
                return self._create_no_signal("ML confirmation failed")
        
        # Registrar decisión
        self._log_adaptation_decision(adjusted_signal)
        
        return adjusted_signal
    
    def _adapt_parameters(self, data: pd.DataFrame, 
                         market_conditions: Dict[str, Any]):
        """Adapta parámetros según análisis del mercado"""
        
        # Calcular métricas de adaptación
        volatility_ratio = self._calculate_volatility_ratio(data)
        trend_strength = self._calculate_trend_strength(data)
        recent_performance = self._get_recent_performance()
        
        # Resetear parámetros
        self.adaptive_params = AdaptiveParameters()
        
        # Adaptar según volatilidad
        self.adaptive_params.adapt_to_volatility(volatility_ratio)
        
        # Adaptar según tendencia
        if trend_strength > 0.7:
            # Mercado con tendencia fuerte
            self.adaptive_params.active_indicators = [
                'ema_20', 'ema_50', 'adx', 'macd'
            ]
            self.adaptive_params.confirmation_required = False
        else:
            # Mercado lateral
            self.adaptive_params.active_indicators = [
                'rsi', 'stochastic', 'bollinger_bands'
            ]
            self.adaptive_params.confirmation_required = True
        
        # Adaptar según performance reciente
        if recent_performance < 0.4:  # Win rate bajo
            self.adaptive_params.entry_threshold *= 1.2
            self.adaptive_params.max_positions -= 1
        elif recent_performance > 0.7:  # Win rate alto
            self.adaptive_params.entry_threshold *= 0.9
            self.adaptive_params.max_positions += 1
        
        # Restricciones por condiciones especiales
        if market_conditions.get('high_impact_news_soon'):
            self.adaptive_params.avoid_news_trading = True
            self.adaptive_params.position_size_multiplier *= 0.5
    
    def _select_strategy(self, market_conditions: Dict[str, Any]) -> BaseStrategy:
        """Selecciona la mejor sub-estrategia para las condiciones actuales"""
        
        regime_strategy_map = {
            'trending_up': 'trend_following',
            'trending_down': 'trend_following',
            'ranging': 'range_trading',
            'breakout_imminent': 'breakout',
            'high_volatility': 'mean_reversion',
            'low_volatility': 'range_trading'
        }
        
        # Seleccionar basado en régimen
        strategy_name = regime_strategy_map.get(
            self.current_regime, 'trend_following'
        )
        
        # Override por condiciones específicas
        if market_conditions.get('volatility', 'normal') == 'extreme':
            strategy_name = 'mean_reversion'
        
        selected = self.sub_strategies[strategy_name]
        
        # Configurar sub-estrategia con parámetros adaptativos
        selected.update_config(self.adaptive_params)
        
        return selected
    
    def _apply_adaptive_filters(self, signal: TradingSignal, 
                               data: pd.DataFrame) -> TradingSignal:
        """Aplica filtros que se adaptan a las condiciones"""
        
        if signal.signal_type == SignalType.NO_SIGNAL:
            return signal
        
        # Filtro de volatilidad
        current_atr = data['atr_14'].iloc[-1]
        avg_atr = data['atr_14'].rolling(50).mean().iloc[-1]
        
        if current_atr > avg_atr * 2:  # Volatilidad muy alta
            signal.confidence *= 0.7
            signal.reasoning += " | High volatility filter applied"
        
        # Filtro de sesión
        current_hour = datetime.now().hour
        
        if self.current_regime == 'ranging':
            # En rango, evitar horas de baja liquidez
            if current_hour < 8 or current_hour > 20:
                signal.confidence *= 0.5
                signal.reasoning += " | Low liquidity hours filter"
        
        # Filtro de correlación con otros mercados
        if hasattr(self, 'correlation_data'):
            correlation_filter = self._apply_correlation_filter(signal)
            signal.confidence *= correlation_filter
        
        return signal
    
    def _adjust_price_levels(self, signal: TradingSignal, 
                           data: pd.DataFrame) -> TradingSignal:
        """Ajusta SL/TP según condiciones actuales"""
        
        if signal.signal_type in [SignalType.BUY, SignalType.SELL]:
            current_atr = data['atr_14'].iloc[-1]
            
            # Ajustar stop loss
            base_sl_distance = current_atr * 2
            adjusted_sl_distance = base_sl_distance * self.adaptive_params.stop_loss_multiplier
            
            # Ajustar take profit
            base_tp_distance = current_atr * 3
            adjusted_tp_distance = base_tp_distance * self.adaptive_params.take_profit_multiplier
            
            # Aplicar ajustes
            if signal.signal_type == SignalType.BUY:
                signal.stop_loss = signal.entry_price - adjusted_sl_distance
                signal.take_profit = signal.entry_price + adjusted_tp_distance
            else:
                signal.stop_loss = signal.entry_price + adjusted_sl_distance
                signal.take_profit = signal.entry_price - adjusted_tp_distance
            
            # Ajuste por régimen
            if self.current_regime == 'trending_up' and signal.signal_type == SignalType.BUY:
                # En tendencia alcista, dar más espacio al trade
                signal.take_profit *= 1.5
        
        return signal

class MarketRegimeAnalyzer:
    """Analiza y clasifica el régimen actual del mercado"""
    
    def __init__(self):
        self.regime_history = deque(maxlen=100)
        self.regime_models = {
            'hmm': HiddenMarkovRegimeModel(),
            'ml_classifier': RegimeClassifier(),
            'rule_based': RuleBasedRegimeDetector()
        }
    
    def detect_regime(self, data: pd.DataFrame) -> str:
        """Detecta el régimen actual del mercado"""
        
        # Obtener predicciones de cada modelo
        predictions = {}
        
        for name, model in self.regime_models.items():
            try:
                regime = model.predict(data)
                predictions[name] = regime
            except Exception as e:
                logger.warning(f"Error en {name}: {e}")
        
        # Consenso entre modelos
        if predictions:
            # Mayoría simple
            regime_counts = {}
            for regime in predictions.values():
                regime_counts[regime] = regime_counts.get(regime, 0) + 1
            
            consensus_regime = max(regime_counts.items(), key=lambda x: x[1])[0]
        else:
            # Fallback a detección simple
            consensus_regime = self._simple_regime_detection(data)
        
        # Actualizar historial
        self.regime_history.append({
            'timestamp': datetime.now(),
            'regime': consensus_regime,
            'predictions': predictions
        })
        
        return consensus_regime
    
    def _simple_regime_detection(self, data: pd.DataFrame) -> str:
        """Detección simple basada en reglas"""
        
        # Calcular indicadores
        sma_20 = data['close'].rolling(20).mean()
        sma_50 = data['close'].rolling(50).mean()
        
        current_price = data['close'].iloc[-1]
        atr = data['atr_14'].iloc[-1] if 'atr_14' in data else None
        
        # Tendencia
        if current_price > sma_20.iloc[-1] > sma_50.iloc[-1]:
            trend = 'up'
        elif current_price < sma_20.iloc[-1] < sma_50.iloc[-1]:
            trend = 'down'
        else:
            trend = 'neutral'
        
        # Volatilidad
        if atr:
            avg_atr = data['atr_14'].rolling(50).mean().iloc[-1]
            if atr > avg_atr * 1.5:
                volatility = 'high'
            elif atr < avg_atr * 0.7:
                volatility = 'low'
            else:
                volatility = 'normal'
        else:
            volatility = 'normal'
        
        # Mapear a régimen
        if trend == 'up' and volatility != 'high':
            return 'trending_up'
        elif trend == 'down' and volatility != 'high':
            return 'trending_down'
        elif volatility == 'high':
            return 'high_volatility'
        elif volatility == 'low':
            return 'low_volatility'
        else:
            return 'ranging'

class AdaptiveIndicatorWeighting:
    """Sistema para ponderar indicadores dinámicamente"""
    
    def __init__(self):
        self.indicator_performance = {}
        self.current_weights = {}
        self.weight_history = []
        
    def update_weights(self, indicator_signals: Dict[str, float],
                      actual_outcome: float) -> Dict[str, float]:
        """Actualiza pesos basado en performance"""
        
        # Registrar performance de cada indicador
        for indicator, signal in indicator_signals.items():
            if indicator not in self.indicator_performance:
                self.indicator_performance[indicator] = deque(maxlen=100)
            
            # Calcular accuracy del indicador
            accuracy = 1.0 if (signal > 0 and actual_outcome > 0) or \
                            (signal < 0 and actual_outcome < 0) else 0.0
            
            self.indicator_performance[indicator].append(accuracy)
        
        # Calcular nuevos pesos basados en performance reciente
        new_weights = {}
        
        for indicator, performance in self.indicator_performance.items():
            if len(performance) >= 10:
                # Peso basado en accuracy con decay temporal
                recent_accuracy = np.average(
                    list(performance),
                    weights=np.exp(np.linspace(-1, 0, len(performance)))
                )
                new_weights[indicator] = max(0.1, recent_accuracy)
            else:
                new_weights[indicator] = 0.5  # Peso neutral
        
        # Normalizar pesos
        total_weight = sum(new_weights.values())
        if total_weight > 0:
            new_weights = {k: v/total_weight for k, v in new_weights.items()}
        
        # Suavizar con pesos anteriores
        if self.current_weights:
            alpha = 0.3  # Factor de suavizado
            for indicator in new_weights:
                if indicator in self.current_weights:
                    new_weights[indicator] = (
                        alpha * new_weights[indicator] + 
                        (1 - alpha) * self.current_weights[indicator]
                    )
        
        self.current_weights = new_weights
        self.weight_history.append({
            'timestamp': datetime.now(),
            'weights': new_weights.copy()
        })
        
        return new_weights

class PerformanceAdaptation:
    """Adapta estrategia basado en performance reciente"""
    
    def __init__(self, lookback_window: int = 50):
        self.lookback_window = lookback_window
        self.trade_results = deque(maxlen=lookback_window)
        self.adaptation_rules = []
        
    def record_trade(self, trade_result: Dict[str, Any]):
        """Registra resultado de trade"""
        self.trade_results.append(trade_result)
    
    def suggest_adaptations(self) -> Dict[str, Any]:
        """Sugiere adaptaciones basadas en performance"""
        
        if len(self.trade_results) < 10:
            return {}
        
        # Calcular métricas de performance
        metrics = self._calculate_performance_metrics()
        
        adaptations = {}
        
        # Regla 1: Si win rate es bajo, ser más selectivo
        if metrics['win_rate'] < 0.4:
            adaptations['increase_entry_threshold'] = 0.1
            adaptations['reduce_position_size'] = 0.2
        
        # Regla 2: Si drawdown es alto, reducir riesgo
        if metrics['max_drawdown'] > 0.15:
            adaptations['reduce_max_positions'] = 1
            adaptations['increase_stop_loss'] = 0.2
        
        # Regla 3: Si profit factor es alto, incrementar agresividad
        if metrics['profit_factor'] > 2.0 and metrics['win_rate'] > 0.6:
            adaptations['increase_position_size'] = 0.2
            adaptations['decrease_entry_threshold'] = 0.05
        
        # Regla 4: Adaptación por tiempo del día
        if metrics['best_trading_hours']:
            adaptations['restrict_trading_hours'] = metrics['best_trading_hours']
        
        return adaptations
    
    def _calculate_performance_metrics(self) -> Dict[str, float]:
        """Calcula métricas de performance"""
        
        wins = [t for t in self.trade_results if t['profit'] > 0]
        losses = [t for t in self.trade_results if t['profit'] <= 0]
        
        win_rate = len(wins) / len(self.trade_results) if self.trade_results else 0
        
        avg_win = np.mean([t['profit'] for t in wins]) if wins else 0
        avg_loss = abs(np.mean([t['profit'] for t in losses])) if losses else 0
        
        profit_factor = (avg_win * len(wins)) / (avg_loss * len(losses)) \
                       if losses and avg_loss > 0 else 0
        
        # Análisis por hora
        hourly_performance = {}
        for trade in self.trade_results:
            hour = trade['entry_time'].hour
            if hour not in hourly_performance:
                hourly_performance[hour] = []
            hourly_performance[hour].append(trade['profit'])
        
        best_hours = sorted(
            hourly_performance.items(),
            key=lambda x: np.mean(x[1]),
            reverse=True
        )[:3]
        
        return {
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'max_drawdown': self._calculate_max_drawdown(),
            'best_trading_hours': [h[0] for h in best_hours]
        }
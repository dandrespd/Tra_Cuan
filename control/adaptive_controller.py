from typing import Dict, List, Optional
import numpy as np
from config.trading_config import TradingConfig
from analysis.market_analyzer import MarketAnalyzer
from utils.log_config import get_logger

@dataclass
class SystemState:
    """Estado actual del sistema de trading"""
    mode: str  # 'aggressive', 'normal', 'conservative', 'stopped'
    active_strategies: List[str]
    position_sizing_multiplier: float
    max_positions: int
    risk_per_trade: float
    confidence_threshold: float
    trading_enabled: bool
    last_update: datetime
    
class AdaptiveController:
    """Controlador principal adaptativo del sistema"""
    
    def __init__(self, config: TradingConfig, 
                market_analyzer: MarketAnalyzer,
                performance_analyzer: PerformanceAnalyzer):
        self.config = config
        self.market_analyzer = market_analyzer
        self.performance_analyzer = performance_analyzer
        
        # Estado inicial
        self.system_state = SystemState(
            mode='normal',
            active_strategies=[],
            position_sizing_multiplier=1.0,
            max_positions=config.max_positions,
            risk_per_trade=config.risk_per_trade,
            confidence_threshold=0.65,
            trading_enabled=True,
            last_update=datetime.now()
        )
        
        # Reglas de adaptación
        self.adaptation_rules = AdaptationRules()
        
        # State machine
        self.state_machine = TradingStateMachine()
        
        # Historial de decisiones
        self.decision_history = []
        
    def update_system_parameters(self, market_data: pd.DataFrame,
                               performance_metrics: PerformanceMetrics,
                               risk_metrics: RiskMetrics) -> SystemState:
        """Actualiza parámetros del sistema basado en condiciones actuales"""
        
        # Analizar mercado
        market_conditions = self.market_analyzer.analyze(market_data)
        
        # Evaluar performance reciente
        performance_score = self._evaluate_recent_performance(performance_metrics)
        
        # Evaluar riesgo actual
        risk_score = self._evaluate_risk_level(risk_metrics)
        
        # Tomar decisión de adaptación
        adaptation_decision = self.adaptation_rules.evaluate(
            market_conditions, performance_score, risk_score
        )
        
        # Aplicar cambios
        self._apply_adaptation(adaptation_decision)
        
        # Registrar decisión
        self.decision_history.append({
            'timestamp': datetime.now(),
            'decision': adaptation_decision,
            'market_conditions': market_conditions,
            'performance_score': performance_score,
            'risk_score': risk_score
        })
        
        return self.system_state
    
    def _apply_adaptation(self, decision: Dict[str, Any]):
        """Aplica decisión de adaptación al sistema"""
        
        # Cambiar modo del sistema
        if decision['mode_change']:
            self.system_state.mode = decision['new_mode']
            logger.info(f"Sistema cambiado a modo: {decision['new_mode']}")
        
        # Ajustar parámetros de riesgo
        if decision['adjust_risk']:
            self.system_state.risk_per_trade = decision['new_risk_per_trade']
            self.system_state.position_sizing_multiplier = decision['position_size_mult']
            
        # Cambiar estrategias activas
        if decision['strategy_changes']:
            self._update_active_strategies(decision['strategies_to_activate'],
                                         decision['strategies_to_deactivate'])
        
        # Ajustar umbrales
        if decision['adjust_thresholds']:
            self.system_state.confidence_threshold = decision['new_confidence_threshold']
        
        # Control de emergencia
        if decision['emergency_stop']:
            self.emergency_stop(decision['stop_reason'])
        
        self.system_state.last_update = datetime.now()

class AdaptationRules:
    """Reglas para adaptación del sistema"""
    
    def __init__(self):
        # Definir reglas como funciones evaluables
        self.rules = {
            'high_volatility_rule': self._high_volatility_adaptation,
            'drawdown_rule': self._drawdown_adaptation,
            'winning_streak_rule': self._winning_streak_adaptation,
            'regime_change_rule': self._regime_change_adaptation,
            'low_liquidity_rule': self._low_liquidity_adaptation
        }
        
    def evaluate(self, market_conditions: MarketConditions,
                performance_score: float, risk_score: float) -> Dict[str, Any]:
        """Evalúa todas las reglas y genera decisión"""
        
        decisions = []
        
        # Evaluar cada regla
        for rule_name, rule_func in self.rules.items():
            decision = rule_func(market_conditions, performance_score, risk_score)
            if decision:
                decisions.append(decision)
        
        # Combinar decisiones (resolver conflictos)
        final_decision = self._combine_decisions(decisions)
        
        return final_decision
    
    def _high_volatility_adaptation(self, market_conditions, perf_score, risk_score):
        """Adapta sistema en alta volatilidad"""
        if market_conditions.volatility == 'extreme':
            return {
                'mode_change': True,
                'new_mode': 'conservative',
                'adjust_risk': True,
                'new_risk_per_trade': 0.01,  # Reducir riesgo a 1%
                'position_size_mult': 0.5,    # Reducir tamaño 50%
                'strategy_changes': True,
                'strategies_to_deactivate': ['scalping', 'momentum'],
                'strategies_to_activate': ['mean_reversion']
            }
        return None

class TradingStateMachine:
    """Máquina de estados para el sistema de trading"""
    
    def __init__(self):
        self.states = {
            'initializing': InitializingState(),
            'warming_up': WarmingUpState(),
            'trading_normal': TradingNormalState(),
            'trading_conservative': TradingConservativeState(),
            'trading_aggressive': TradingAggressiveState(),
            'reducing_exposure': ReducingExposureState(),
            'emergency_stop': EmergencyStopState(),
            'stopped': StoppedState()
        }
        
        self.current_state = self.states['initializing']
        self.state_history = []
        
    def transition_to(self, new_state_name: str, reason: str = ""):
        """Transición a nuevo estado"""
        if new_state_name not in self.states:
            raise ValueError(f"Estado desconocido: {new_state_name}")
        
        # Registrar transición
        self.state_history.append({
            'from': self.current_state.name,
            'to': new_state_name,
            'timestamp': datetime.now(),
            'reason': reason
        })
        
        # Ejecutar salida del estado actual
        self.current_state.on_exit()
        
        # Cambiar estado
        self.current_state = self.states[new_state_name]
        
        # Ejecutar entrada al nuevo estado
        self.current_state.on_enter()
        
        logger.info(f"Transición de estado: {self.state_history[-1]['from']} -> {new_state_name}")
        logger.info(f"Razón: {reason}")

class ParameterOptimizer:
    """Optimiza parámetros del sistema en tiempo real"""
    
    def __init__(self):
        self.parameter_history = {}
        self.performance_history = {}
        self.optimizer = BayesianOptimizer()
        
    def optimize_parameters(self, current_params: Dict[str, Any],
                          recent_performance: float) -> Dict[str, Any]:
        """Optimiza parámetros basado en performance reciente"""
        
        # Registrar performance con parámetros actuales
        param_key = self._hash_params(current_params)
        self.performance_history[param_key] = recent_performance
        
        # Sugerir nuevos parámetros
        suggested_params = self.optimizer.suggest_next_params(
            self.parameter_history,
            self.performance_history
        )
        
        # Aplicar constraints
        validated_params = self._validate_parameters(suggested_params)
        
        return validated_params

class StrategySelector:
    """Selecciona mejores estrategias según condiciones"""
    
    def __init__(self, available_strategies: List[BaseStrategy]):
        self.strategies = {s.config.name: s for s in available_strategies}
        self.performance_tracker = StrategyPerformanceTracker()
        
    def select_strategies(self, market_conditions: MarketConditions,
                         max_strategies: int = 3) -> List[str]:
        """Selecciona estrategias óptimas para condiciones actuales"""
        
        strategy_scores = {}
        
        for name, strategy in self.strategies.items():
            # Score basado en performance histórico en condiciones similares
            historical_score = self.performance_tracker.get_performance_in_conditions(
                name, market_conditions
            )
            
            # Score basado en idoneidad para condiciones actuales
            suitability_score = self._calculate_suitability(strategy, market_conditions)
            
            # Score combinado
            strategy_scores[name] = 0.7 * historical_score + 0.3 * suitability_score
        
        # Seleccionar top estrategias
        sorted_strategies = sorted(strategy_scores.items(), 
                                 key=lambda x: x[1], reverse=True)
        
        selected = [name for name, score in sorted_strategies[:max_strategies]]
        
        logger.info(f"Estrategias seleccionadas: {selected}")
        
        return selected

class EmergencyController:
    """Maneja situaciones de emergencia"""
    
    def __init__(self):
        self.emergency_conditions = {
            'flash_crash': self._detect_flash_crash,
            'connection_loss': self._detect_connection_loss,
            'abnormal_spread': self._detect_abnormal_spread,
            'margin_call_risk': self._detect_margin_call_risk
        }
        
    def check_emergency_conditions(self, system_data: Dict[str, Any]) -> Optional[str]:
        """Verifica condiciones de emergencia"""
        
        for condition_name, detector_func in self.emergency_conditions.items():
            if detector_func(system_data):
                logger.critical(f"CONDICIÓN DE EMERGENCIA DETECTADA: {condition_name}")
                return condition_name
        
        return None
    
    def execute_emergency_protocol(self, emergency_type: str):
        """Ejecuta protocolo de emergencia"""
        protocols = {
            'flash_crash': self._flash_crash_protocol,
            'connection_loss': self._connection_loss_protocol,
            'abnormal_spread': self._abnormal_spread_protocol,
            'margin_call_risk': self._margin_call_protocol
        }
        
        protocol = protocols.get(emergency_type)
        if protocol:
            protocol()
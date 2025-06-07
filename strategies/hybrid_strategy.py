from strategies.base_strategy import BaseStrategy
from strategies.ml_strategy import MLStrategy
import talib
from typing import Dict, List

@dataclass
class HybridSignal:
    """Señal híbrida con componentes técnicos y ML"""
    technical_signal: SignalType
    technical_confidence: float
    technical_indicators: Dict[str, float]
    
    ml_signal: SignalType
    ml_confidence: float
    ml_models_used: List[str]
    
    combined_signal: SignalType
    combined_confidence: float
    combination_method: str
    
    weight_technical: float
    weight_ml: float
    
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)

class HybridStrategy(BaseStrategy):
    """Estrategia híbrida que combina análisis técnico con ML"""
    
    def __init__(self, config: StrategyConfig, model_hub: ModelHub):
        super().__init__(config, model_hub)
        
        # Componentes de la estrategia
        self.technical_analyzer = TechnicalAnalyzer()
        self.ml_predictor = MLPredictor(model_hub)
        
        # Sistema de ponderación
        self.weight_optimizer = WeightOptimizer()
        self.current_weights = {
            'technical': 0.5,
            'ml': 0.5
        }
        
        # Combinadores de señales
        self.signal_combiners = {
            'weighted_average': self._weighted_average_combination,
            'voting': self._voting_combination,
            'bayesian': self._bayesian_combination,
            'fuzzy_logic': self._fuzzy_logic_combination
        }
        
        # Configuración
        self.combination_method = 'bayesian'
        self.require_confirmation = True
        self.min_agreement_threshold = 0.7
        
        # Historial para análisis
        self.signal_history = deque(maxlen=500)
        self.performance_by_method = {}
        
    def _generate_signal_logic(self, data: pd.DataFrame, 
                              market_conditions: Dict[str, Any]) -> TradingSignal:
        """Genera señal combinando análisis técnico y ML"""
        
        # 1. Generar señal técnica
        technical_analysis = self._perform_technical_analysis(data)
        
        # 2. Generar predicción ML
        ml_prediction = self._perform_ml_prediction(data, market_conditions)
        
        # 3. Evaluar confiabilidad de cada enfoque
        technical_reliability = self._evaluate_technical_reliability(
            data, market_conditions
        )
        ml_reliability = self._evaluate_ml_reliability(
            ml_prediction, market_conditions
        )
        
        # 4. Ajustar pesos dinámicamente
        self._update_weights(technical_reliability, ml_reliability)
        
        # 5. Combinar señales
        hybrid_signal = self._combine_signals(
            technical_analysis, ml_prediction, market_conditions
        )
        
        # 6. Aplicar filtros y confirmaciones
        validated_signal = self._validate_hybrid_signal(
            hybrid_signal, data, market_conditions
        )
        
        # 7. Generar señal final
        final_signal = self._create_trading_signal(validated_signal, data)
        
        # Registrar para análisis
        self._record_signal_generation(
            technical_analysis, ml_prediction, hybrid_signal, final_signal
        )
        
        return final_signal
    
    def _perform_technical_analysis(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Realiza análisis técnico completo"""
        
        analysis = {
            'signal': SignalType.NO_SIGNAL,
            'confidence': 0.0,
            'indicators': {},
            'patterns': [],
            'support_resistance': {}
        }
        
        # Indicadores de tendencia
        trend_indicators = self._analyze_trend_indicators(data)
        analysis['indicators']['trend'] = trend_indicators
        
        # Indicadores de momentum
        momentum_indicators = self._analyze_momentum_indicators(data)
        analysis['indicators']['momentum'] = momentum_indicators
        
        # Indicadores de volatilidad
        volatility_indicators = self._analyze_volatility_indicators(data)
        analysis['indicators']['volatility'] = volatility_indicators
        
        # Patrones de velas
        candlestick_patterns = self._detect_candlestick_patterns(data)
        analysis['patterns'] = candlestick_patterns
        
        # Niveles de soporte/resistencia
        sr_levels = self._identify_support_resistance(data)
        analysis['support_resistance'] = sr_levels
        
        # Generar señal técnica consolidada
        technical_signal = self._consolidate_technical_signals(analysis)
        analysis['signal'] = technical_signal['signal']
        analysis['confidence'] = technical_signal['confidence']
        
        return analysis
    
    def _analyze_trend_indicators(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analiza indicadores de tendencia"""
        
        current_idx = -1
        
        # Moving Averages
        ma_analysis = {
            'sma_20': data['sma_20'].iloc[current_idx],
            'ema_20': data['ema_20'].iloc[current_idx],
            'sma_50': data['sma_50'].iloc[current_idx],
            'sma_200': data['sma_200'].iloc[current_idx] if 'sma_200' in data else None
        }
        
        # Análisis de cruces
        current_price = data['close'].iloc[current_idx]
        ma_analysis['price_above_sma20'] = current_price > ma_analysis['sma_20']
        ma_analysis['sma20_above_sma50'] = ma_analysis['sma_20'] > ma_analysis['sma_50']
        
        # MACD
        if all(col in data.columns for col in ['macd', 'macd_signal']):
            ma_analysis['macd'] = data['macd'].iloc[current_idx]
            ma_analysis['macd_signal'] = data['macd_signal'].iloc[current_idx]
            ma_analysis['macd_histogram'] = data['macd_hist'].iloc[current_idx]
            ma_analysis['macd_bullish'] = ma_analysis['macd'] > ma_analysis['macd_signal']
        
        # ADX para fuerza de tendencia
        if 'adx' in data.columns:
            ma_analysis['adx'] = data['adx'].iloc[current_idx]
            ma_analysis['strong_trend'] = ma_analysis['adx'] > 25
        
        # Señal de tendencia
        bullish_signals = sum([
            ma_analysis.get('price_above_sma20', False),
            ma_analysis.get('sma20_above_sma50', False),
            ma_analysis.get('macd_bullish', False),
            ma_analysis.get('strong_trend', False)
        ])
        
        ma_analysis['trend_signal'] = 'bullish' if bullish_signals >= 3 else \
                                     'bearish' if bullish_signals <= 1 else 'neutral'
        
        return ma_analysis
    
    def _consolidate_technical_signals(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Consolida múltiples señales técnicas en una señal unificada"""
        
        signal_votes = {'buy': 0, 'sell': 0, 'neutral': 0}
        confidence_scores = []
        
        # Votos de indicadores de tendencia
        trend = analysis['indicators']['trend'].get('trend_signal', 'neutral')
        if trend == 'bullish':
            signal_votes['buy'] += 2
            confidence_scores.append(0.7)
        elif trend == 'bearish':
            signal_votes['sell'] += 2
            confidence_scores.append(0.7)
        
        # Votos de momentum
        momentum = analysis['indicators']['momentum']
        if momentum.get('rsi', 50) < 30:
            signal_votes['buy'] += 1
            confidence_scores.append(0.6)
        elif momentum.get('rsi', 50) > 70:
            signal_votes['sell'] += 1
            confidence_scores.append(0.6)
        
        # Votos de patrones
        for pattern in analysis['patterns']:
            if pattern['bullish']:
                signal_votes['buy'] += pattern['strength']
                confidence_scores.append(pattern['reliability'])
            else:
                signal_votes['sell'] += pattern['strength']
                confidence_scores.append(pattern['reliability'])
        
        # Determinar señal final
        total_votes = sum(signal_votes.values())
        if total_votes == 0:
            return {'signal': SignalType.NO_SIGNAL, 'confidence': 0.0}
        
        if signal_votes['buy'] > signal_votes['sell'] * 1.5:
            signal = SignalType.BUY
        elif signal_votes['sell'] > signal_votes['buy'] * 1.5:
            signal = SignalType.SELL
        else:
            signal = SignalType.NO_SIGNAL
        
        # Calcular confianza
        confidence = np.mean(confidence_scores) if confidence_scores else 0.0
        
        # Ajustar por calidad de señales
        signal_quality = max(signal_votes.values()) / total_votes
        confidence *= signal_quality
        
        return {
            'signal': signal,
            'confidence': confidence,
            'vote_distribution': signal_votes
        }
    
    def _perform_ml_prediction(self, data: pd.DataFrame,
                             market_conditions: Dict[str, Any]) -> Dict[str, Any]:
        """Genera predicción usando modelos ML"""
        
        if not self.model_hub:
            return {
                'signal': SignalType.NO_SIGNAL,
                'confidence': 0.0,
                'models_used': []
            }
        
        # Obtener modelos activos
        active_models = self.model_hub.get_active_models()
        
        predictions = []
        
        for model in active_models:
            try:
                # Preparar features para el modelo
                features = self._prepare_ml_features(data, model.required_features)
                
                # Obtener predicción
                prediction = model.predict(features.iloc[-1:])
                
                # Obtener probabilidades si están disponibles
                if hasattr(model, 'predict_proba'):
                    probabilities = model.predict_proba(features.iloc[-1:])
                    prob = probabilities[0, 1] if len(probabilities.shape) > 1 else probabilities[0]
                else:
                    prob = 0.5 + (prediction[0] - 0.5) * 0.3  # Estimación
                
                predictions.append({
                    'model': model.name,
                    'prediction': prediction[0],
                    'probability': prob,
                    'model_confidence': model.get_confidence_score()
                })
                
            except Exception as e:
                logger.warning(f"Error en predicción de {model.name}: {e}")
        
        # Combinar predicciones ML
        if predictions:
            combined_ml = self._combine_ml_predictions(predictions)
        else:
            combined_ml = {
                'signal': SignalType.NO_SIGNAL,
                'confidence': 0.0,
                'models_used': []
            }
        
        return combined_ml
    
    def _combine_signals(self, technical: Dict[str, Any],
                        ml: Dict[str, Any],
                        market_conditions: Dict[str, Any]) -> HybridSignal:
        """Combina señales técnicas y ML"""
        
        # Usar método de combinación configurado
        combiner = self.signal_combiners.get(
            self.combination_method,
            self._weighted_average_combination
        )
        
        hybrid_result = combiner(technical, ml, market_conditions)
        
        # Crear objeto HybridSignal
        hybrid_signal = HybridSignal(
            technical_signal=technical['signal'],
            technical_confidence=technical['confidence'],
            technical_indicators=technical['indicators'],
            ml_signal=ml['signal'],
            ml_confidence=ml['confidence'],
            ml_models_used=ml.get('models_used', []),
            combined_signal=hybrid_result['signal'],
            combined_confidence=hybrid_result['confidence'],
            combination_method=self.combination_method,
            weight_technical=self.current_weights['technical'],
            weight_ml=self.current_weights['ml'],
            timestamp=datetime.now(),
            metadata=hybrid_result.get('metadata', {})
        )
        
        return hybrid_signal
    
    def _bayesian_combination(self, technical: Dict[str, Any],
                            ml: Dict[str, Any],
                            market_conditions: Dict[str, Any]) -> Dict[str, Any]:
        """Combinación Bayesiana de señales"""
        
        # Priors basados en performance histórico
        prior_technical = self.current_weights['technical']
        prior_ml = self.current_weights['ml']
        
        # Likelihoods basados en confianza actual
        likelihood_technical = technical['confidence']
        likelihood_ml = ml['confidence']
        
        # Calcular posteriors
        evidence = (prior_technical * likelihood_technical + 
                   prior_ml * likelihood_ml)
        
        if evidence == 0:
            return {
                'signal': SignalType.NO_SIGNAL,
                'confidence': 0.0
            }
        
        posterior_technical = (prior_technical * likelihood_technical) / evidence
        posterior_ml = (prior_ml * likelihood_ml) / evidence
        
        # Determinar señal basada en posteriors
        if technical['signal'] == ml['signal']:
            # Acuerdo entre métodos
            combined_signal = technical['signal']
            combined_confidence = (posterior_technical * technical['confidence'] +
                                 posterior_ml * ml['confidence'])
        else:
            # Desacuerdo - usar el de mayor posterior
            if posterior_technical > posterior_ml:
                combined_signal = technical['signal']
                combined_confidence = technical['confidence'] * posterior_technical
            else:
                combined_signal = ml['signal']
                combined_confidence = ml['confidence'] * posterior_ml
        
        return {
            'signal': combined_signal,
            'confidence': combined_confidence,
            'metadata': {
                'posterior_technical': posterior_technical,
                'posterior_ml': posterior_ml,
                'evidence': evidence
            }
        }
    
    def _fuzzy_logic_combination(self, technical: Dict[str, Any],
                               ml: Dict[str, Any],
                               market_conditions: Dict[str, Any]) -> Dict[str, Any]:
        """Combinación usando lógica difusa"""
        
        # Definir funciones de membresía
        def confidence_membership(conf):
            """Función de membresía para confianza"""
            if conf < 0.3:
                return {'low': 1.0, 'medium': 0.0, 'high': 0.0}
            elif conf < 0.6:
                low = (0.6 - conf) / 0.3
                medium = (conf - 0.3) / 0.3
                return {'low': low, 'medium': medium, 'high': 0.0}
            elif conf < 0.8:
                medium = (0.8 - conf) / 0.2
                high = (conf - 0.6) / 0.2
                return {'low': 0.0, 'medium': medium, 'high': high}
            else:
                return {'low': 0.0, 'medium': 0.0, 'high': 1.0}
        
        # Obtener membresías
        tech_membership = confidence_membership(technical['confidence'])
        ml_membership = confidence_membership(ml['confidence'])
        
        # Reglas difusas
        rules = []
        
        # Si ambos tienen alta confianza y coinciden, señal muy fuerte
        if technical['signal'] == ml['signal']:
            strength = min(tech_membership['high'], ml_membership['high'])
            if strength > 0:
                rules.append({
                    'signal': technical['signal'],
                    'strength': strength,
                    'confidence': 0.9
                })
        
        # Si uno tiene alta confianza y otro media, seguir al de alta
        if tech_membership['high'] > ml_membership['high']:
            strength = tech_membership['high'] * (1 - ml_membership['low'])
            rules.append({
                'signal': technical['signal'],
                'strength': strength,
                'confidence': technical['confidence'] * 0.8
            })
        elif ml_membership['high'] > tech_membership['high']:
            strength = ml_membership['high'] * (1 - tech_membership['low'])
            rules.append({
                'signal': ml['signal'],
                'strength': strength,
                'confidence': ml['confidence'] * 0.8
            })
        
        # Defuzzificación
        if not rules:
            return {
                'signal': SignalType.NO_SIGNAL,
                'confidence': 0.0
            }
        
        # Seleccionar regla con mayor fuerza
        best_rule = max(rules, key=lambda r: r['strength'])
        
        return {
            'signal': best_rule['signal'],
            'confidence': best_rule['confidence'],
            'metadata': {
                'fuzzy_rules_activated': len(rules),
                'winning_rule_strength': best_rule['strength']
            }
        }
    
    def _update_weights(self, technical_reliability: float,
                       ml_reliability: float):
        """Actualiza pesos de combinación dinámicamente"""
        
        # Calcular nuevos pesos basados en reliability
        total_reliability = technical_reliability + ml_reliability
        
        if total_reliability > 0:
            new_technical_weight = technical_reliability / total_reliability
            new_ml_weight = ml_reliability / total_reliability
        else:
            new_technical_weight = 0.5
            new_ml_weight = 0.5
        
        # Suavizar cambios
        alpha = 0.1  # Factor de suavizado
        
        self.current_weights['technical'] = (
            alpha * new_technical_weight + 
            (1 - alpha) * self.current_weights['technical']
        )
        
        self.current_weights['ml'] = (
            alpha * new_ml_weight + 
            (1 - alpha) * self.current_weights['ml']
        )
        
        # Normalizar
        total = sum(self.current_weights.values())
        self.current_weights = {k: v/total for k, v in self.current_weights.items()}

class PatternRecognizer:
    """Reconocedor avanzado de patrones técnicos"""
    
    def __init__(self):
        self.patterns = {
            'head_and_shoulders': self._detect_head_shoulders,
            'double_top': self._detect_double_top,
            'double_bottom': self._detect_double_bottom,
            'triangle': self._detect_triangle,
            'flag': self._detect_flag,
            'wedge': self._detect_wedge
        }
        
    def detect_all_patterns(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detecta todos los patrones en los datos"""
        detected_patterns = []
        
        for pattern_name, detector_func in self.patterns.items():
            try:
                pattern = detector_func(data)
                if pattern['detected']:
                    detected_patterns.append({
                        'name': pattern_name,
                        'bullish': pattern['bullish'],
                        'strength': pattern['strength'],
                        'reliability': pattern['reliability'],
                        'target_price': pattern.get('target_price'),
                        'stop_loss': pattern.get('stop_loss')
                    })
            except Exception as e:
                logger.debug(f"Error detectando {pattern_name}: {e}")
        
        return detected_patterns

class MLPredictor:
    """Predictor ML especializado para estrategia híbrida"""
    
    def __init__(self, model_hub: ModelHub):
        self.model_hub = model_hub
        self.feature_engineer = FeatureEngineer()
        self.prediction_cache = {}
        
    def get_ensemble_prediction(self, features: pd.DataFrame) -> Dict[str, Any]:
        """Obtiene predicción de ensemble de modelos"""
        
        models = self.model_hub.get_models_by_performance(top_n=5)
        
        if not models:
            return {
                'signal': SignalType.NO_SIGNAL,
                'confidence': 0.0,
                'models_used': []
            }
        
        predictions = []
        weights = []
        
        for model in models:
            pred = model.predict(features)
            conf = model.get_recent_performance()
            
            predictions.append(pred)
            weights.append(conf)
        
        # Weighted voting
        weighted_sum = sum(p * w for p, w in zip(predictions, weights))
        total_weight = sum(weights)
        
        if total_weight > 0:
            final_prediction = weighted_sum / total_weight
            
            # Convertir a señal
            if final_prediction > 0.6:
                signal = SignalType.BUY
            elif final_prediction < 0.4:
                signal = SignalType.SELL
            else:
                signal = SignalType.NO_SIGNAL
            
            confidence = abs(final_prediction - 0.5) * 2
        else:
            signal = SignalType.NO_SIGNAL
            confidence = 0.0
        
        return {
            'signal': signal,
            'confidence': confidence,
            'models_used': [m.name for m in models],
            'raw_prediction': final_prediction
        }
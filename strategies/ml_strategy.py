'''
15. strategies/ml_strategy.py
Ruta: TradingBot_Cuantitative_MT5/strategies/ml_strategy.py
Resumen:

Estrategia de trading basada en Machine Learning que hereda de BaseStrategy
Integra con ModelHub para predicciones inteligentes y combina múltiples modelos ML
Aplica filtros de confianza adaptativos y re-entrena modelos según performance
Incluye features específicas como análisis de sentimiento y detección de regímenes de mercado
'''
# strategies/ml_strategy.py
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Local imports
from strategies.base_strategy import (
    BaseStrategy, TradingSignal, SignalType, SignalStrength, 
    StrategyConfig, StrategyState
)
from models.model_hub import ModelHub
from models.base_model import ModelType, BaseModel
from utils.log_config import get_logger, log_performance

# Sklearn imports
try:
    from sklearn.preprocessing import StandardScaler, RobustScaler
    from sklearn.ensemble import IsolationForest
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

logger = get_logger('strategies')


class MLStrategyConfig(StrategyConfig):
    """Configuración específica para estrategia ML"""
    
    def __init__(self, **kwargs):
        # Configuración base
        super().__init__(**kwargs)
        
        # Configuración específica de ML
        self.model_types = kwargs.get('model_types', ['random_forest', 'xgboost', 'lightgbm'])
        self.use_ensemble = kwargs.get('use_ensemble', True)
        self.ensemble_method = kwargs.get('ensemble_method', 'voting')  # 'voting' or 'stacking'
        
        # Predicciones y confianza
        self.prediction_horizon = kwargs.get('prediction_horizon', 5)  # Barras hacia adelante
        self.confidence_method = kwargs.get('confidence_method', 'probability')  # 'probability' or 'consensus'
        self.adaptive_threshold = kwargs.get('adaptive_threshold', True)
        self.min_model_agreement = kwargs.get('min_model_agreement', 0.6)  # 60% de modelos deben coincidir
        
        # Re-entrenamiento
        self.retrain_frequency = kwargs.get('retrain_frequency', 'weekly')  # 'daily', 'weekly', 'monthly'
        self.retrain_trigger_accuracy = kwargs.get('retrain_trigger_accuracy', 0.6)  # Re-entrenar si accuracy < 60%
        self.max_data_age_days = kwargs.get('max_data_age_days', 30)
        
        # Features específicas
        self.use_technical_features = kwargs.get('use_technical_features', True)
        self.use_market_regime_features = kwargs.get('use_market_regime_features', True)
        self.use_volatility_features = kwargs.get('use_volatility_features', True)
        self.use_sentiment_features = kwargs.get('use_sentiment_features', False)
        
        # Filtros avanzados
        self.market_regime_filter = kwargs.get('market_regime_filter', True)
        self.volatility_filter = kwargs.get('volatility_filter', True)
        self.outlier_detection = kwargs.get('outlier_detection', True)
        
        # Performance tracking
        self.track_feature_importance = kwargs.get('track_feature_importance', True)
        self.track_prediction_accuracy = kwargs.get('track_prediction_accuracy', True)


class MLStrategy(BaseStrategy):
    """
    Estrategia de trading basada en Machine Learning
    
    Características:
    - Múltiples modelos ML con ensambles
    - Predicción adaptativa con confianza dinámica
    - Re-entrenamiento automático
    - Detección de regímenes de mercado
    - Filtros de outliers y volatilidad
    """
    
    def __init__(self, config: MLStrategyConfig, model_hub: ModelHub):
        """
        Inicializar estrategia ML
        
        Args:
            config: Configuración específica de ML
            model_hub: Hub de modelos ML
        """
        if not isinstance(config, MLStrategyConfig):
            raise TypeError("Config debe ser MLStrategyConfig")
        
        if not model_hub:
            raise ValueError("ModelHub es requerido para MLStrategy")
        
        super().__init__(config, model_hub)
        
        # Configuración específica
        self.ml_config = config
        
        # Modelos y predictores
        self.active_models: List[BaseModel] = []
        self.ensemble_model: Optional[BaseModel] = None
        self.scaler: Optional[StandardScaler] = None
        
        # Estado de mercado
        self.current_market_regime = "unknown"
        self.volatility_state = "normal"
        self.last_retrain_date: Optional[datetime] = None
        
        # Tracking de performance
        self.prediction_history: List[Dict[str, Any]] = []
        self.accuracy_tracker = {
            'correct_predictions': 0,
            'total_predictions': 0,
            'rolling_accuracy': []
        }
        
        # Features y datos
        self.feature_importance_history: List[Dict[str, float]] = []
        self.outlier_detector: Optional[IsolationForest] = None
        
        # Inicialización
        self._initialize_ml_components()
        
        logger.info(f"MLStrategy inicializada: {config.name}")
        logger.info(f"Modelos configurados: {config.model_types}")
        logger.info(f"Uso de ensemble: {config.use_ensemble}")
    
    # ==================== MÉTODOS ABSTRACTOS IMPLEMENTADOS ====================
    
    def _generate_signal_logic(self, data: pd.DataFrame, 
                              market_conditions: Dict[str, Any]) -> TradingSignal:
        """Generar señal usando modelos ML"""
        
        # Preparar features
        features = self._prepare_features(data)
        
        if features is None or features.empty:
            return self._create_no_signal("Features no disponibles")
        
        # Detectar outliers
        if self.ml_config.outlier_detection and self._is_outlier(features.iloc[-1:]):
            return self._create_no_signal("Condiciones de mercado atípicas detectadas")
        
        # Detectar régimen de mercado
        self._update_market_regime(data, features)
        
        # Filtrar por régimen de mercado
        if self.ml_config.market_regime_filter and not self._is_favorable_regime():
            return self._create_no_signal(f"Régimen de mercado desfavorable: {self.current_market_regime}")
        
        # Verificar modelos disponibles
        if not self.active_models:
            self._load_or_train_models(data, features)
        
        if not self.active_models:
            return self._create_no_signal("No hay modelos disponibles")
        
        # Generar predicciones
        predictions = self._generate_predictions(features.iloc[-1:])
        
        if not predictions:
            return self._create_no_signal("Error generando predicciones")
        
        # Combinar predicciones
        signal_info = self._combine_predictions(predictions, data.iloc[-1])
        
        # Crear señal
        signal = self._create_signal_from_prediction(signal_info, data.iloc[-1])
        
        # Tracking de predicción
        self._track_prediction(signal_info, data.iloc[-1])
        
        return signal
    
    def _validate_signal(self, signal: TradingSignal, data: pd.DataFrame) -> bool:
        """Validar señal ML"""
        
        # Validación base
        if signal.confidence < self.ml_config.min_confidence_threshold:
            return False
        
        # Verificar acuerdo entre modelos
        if hasattr(signal, 'model_agreement'):
            if signal.model_agreement < self.ml_config.min_model_agreement:
                logger.debug(f"Acuerdo insuficiente entre modelos: {signal.model_agreement:.2%}")
                return False
        
        # Filtros de volatilidad
        if self.ml_config.volatility_filter:
            current_volatility = self._calculate_current_volatility(data)
            
            if current_volatility > self._get_volatility_threshold():
                logger.debug(f"Volatilidad muy alta: {current_volatility:.4f}")
                return False
        
        # Verificar calidad de predicción reciente
        if self.ml_config.track_prediction_accuracy:
            recent_accuracy = self._get_recent_accuracy()
            
            if recent_accuracy < self.ml_config.retrain_trigger_accuracy:
                logger.warning(f"Accuracy reciente baja: {recent_accuracy:.2%}. Considerando re-entrenamiento.")
                self._schedule_retrain()
        
        return True
    
    def get_required_features(self) -> List[str]:
        """Obtener features requeridas para la estrategia ML"""
        required = ['open', 'high', 'low', 'close', 'tick_volume']
        
        if self.ml_config.use_technical_features:
            required.extend([
                'rsi_14', 'macd', 'bb_upper_20', 'bb_lower_20',
                'sma_20', 'ema_20', 'atr_14'
            ])
        
        if self.ml_config.use_volatility_features:
            required.extend([
                'realized_vol_20', 'parkinson_vol_20'
            ])
        
        if self.ml_config.use_market_regime_features:
            required.extend([
                'momentum_20', 'trend_strength', 'market_balance_20'
            ])
        
        return required
    
    # ==================== PREPARACIÓN DE FEATURES ====================
    
    def _prepare_features(self, data: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Preparar features para modelos ML"""
        try:
            features_df = data.copy()
            
            # Features técnicas adicionales
            if self.ml_config.use_technical_features:
                features_df = self._add_technical_features(features_df)
            
            # Features de régimen de mercado
            if self.ml_config.use_market_regime_features:
                features_df = self._add_market_regime_features(features_df)
            
            # Features de volatilidad
            if self.ml_config.use_volatility_features:
                features_df = self._add_volatility_features(features_df)
            
            # Features de sentimiento (si están disponibles)
            if self.ml_config.use_sentiment_features:
                features_df = self._add_sentiment_features(features_df)
            
            # Seleccionar solo columnas numéricas
            numeric_features = features_df.select_dtypes(include=[np.number])
            
            # Remover infinitos y NaN
            numeric_features = numeric_features.replace([np.inf, -np.inf], np.nan)
            numeric_features = numeric_features.fillna(method='ffill').fillna(0)
            
            # Normalizar si es necesario
            if self.scaler is None and len(numeric_features) > 100:
                self.scaler = RobustScaler()
                self.scaler.fit(numeric_features.iloc[:-1])  # No incluir última fila para evitar look-ahead
            
            if self.scaler is not None:
                scaled_features = pd.DataFrame(
                    self.scaler.transform(numeric_features),
                    index=numeric_features.index,
                    columns=numeric_features.columns
                )
                return scaled_features
            
            return numeric_features
            
        except Exception as e:
            logger.error(f"Error preparando features: {e}")
            return None
    
    def _add_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Agregar features técnicas avanzadas"""
        # Momentos de precio
        for window in [5, 10, 20]:
            df[f'price_momentum_{window}'] = df['close'].pct_change(window)
            df[f'volume_momentum_{window}'] = df['tick_volume'].pct_change(window)
        
        # Ratios de precio
        df['high_low_ratio'] = df['high'] / df['low']
        df['close_open_ratio'] = df['close'] / df['open']
        
        # Rangos normalizados
        df['true_range'] = np.maximum(
            df['high'] - df['low'],
            np.maximum(
                abs(df['high'] - df['close'].shift(1)),
                abs(df['low'] - df['close'].shift(1))
            )
        )
        
        # Volumen relativo
        df['volume_sma_ratio'] = df['tick_volume'] / df['tick_volume'].rolling(20).mean()
        
        return df
    
    def _add_market_regime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Agregar features de régimen de mercado"""
        # Tendencia
        for window in [10, 20, 50]:
            df[f'trend_{window}'] = (df['close'] - df['close'].shift(window)) / df['close'].shift(window)
        
        # Volatilidad rolling
        for window in [5, 10, 20]:
            df[f'volatility_{window}'] = df['close'].pct_change().rolling(window).std()
        
        # Correlación precio-volumen
        for window in [10, 20]:
            df[f'price_volume_corr_{window}'] = df['close'].rolling(window).corr(df['tick_volume'])
        
        # Detección de breakouts
        for window in [20, 50]:
            df[f'breakout_up_{window}'] = (df['high'] > df['high'].rolling(window).max().shift(1)).astype(int)
            df[f'breakout_down_{window}'] = (df['low'] < df['low'].rolling(window).min().shift(1)).astype(int)
        
        return df
    
    def _add_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Agregar features de volatilidad"""
        # Diferentes medidas de volatilidad
        returns = df['close'].pct_change()
        
        # Volatilidad GARCH simplificada
        for window in [5, 10, 20]:
            df[f'garch_vol_{window}'] = returns.rolling(window).std() * np.sqrt(252)
        
        # Volatilidad de Parkinson
        df['parkinson_vol'] = np.sqrt(
            (1 / (4 * np.log(2))) * (np.log(df['high'] / df['low']) ** 2)
        ) * np.sqrt(252)
        
        # VIX sintético (proxy)
        df['synthetic_vix'] = df['parkinson_vol'].rolling(20).mean() * 100
        
        return df
    
    def _add_sentiment_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Agregar features de sentimiento (placeholder)"""
        # Placeholder para features de sentimiento
        # En implementación real, podría incluir datos de news, social media, etc.
        
        # Por ahora, usar proxies basados en precios
        df['sentiment_proxy'] = df['close'].pct_change().rolling(20).sum()
        df['fear_greed_index'] = ((df['close'] - df['close'].rolling(50).min()) / 
                                 (df['close'].rolling(50).max() - df['close'].rolling(50).min())) * 100
        
        return df
    
    # ==================== PREDICCIONES Y MODELOS ====================
    
    def _load_or_train_models(self, data: pd.DataFrame, features: pd.DataFrame):
        """Cargar modelos existentes o entrenar nuevos"""
        
        # Intentar cargar modelos del hub
        self.active_models = []
        
        for model_type in self.ml_config.model_types:
            models = self.model_hub.list_models({'type': 'CLASSIFIER'})
            model_found = False
            
            for model_info in models:
                if model_type in model_info['name'].lower():
                    model = self.model_hub.get_model(model_info['name'])
                    if model and model.info.status.value == 'TRAINED':
                        self.active_models.append(model)
                        model_found = True
                        logger.info(f"Modelo cargado: {model_info['name']}")
                        break
            
            if not model_found:
                logger.info(f"Entrenando nuevo modelo: {model_type}")
                new_model = self._train_new_model(model_type, data, features)
                if new_model:
                    self.active_models.append(new_model)
        
        # Crear ensemble si está configurado
        if self.ml_config.use_ensemble and len(self.active_models) >= 2:
            self._create_ensemble()
    
    def _train_new_model(self, model_type: str, data: pd.DataFrame, 
                        features: pd.DataFrame) -> Optional[BaseModel]:
        """Entrenar nuevo modelo"""
        try:
            # Preparar datos de entrenamiento
            X, y = self._prepare_training_data(data, features)
            
            if X is None or len(X) < 100:
                logger.warning(f"Datos insuficientes para entrenar {model_type}")
                return None
            
            # Crear modelo
            from models.ml_models import MLModelFactory
            model_name = f"ml_strategy_{model_type}_{datetime.now().strftime('%Y%m%d_%H%M')}"
            
            model = MLModelFactory.create_model(model_type, model_name, 'classification')
            
            # Dividir en entrenamiento y validación
            split_idx = int(len(X) * 0.8)
            X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]
            
            # Entrenar
            model.fit(X_train, y_train, validation_data=(X_val, y_val))
            
            # Registrar en hub
            self.model_hub.register_model(
                model,
                tags=['ml_strategy', model_type, 'auto_trained'],
                auto_retrain=True
            )
            
            logger.info(f"✅ Modelo entrenado: {model_name}")
            return model
            
        except Exception as e:
            logger.error(f"Error entrenando modelo {model_type}: {e}")
            return None
    
    def _prepare_training_data(self, data: pd.DataFrame, 
                             features: pd.DataFrame) -> Tuple[Optional[pd.DataFrame], Optional[pd.Series]]:
        """Preparar datos para entrenamiento"""
        try:
            # Crear target (predicción de movimiento futuro)
            future_returns = data['close'].pct_change(self.ml_config.prediction_horizon).shift(-self.ml_config.prediction_horizon)
            
            # Clasificación binaria: 1 = subida, 0 = bajada
            target = (future_returns > 0).astype(int)
            
            # Alinear features y target
            valid_idx = target.dropna().index
            X = features.loc[valid_idx]
            y = target.loc[valid_idx]
            
            # Verificar balance de clases
            class_balance = y.mean()
            if class_balance < 0.3 or class_balance > 0.7:
                logger.warning(f"Desbalance de clases detectado: {class_balance:.2%} positivos")
            
            logger.info(f"Datos de entrenamiento preparados: {len(X)} muestras")
            logger.info(f"Balance de clases: {class_balance:.2%} positivos")
            
            return X, y
            
        except Exception as e:
            logger.error(f"Error preparando datos de entrenamiento: {e}")
            return None, None
    
    def _create_ensemble(self):
        """Crear modelo ensemble"""
        try:
            from models.ensemble_models import EnsembleFactory
            
            ensemble_name = f"ml_strategy_ensemble_{datetime.now().strftime('%Y%m%d_%H%M')}"
            
            if self.ml_config.ensemble_method == 'voting':
                self.ensemble_model = EnsembleFactory.create_voting_ensemble(
                    base_models=self.active_models,
                    name=ensemble_name,
                    voting_type='soft'
                )
            elif self.ml_config.ensemble_method == 'stacking':
                self.ensemble_model = EnsembleFactory.create_stacking_ensemble(
                    base_models=self.active_models,
                    name=ensemble_name,
                    meta_model_type='logistic'
                )
            
            # Registrar ensemble
            self.model_hub.register_model(
                self.ensemble_model,
                tags=['ml_strategy', 'ensemble', 'auto_created'],
                is_production=True
            )
            
            logger.info(f"✅ Ensemble creado: {ensemble_name}")
            
        except Exception as e:
            logger.error(f"Error creando ensemble: {e}")
    
    def _generate_predictions(self, features: pd.DataFrame) -> List[Dict[str, Any]]:
        """Generar predicciones de todos los modelos"""
        predictions = []
        
        # Predicciones de modelos individuales
        for model in self.active_models:
            try:
                # Predicción de clase
                pred_class = model.predict(features)[0]
                
                # Probabilidades si están disponibles
                pred_proba = None
                if hasattr(model, 'predict_proba'):
                    probas = model.predict_proba(features)
                    pred_proba = probas[0, 1] if probas.shape[1] > 1 else probas[0, 0]
                
                predictions.append({
                    'model_name': model.info.name,
                    'prediction': pred_class,
                    'probability': pred_proba,
                    'model_type': 'individual'
                })
                
            except Exception as e:
                logger.warning(f"Error predicción de {model.info.name}: {e}")
        
        # Predicción del ensemble
        if self.ensemble_model:
            try:
                pred_class = self.ensemble_model.predict(features)[0]
                pred_proba = None
                
                if hasattr(self.ensemble_model, 'predict_proba'):
                    probas = self.ensemble_model.predict_proba(features)
                    pred_proba = probas[0, 1] if probas.shape[1] > 1 else probas[0, 0]
                
                predictions.append({
                    'model_name': self.ensemble_model.info.name,
                    'prediction': pred_class,
                    'probability': pred_proba,
                    'model_type': 'ensemble'
                })
                
            except Exception as e:
                logger.warning(f"Error predicción ensemble: {e}")
        
        return predictions
    
    def _combine_predictions(self, predictions: List[Dict[str, Any]], 
                           current_bar: pd.Series) -> Dict[str, Any]:
        """Combinar predicciones de múltiples modelos"""
        if not predictions:
            return {'signal': SignalType.NO_SIGNAL, 'confidence': 0.0}
        
        # Priorizar ensemble si existe
        ensemble_pred = [p for p in predictions if p['model_type'] == 'ensemble']
        if ensemble_pred:
            main_prediction = ensemble_pred[0]
        else:
            # Usar votación por mayoría
            pred_classes = [p['prediction'] for p in predictions if p['prediction'] is not None]
            
            if not pred_classes:
                return {'signal': SignalType.NO_SIGNAL, 'confidence': 0.0}
            
            # Votación
            votes_up = sum(pred_classes)
            votes_total = len(pred_classes)
            majority_vote = 1 if votes_up > votes_total / 2 else 0
            
            main_prediction = {
                'prediction': majority_vote,
                'probability': votes_up / votes_total
            }
        
        # Calcular confianza
        confidence = self._calculate_confidence(predictions, main_prediction)
        
        # Calcular acuerdo entre modelos
        model_agreement = self._calculate_model_agreement(predictions)
        
        # Determinar tipo de señal
        signal_type = SignalType.NO_SIGNAL
        
        if main_prediction['prediction'] == 1:
            signal_type = SignalType.BUY
        elif main_prediction['prediction'] == 0:
            signal_type = SignalType.SELL
        
        # Ajustar fuerza de la señal
        strength = SignalStrength.WEAK
        if confidence > 0.8:
            strength = SignalStrength.VERY_STRONG
        elif confidence > 0.7:
            strength = SignalStrength.STRONG
        elif confidence > 0.6:
            strength = SignalStrength.MODERATE
        
        return {
            'signal': signal_type,
            'confidence': confidence,
            'strength': strength,
            'model_agreement': model_agreement,
            'raw_predictions': predictions,
            'main_prediction': main_prediction
        }
    
    def _calculate_confidence(self, predictions: List[Dict[str, Any]], 
                            main_prediction: Dict[str, Any]) -> float:
        """Calcular confianza de la predicción"""
        
        if self.ml_config.confidence_method == 'probability':
            # Usar probabilidad del modelo principal
            if main_prediction.get('probability') is not None:
                prob = main_prediction['probability']
                # Convertir a confianza (distancia de 0.5)
                confidence = abs(prob - 0.5) * 2
                return min(confidence, 1.0)
        
        elif self.ml_config.confidence_method == 'consensus':
            # Calcular consenso entre modelos
            pred_classes = [p['prediction'] for p in predictions if p['prediction'] is not None]
            
            if not pred_classes:
                return 0.0
            
            majority_count = max(
                sum(1 for p in pred_classes if p == 1),
                sum(1 for p in pred_classes if p == 0)
            )
            
            consensus = majority_count / len(pred_classes)
            # Escalar consenso a confianza
            confidence = (consensus - 0.5) * 2
            return max(0.0, min(confidence, 1.0))
        
        # Fallback
        return 0.5
    
    def _calculate_model_agreement(self, predictions: List[Dict[str, Any]]) -> float:
        """Calcular acuerdo entre modelos"""
        pred_classes = [p['prediction'] for p in predictions if p['prediction'] is not None]
        
        if len(pred_classes) <= 1:
            return 1.0
        
        # Porcentaje de modelos que predicen la clase mayoritaria
        votes_up = sum(pred_classes)
        votes_down = len(pred_classes) - votes_up
        
        majority_votes = max(votes_up, votes_down)
        agreement = majority_votes / len(pred_classes)
        
        return agreement
    
    def _create_signal_from_prediction(self, signal_info: Dict[str, Any], 
                                     current_bar: pd.Series) -> TradingSignal:
        """Crear señal de trading desde predicción"""
        
        current_price = current_bar['close']
        
        signal = TradingSignal(
            signal_type=signal_info['signal'],
            confidence=signal_info['confidence'],
            strength=signal_info['strength'],
            price=current_price,
            timestamp=datetime.now(),
            strategy_name=self.config.name,
            reasoning=f"ML prediction: {signal_info['main_prediction']['prediction']} "
                     f"(agreement: {signal_info['model_agreement']:.2%})"
        )
        
        # Agregar metadatos específicos de ML
        signal.model_agreement = signal_info['model_agreement']
        signal.features_used = list(self.current_data.columns) if self.current_data is not None else []
        
        # Calcular SL/TP dinámicos basados en volatilidad
        atr = current_bar.get('atr_14', current_price * 0.001)  # Fallback a 0.1%
        
        if signal.signal_type == SignalType.BUY:
            signal.stop_loss = current_price - (atr * 2)
            signal.take_profit = current_price + (atr * 3)
        elif signal.signal_type == SignalType.SELL:
            signal.stop_loss = current_price + (atr * 2)
            signal.take_profit = current_price - (atr * 3)
        
        return signal
    
    # ==================== GESTIÓN DE MERCADO ====================
    
    def _update_market_regime(self, data: pd.DataFrame, features: pd.DataFrame):
        """Detectar y actualizar régimen de mercado"""
        try:
            # Análisis de tendencia
            recent_returns = data['close'].pct_change(20).iloc[-1]
            volatility = data['close'].pct_change().rolling(20).std().iloc[-1]
            
            # Clasificación simple de régimen
            if recent_returns > 0.02 and volatility < 0.015:
                regime = "bull_low_vol"
            elif recent_returns > 0.02 and volatility >= 0.015:
                regime = "bull_high_vol"
            elif recent_returns < -0.02 and volatility < 0.015:
                regime = "bear_low_vol"
            elif recent_returns < -0.02 and volatility >= 0.015:
                regime = "bear_high_vol"
            elif volatility < 0.01:
                regime = "sideways_low_vol"
            elif volatility >= 0.025:
                regime = "high_volatility"
            else:
                regime = "neutral"
            
            self.current_market_regime = regime
            
            # Actualizar estado de volatilidad
            if volatility < 0.01:
                self.volatility_state = "low"
            elif volatility > 0.025:
                self.volatility_state = "high"
            else:
                self.volatility_state = "normal"
                
        except Exception as e:
            logger.warning(f"Error actualizando régimen de mercado: {e}")
            self.current_market_regime = "unknown"
    
    def _is_favorable_regime(self) -> bool:
        """Verificar si el régimen actual es favorable para trading"""
        # Configurar regímenes favorables
        favorable_regimes = [
            "bull_low_vol", "bear_low_vol", "neutral", "sideways_low_vol"
        ]
        
        # Evitar trading en alta volatilidad extrema
        unfavorable_regimes = ["high_volatility"]
        
        return self.current_market_regime in favorable_regimes
    
    def _is_outlier(self, features: pd.DataFrame) -> bool:
        """Detectar condiciones de mercado atípicas"""
        if not SKLEARN_AVAILABLE or features.empty:
            return False
        
        try:
            # Inicializar detector si no existe
            if self.outlier_detector is None and len(self.current_data) > 100:
                # Usar datos históricos para entrenar detector
                historical_features = features.iloc[:-1]  # Excluir punto actual
                
                self.outlier_detector = IsolationForest(
                    contamination=0.1,  # 10% de outliers esperados
                    random_state=42
                )
                self.outlier_detector.fit(historical_features.fillna(0))
            
            if self.outlier_detector is not None:
                # Predecir outlier (-1 = outlier, 1 = normal)
                prediction = self.outlier_detector.predict(features.fillna(0))
                return prediction[0] == -1
            
            return False
            
        except Exception as e:
            logger.warning(f"Error detectando outliers: {e}")
            return False
    
    def _calculate_current_volatility(self, data: pd.DataFrame) -> float:
        """Calcular volatilidad actual"""
        returns = data['close'].pct_change()
        current_vol = returns.rolling(20).std().iloc[-1]
        return current_vol if not np.isnan(current_vol) else 0.0
    
    def _get_volatility_threshold(self) -> float:
        """Obtener threshold de volatilidad dinámico"""
        # Threshold adaptativo basado en régimen
        base_threshold = 0.02  # 2%
        
        if self.current_market_regime in ["high_volatility", "bear_high_vol"]:
            return base_threshold * 1.5
        elif self.current_market_regime in ["bull_low_vol", "sideways_low_vol"]:
            return base_threshold * 0.7
        else:
            return base_threshold
    
    # ==================== TRACKING Y PERFORMANCE ====================
    
    def _track_prediction(self, signal_info: Dict[str, Any], current_bar: pd.Series):
        """Registrar predicción para tracking posterior"""
        prediction_record = {
            'timestamp': datetime.now(),
            'signal_type': signal_info['signal'].value,
            'confidence': signal_info['confidence'],
            'model_agreement': signal_info['model_agreement'],
            'current_price': current_bar['close'],
            'market_regime': self.current_market_regime,
            'volatility_state': self.volatility_state,
            'prediction_horizon': self.ml_config.prediction_horizon
        }
        
        self.prediction_history.append(prediction_record)
        
        # Mantener historial limitado
        if len(self.prediction_history) > 500:
            self.prediction_history = self.prediction_history[-250:]
    
    def _get_recent_accuracy(self, lookback_days: int = 7) -> float:
        """Calcular accuracy reciente de predicciones"""
        if not self.prediction_history:
            return 0.5  # Neutral
        
        cutoff_date = datetime.now() - timedelta(days=lookback_days)
        recent_predictions = [
            p for p in self.prediction_history 
            if p['timestamp'] >= cutoff_date
        ]
        
        if len(recent_predictions) < 5:  # Mínimo para cálculo
            return 0.5
        
        # Simplificación: usar accuracy tracker global
        if self.accuracy_tracker['total_predictions'] > 0:
            return self.accuracy_tracker['correct_predictions'] / self.accuracy_tracker['total_predictions']
        
        return 0.5
    
    def _schedule_retrain(self):
        """Programar re-entrenamiento de modelos"""
        if self.last_retrain_date is None:
            self.last_retrain_date = datetime.now() - timedelta(days=30)
        
        days_since_retrain = (datetime.now() - self.last_retrain_date).days
        
        # Determinar si es momento de re-entrenar
        retrain_intervals = {
            'daily': 1,
            'weekly': 7,
            'monthly': 30
        }
        
        required_interval = retrain_intervals.get(self.ml_config.retrain_frequency, 7)
        
        if days_since_retrain >= required_interval:
            logger.info("Re-entrenamiento programado por baja performance")
            # En implementación real, esto dispararía re-entrenamiento asíncrono
            self._trigger_retrain()
    
    def _trigger_retrain(self):
        """Disparar re-entrenamiento de modelos"""
        logger.info("🔄 Iniciando re-entrenamiento de modelos...")
        
        try:
            # Obtener datos recientes
            if self.current_data is not None and len(self.current_data) > 200:
                recent_data = self.current_data.tail(1000)  # Últimos 1000 puntos
                features = self._prepare_features(recent_data)
                
                # Re-entrenar cada modelo
                retrained_models = []
                
                for model_type in self.ml_config.model_types:
                    retrained_model = self._train_new_model(model_type, recent_data, features)
                    if retrained_model:
                        retrained_models.append(retrained_model)
                
                # Actualizar modelos activos
                if retrained_models:
                    self.active_models = retrained_models
                    
                    # Crear nuevo ensemble
                    if self.ml_config.use_ensemble and len(retrained_models) >= 2:
                        self._create_ensemble()
                    
                    self.last_retrain_date = datetime.now()
                    logger.info(f"✅ Re-entrenamiento completado: {len(retrained_models)} modelos")
                    
                    # Log performance
                    log_performance({
                        'retrain_triggered': True,
                        'models_retrained': len(retrained_models),
                        'strategy_name': self.config.name,
                        'trigger_reason': 'low_accuracy'
                    })
                
        except Exception as e:
            logger.error(f"Error en re-entrenamiento: {e}")
    
    def update_prediction_accuracy(self, prediction_id: str, was_correct: bool):
        """Actualizar accuracy de predicción específica"""
        self.accuracy_tracker['total_predictions'] += 1
        
        if was_correct:
            self.accuracy_tracker['correct_predictions'] += 1
        
        # Mantener rolling accuracy
        self.accuracy_tracker['rolling_accuracy'].append(was_correct)
        
        if len(self.accuracy_tracker['rolling_accuracy']) > 100:
            self.accuracy_tracker['rolling_accuracy'] = self.accuracy_tracker['rolling_accuracy'][-50:]
    
    def get_ml_statistics(self) -> Dict[str, Any]:
        """Obtener estadísticas específicas de ML"""
        base_stats = self.get_strategy_statistics()
        
        ml_stats = {
            'models': {
                'active_models': len(self.active_models),
                'model_names': [m.info.name for m in self.active_models],
                'ensemble_enabled': self.ensemble_model is not None,
                'last_retrain': self.last_retrain_date.isoformat() if self.last_retrain_date else None
            },
            'market_state': {
                'current_regime': self.current_market_regime,
                'volatility_state': self.volatility_state
            },
            'prediction_accuracy': {
                'total_predictions': self.accuracy_tracker['total_predictions'],
                'correct_predictions': self.accuracy_tracker['correct_predictions'],
                'accuracy_rate': self._get_recent_accuracy()
            },
            'prediction_history_size': len(self.prediction_history)
        }
        
        # Combinar con estadísticas base
        return {**base_stats, 'ml_specific': ml_stats}


# Funciones de utilidad
def create_ml_strategy_config(name: str, **kwargs) -> MLStrategyConfig:
    """Crear configuración para estrategia ML"""
    return MLStrategyConfig(name=name, **kwargs)


def create_default_ml_strategy(model_hub: ModelHub) -> MLStrategy:
    """Crear estrategia ML con configuración por defecto"""
    config = MLStrategyConfig(
        name="default_ml_strategy",
        description="Estrategia ML con configuración balanceada",
        model_types=['random_forest', 'xgboost'],
        use_ensemble=True,
        adaptive_threshold=True
    )
    
    return MLStrategy(config, model_hub)
import numpy as np, pandas as pd
from typing import Dict, List, Any
from models.base_model import BaseModel
from sklearn.linear_model import SGDClassifier
from utils.log_config import get_logger

@dataclass
class DriftDetectionResult:
    """Resultado de detección de drift"""
    drift_detected: bool
    drift_type: str  # 'concept', 'data', 'gradual', 'sudden'
    drift_score: float
    confidence: float
    detection_timestamp: datetime
    affected_features: List[str]

class OnlineLearningSystem:
    """Sistema principal de aprendizaje online"""
    
    def __init__(self, model_hub: ModelHub, buffer_size: int = 1000):
        self.model_hub = model_hub
        self.buffer_size = buffer_size
        
        # Buffers de datos
        self.feature_buffer = deque(maxlen=buffer_size)
        self.label_buffer = deque(maxlen=buffer_size)
        self.prediction_buffer = deque(maxlen=buffer_size)
        
        # Detectores de drift
        self.drift_detectors = {
            'adwin': ADWINDriftDetector(),
            'ddm': DDMDriftDetector(),
            'eddm': EDDMDriftDetector(),
            'page_hinkley': PageHinkleyDetector()
        }
        
        # Modelos incrementales
        self.incremental_models = {}
        self._initialize_incremental_models()
        
        # Métricas de adaptación
        self.adaptation_metrics = AdaptationMetrics()
        
    def update_models(self, new_features: pd.DataFrame, 
                     new_labels: pd.Series,
                     update_strategy: str = 'incremental') -> Dict[str, Any]:
        """Actualiza modelos con nuevos datos"""
        
        # Agregar a buffers
        self._update_buffers(new_features, new_labels)
        
        # Detectar drift
        drift_result = self._detect_drift()
        
        # Decidir estrategia de actualización
        if drift_result.drift_detected:
            if drift_result.drift_type in ['sudden', 'concept']:
                update_strategy = 'retrain'
            else:
                update_strategy = 'incremental_aggressive'
        
        # Aplicar actualización
        update_results = {}
        
        if update_strategy == 'incremental':
            update_results = self._incremental_update()
        elif update_strategy == 'incremental_aggressive':
            update_results = self._aggressive_incremental_update()
        elif update_strategy == 'retrain':
            update_results = self._trigger_retraining()
        
        # Evaluar adaptación
        self.adaptation_metrics.record_update(update_results, drift_result)
        
        return {
            'strategy': update_strategy,
            'drift_detected': drift_result.drift_detected,
            'models_updated': update_results,
            'adaptation_score': self.adaptation_metrics.get_current_score()
        }
    
    def _incremental_update(self) -> Dict[str, Any]:
        """Actualización incremental estándar"""
        results = {}
        
        # Obtener datos del buffer
        X_batch = pd.DataFrame(list(self.feature_buffer))
        y_batch = pd.Series(list(self.label_buffer))
        
        for model_name, model in self.incremental_models.items():
            try:
                # Partial fit para modelos que lo soporten
                if hasattr(model, 'partial_fit'):
                    model.partial_fit(X_batch, y_batch)
                    results[model_name] = {'status': 'updated', 'method': 'partial_fit'}
                
                # Para modelos de ensemble, actualizar pesos
                elif hasattr(model, 'update_weights'):
                    new_weights = self._calculate_adaptive_weights(model_name)
                    model.update_weights(new_weights)
                    results[model_name] = {'status': 'updated', 'method': 'weight_update'}
                
                # Para redes neuronales, fine-tuning
                elif hasattr(model, 'fine_tune'):
                    model.fine_tune(X_batch, y_batch, epochs=1)
                    results[model_name] = {'status': 'updated', 'method': 'fine_tune'}
                    
            except Exception as e:
                logger.error(f"Error actualizando {model_name}: {e}")
                results[model_name] = {'status': 'failed', 'error': str(e)}
        
        return results

class DriftDetector:
    """Detector base de concept drift"""
    
    @abc.abstractmethod
    def detect_drift(self, error_stream: List[float]) -> DriftDetectionResult:
        pass

class ADWINDriftDetector(DriftDetector):
    """Adaptive Windowing para detección de drift"""
    
    def __init__(self, delta: float = 0.002):
        self.delta = delta
        self.window = []
        self.total = 0
        self.variance = 0
        self.n = 0
        
    def detect_drift(self, error_stream: List[float]) -> DriftDetectionResult:
        """Detecta drift usando ADWIN algorithm"""
        drift_detected = False
        drift_points = []
        
        for i, error in enumerate(error_stream):
            self._add_element(error)
            
            if self._check_drift():
                drift_detected = True
                drift_points.append(i)
                self._remove_oldest_elements()
        
        if drift_detected:
            drift_type = 'sudden' if len(drift_points) == 1 else 'gradual'
            confidence = self._calculate_confidence()
            
            return DriftDetectionResult(
                drift_detected=True,
                drift_type=drift_type,
                drift_score=len(drift_points) / len(error_stream),
                confidence=confidence,
                detection_timestamp=datetime.now(),
                affected_features=[]  # Se puede extender para detectar features específicas
            )
        
        return DriftDetectionResult(
            drift_detected=False,
            drift_type='none',
            drift_score=0.0,
            confidence=1.0,
            detection_timestamp=datetime.now(),
            affected_features=[]
        )
    
    def _check_drift(self) -> bool:
        """Verifica si hay drift en la ventana actual"""
        if len(self.window) < 2:
            return False
        
        # Dividir ventana y comparar medias
        for split_point in range(1, len(self.window)):
            n1 = split_point
            n2 = len(self.window) - split_point
            
            mean1 = sum(self.window[:split_point]) / n1
            mean2 = sum(self.window[split_point:]) / n2
            
            # Hoeffding bound
            epsilon = sqrt(0.5 * log(2.0 / self.delta) * (1.0/n1 + 1.0/n2))
            
            if abs(mean1 - mean2) > epsilon:
                return True
        
        return False

class IncrementalLearner:
    """Learner incremental base"""
    
    def __init__(self, base_model: BaseModel):
        self.base_model = base_model
        self.learning_rate = 0.01
        self.momentum = 0.9
        
    def partial_update(self, X_new: pd.DataFrame, y_new: pd.Series,
                      sample_weight: Optional[np.ndarray] = None):
        """Actualización parcial del modelo"""
        # Implementación específica según tipo de modelo

class EnsembleUpdater:
    """Actualiza pesos de ensemble dinámicamente"""
    
    def __init__(self, ensemble_model: EnsembleModel):
        self.ensemble = ensemble_model
        self.performance_window = deque(maxlen=100)
        
    def update_ensemble_weights(self, recent_predictions: Dict[str, np.ndarray],
                               true_labels: np.ndarray) -> np.ndarray:
        """Calcula nuevos pesos basados en performance reciente"""
        
        model_performances = {}
        
        for model_name, predictions in recent_predictions.items():
            # Calcular accuracy reciente
            accuracy = accuracy_score(true_labels, predictions)
            model_performances[model_name] = accuracy
        
        # Convertir performances a pesos (softmax)
        performances = np.array(list(model_performances.values()))
        weights = np.exp(performances) / np.sum(np.exp(performances))
        
        # Suavizar con pesos anteriores
        if hasattr(self.ensemble, 'weights'):
            alpha = 0.3  # Factor de suavizado
            weights = alpha * weights + (1 - alpha) * self.ensemble.weights
        
        return weights

class FeatureEvolver:
    """Evoluciona features basado en importancia cambiante"""
    
    def __init__(self, feature_engineer: FeatureEngineer):
        self.feature_engineer = feature_engineer
        self.feature_importance_history = deque(maxlen=50)
        self.feature_performance = {}
        
    def evolve_features(self, current_features: pd.DataFrame,
                       model_feedback: Dict[str, float]) -> pd.DataFrame:
        """Evoluciona features basado en feedback del modelo"""
        
        # Registrar importancia actual
        self.feature_importance_history.append(model_feedback)
        
        # Identificar features de bajo rendimiento
        low_performing = self._identify_low_performing_features()
        
        # Generar nuevas features candidatas
        new_feature_candidates = self._generate_new_features(current_features)
        
        # Evaluar candidatas
        best_new_features = self._evaluate_feature_candidates(
            new_feature_candidates, current_features
        )
        
        # Reemplazar features de bajo rendimiento
        evolved_features = self._replace_features(
            current_features, low_performing, best_new_features
        )
        
        return evolved_features

class AdaptationMetrics:
    """Métricas para evaluar la adaptación del sistema"""
    
    def __init__(self):
        self.update_history = []
        self.performance_before_after = []
        self.drift_episodes = []
        
    def record_update(self, update_results: Dict[str, Any],
                     drift_result: DriftDetectionResult):
        """Registra actualización y sus resultados"""
        self.update_history.append({
            'timestamp': datetime.now(),
            'update_results': update_results,
            'drift_detected': drift_result.drift_detected,
            'drift_type': drift_result.drift_type
        })
        
        if drift_result.drift_detected:
            self.drift_episodes.append(drift_result)
    
    def get_adaptation_report(self) -> Dict[str, Any]:
        """Genera reporte de adaptación del sistema"""
        if not self.update_history:
            return {'status': 'no_updates_yet'}
        
        total_updates = len(self.update_history)
        successful_updates = sum(
            1 for update in self.update_history 
            if all(r.get('status') == 'updated' for r in update['update_results'].values())
        )
        
        drift_frequency = len(self.drift_episodes) / total_updates if total_updates > 0 else 0
        
        # Tipos de drift más comunes
        drift_types = [d.drift_type for d in self.drift_episodes]
        drift_type_distribution = {
            drift_type: drift_types.count(drift_type) / len(drift_types) 
            for drift_type in set(drift_types)
        } if drift_types else {}
        
        return {
            'total_updates': total_updates,
            'successful_updates': successful_updates,
            'success_rate': successful_updates / total_updates,
            'drift_frequency': drift_frequency,
            'drift_type_distribution': drift_type_distribution,
            'avg_updates_per_day': self._calculate_update_frequency(),
            'adaptation_effectiveness': self._calculate_effectiveness_score()
        }

class OnlineBacktester:
    """Backtesting continuo para evaluar adaptaciones"""
    
    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self.results_buffer = deque(maxlen=window_size)
        
    def evaluate_online_performance(self, predictions: np.ndarray,
                                  actuals: np.ndarray,
                                  timestamps: List[datetime]) -> Dict[str, float]:
        """Evalúa performance en ventana deslizante"""
        
        # Agregar resultados al buffer
        for pred, actual, ts in zip(predictions, actuals, timestamps):
            self.results_buffer.append({
                'prediction': pred,
                'actual': actual,
                'timestamp': ts
            })
        
        # Calcular métricas en ventana
        if len(self.results_buffer) < 50:  # Mínimo para estadísticas
            return {}
        
        df_results = pd.DataFrame(list(self.results_buffer))
        
        # Métricas básicas
        accuracy = accuracy_score(df_results['actual'], df_results['prediction'])
        
        # Métricas temporales
        recent_accuracy = accuracy_score(
            df_results.tail(100)['actual'], 
            df_results.tail(100)['prediction']
        )
        
        # Tendencia de performance
        rolling_accuracy = df_results.rolling(50).apply(
            lambda x: accuracy_score(x['actual'], x['prediction'])
        )
        
        performance_trend = np.polyfit(
            range(len(rolling_accuracy.dropna())), 
            rolling_accuracy.dropna(), 
            1
        )[0]
        
        return {
            'overall_accuracy': accuracy,
            'recent_accuracy': recent_accuracy,
            'performance_trend': performance_trend,
            'is_improving': performance_trend > 0
        }
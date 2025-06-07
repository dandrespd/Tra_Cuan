'''
9. models/base_model.py
Ruta: TradingBot_Cuantitative_MT5/models/base_model.py
Resumen:

Clase abstracta BaseModel para todos los modelos de ML/DL del sistema de trading
Define la interfaz común con métodos abstractos fit(), predict(), save(), load()
Proporciona funcionalidades base como registro de métricas, validación, hiperparámetros
Integra completamente con el sistema de logging para registrar progreso y estadísticas
'''
# models/base_model.py
import abc
import joblib
import pickle
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
import warnings
from enum import Enum

from utils.log_config import get_logger, log_performance
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score
)

logger = get_logger('models')


class ModelType(Enum):
    """Tipos de modelos soportados"""
    CLASSIFIER = "CLASSIFIER"
    REGRESSOR = "REGRESSOR"
    ENSEMBLE = "ENSEMBLE"
    DEEP_LEARNING = "DEEP_LEARNING"
    TIME_SERIES = "TIME_SERIES"


class ModelStatus(Enum):
    """Estados del modelo"""
    CREATED = "CREATED"
    TRAINING = "TRAINING"
    TRAINED = "TRAINED"
    VALIDATING = "VALIDATING"
    VALIDATED = "VALIDATED"
    DEPLOYED = "DEPLOYED"
    ERROR = "ERROR"
    DEPRECATED = "DEPRECATED"


@dataclass
class ModelMetrics:
    """Métricas de rendimiento del modelo"""
    # Métricas de clasificación
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    
    # Métricas de regresión
    mse: Optional[float] = None
    mae: Optional[float] = None
    r2_score: Optional[float] = None
    
    # Métricas de trading específicas
    sharpe_ratio: Optional[float] = None
    max_drawdown: Optional[float] = None
    win_rate: Optional[float] = None
    profit_factor: Optional[float] = None
    
    # Métricas de entrenamiento
    training_loss: Optional[float] = None
    validation_loss: Optional[float] = None
    training_time: Optional[float] = None
    
    # Metadatos
    timestamp: datetime = field(default_factory=datetime.now)
    sample_size: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir métricas a diccionario"""
        return {
            'accuracy': self.accuracy,
            'precision': self.precision,
            'recall': self.recall,
            'f1_score': self.f1_score,
            'mse': self.mse,
            'mae': self.mae,
            'r2_score': self.r2_score,
            'sharpe_ratio': self.sharpe_ratio,
            'max_drawdown': self.max_drawdown,
            'win_rate': self.win_rate,
            'profit_factor': self.profit_factor,
            'training_loss': self.training_loss,
            'validation_loss': self.validation_loss,
            'training_time': self.training_time,
            'timestamp': self.timestamp.isoformat(),
            'sample_size': self.sample_size
        }


@dataclass
class ModelInfo:
    """Información completa del modelo"""
    name: str
    model_type: ModelType
    version: str
    description: str
    status: ModelStatus = ModelStatus.CREATED
    
    # Hiperparámetros
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    
    # Métricas
    metrics: Optional[ModelMetrics] = None
    
    # Metadatos
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    trained_at: Optional[datetime] = None
    
    # Información de datos
    feature_names: List[str] = field(default_factory=list)
    target_name: Optional[str] = None
    data_shape: Optional[Tuple[int, int]] = None
    
    # Archivos
    model_path: Optional[Path] = None
    metadata_path: Optional[Path] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir información a diccionario"""
        return {
            'name': self.name,
            'model_type': self.model_type.value,
            'version': self.version,
            'description': self.description,
            'status': self.status.value,
            'hyperparameters': self.hyperparameters,
            'metrics': self.metrics.to_dict() if self.metrics else None,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'trained_at': self.trained_at.isoformat() if self.trained_at else None,
            'feature_names': self.feature_names,
            'target_name': self.target_name,
            'data_shape': self.data_shape,
            'model_path': str(self.model_path) if self.model_path else None,
            'metadata_path': str(self.metadata_path) if self.metadata_path else None
        }


class BaseModel(abc.ABC):
    """
    Clase base abstracta para todos los modelos de ML/DL
    
    Proporciona funcionalidades comunes:
    - Interfaz estándar (fit, predict, save, load)
    - Manejo de métricas y logging
    - Serialización y versionado
    - Validación de datos
    """
    
    def __init__(self, name: str, model_type: ModelType, 
                 version: str = "1.0.0", description: str = ""):
        """
        Inicializar modelo base
        
        Args:
            name: Nombre único del modelo
            model_type: Tipo de modelo
            version: Versión del modelo
            description: Descripción del modelo
        """
        self.info = ModelInfo(
            name=name,
            model_type=model_type,
            version=version,
            description=description
        )
        
        # Modelo interno (será definido por subclases)
        self._model = None
        
        # Configuración
        self.random_state = 42
        self.verbose = True
        
        # Historial de entrenamiento
        self.training_history: List[Dict[str, Any]] = []
        
        # Cache de predicciones
        self._prediction_cache: Dict[str, Any] = {}
        
        logger.info(f"Modelo inicializado: {name} v{version} ({model_type.value})")
    
    # ==================== MÉTODOS ABSTRACTOS ====================
    
    @abc.abstractmethod
    def _build_model(self, **kwargs) -> Any:
        """
        Construir el modelo interno
        
        Returns:
            Instancia del modelo específico
        """
        pass
    
    @abc.abstractmethod
    def _fit_model(self, X: pd.DataFrame, y: pd.Series, 
                   validation_data: Optional[Tuple[pd.DataFrame, pd.Series]] = None,
                   **kwargs) -> Dict[str, Any]:
        """
        Entrenar el modelo interno
        
        Args:
            X: Features de entrenamiento
            y: Target de entrenamiento
            validation_data: Datos de validación (X_val, y_val)
            **kwargs: Parámetros adicionales
            
        Returns:
            Diccionario con métricas de entrenamiento
        """
        pass
    
    @abc.abstractmethod
    def _predict_model(self, X: pd.DataFrame, **kwargs) -> np.ndarray:
        """
        Hacer predicciones con el modelo interno
        
        Args:
            X: Features para predicción
            **kwargs: Parámetros adicionales
            
        Returns:
            Array de predicciones
        """
        pass
    
    @abc.abstractmethod
    def _save_model(self, filepath: Path) -> bool:
        """
        Guardar el modelo interno
        
        Args:
            filepath: Ruta donde guardar el modelo
            
        Returns:
            True si se guardó exitosamente
        """
        pass
    
    @abc.abstractmethod
    def _load_model(self, filepath: Path) -> bool:
        """
        Cargar el modelo interno
        
        Args:
            filepath: Ruta del modelo a cargar
            
        Returns:
            True si se cargó exitosamente
        """
        pass
    
    # ==================== MÉTODOS PÚBLICOS ====================
    
    def fit(self, X: pd.DataFrame, y: pd.Series,
            validation_data: Optional[Tuple[pd.DataFrame, pd.Series]] = None,
            **kwargs) -> 'BaseModel':
        """
        Entrenar el modelo
        
        Args:
            X: Features de entrenamiento
            y: Target de entrenamiento
            validation_data: Datos de validación (X_val, y_val)
            **kwargs: Parámetros adicionales
            
        Returns:
            Self para chaining
        """
        logger.info("="*60)
        logger.info(f"ENTRENANDO MODELO: {self.info.name}")
        logger.info("="*60)
        logger.info(f"Tipo: {self.info.model_type.value}")
        logger.info(f"Datos de entrenamiento: {X.shape}")
        logger.info(f"Target: {y.name if hasattr(y, 'name') else 'unknown'}")
        
        if validation_data:
            logger.info(f"Datos de validación: {validation_data[0].shape}")
        
        # Validar datos de entrada
        self._validate_training_data(X, y, validation_data)
        
        # Actualizar estado
        self.info.status = ModelStatus.TRAINING
        self.info.data_shape = X.shape
        self.info.feature_names = X.columns.tolist()
        self.info.target_name = y.name if hasattr(y, 'name') else None
        
        start_time = datetime.now()
        
        try:
            # Construir modelo si no existe
            if self._model is None:
                logger.info("Construyendo arquitectura del modelo...")
                self._model = self._build_model(**kwargs)
            
            # Entrenar modelo
            logger.info("Iniciando entrenamiento...")
            training_metrics = self._fit_model(X, y, validation_data, **kwargs)
            
            # Calcular tiempo de entrenamiento
            training_time = (datetime.now() - start_time).total_seconds()
            
            # Calcular métricas
            metrics = self._calculate_metrics(X, y, validation_data, training_time)
            metrics.update(training_metrics)
            
            # Actualizar información del modelo
            self.info.metrics = ModelMetrics(**metrics)
            self.info.status = ModelStatus.TRAINED
            self.info.trained_at = datetime.now()
            self.info.updated_at = datetime.now()
            
            # Registrar en historial
            self.training_history.append({
                'timestamp': datetime.now().isoformat(),
                'metrics': metrics,
                'data_shape': X.shape,
                'hyperparameters': self.info.hyperparameters.copy()
            })
            
            # Log de éxito
            logger.info("✅ ENTRENAMIENTO COMPLETADO")
            logger.info(f"Tiempo: {training_time:.2f} segundos")
            
            # Log métricas
            self._log_metrics(metrics)
            
            # Performance log
            log_performance({
                'model_name': self.info.name,
                'training_time': training_time,
                'data_samples': len(X),
                'features': len(X.columns),
                **{k: v for k, v in metrics.items() if v is not None}
            })
            
        except Exception as e:
            self.info.status = ModelStatus.ERROR
            logger.error(f"❌ Error en entrenamiento: {e}")
            raise
        
        return self
    
    def predict(self, X: pd.DataFrame, return_proba: bool = False, 
                use_cache: bool = True, **kwargs) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Hacer predicciones
        
        Args:
            X: Features para predicción
            return_proba: Si retornar probabilidades (para clasificación)
            use_cache: Si usar cache de predicciones
            **kwargs: Parámetros adicionales
            
        Returns:
            Predicciones (y probabilidades si se solicitan)
        """
        if self.info.status != ModelStatus.TRAINED:
            raise ValueError(f"Modelo no está entrenado. Estado: {self.info.status}")
        
        # Validar datos
        self._validate_prediction_data(X)
        
        # Generar clave de cache
        cache_key = self._generate_cache_key(X, kwargs) if use_cache else None
        
        # Verificar cache
        if cache_key and cache_key in self._prediction_cache:
            logger.debug(f"Usando predicciones en cache para {len(X)} muestras")
            cached_result = self._prediction_cache[cache_key]
            if return_proba and len(cached_result) == 2:
                return cached_result
            elif not return_proba:
                return cached_result[0] if isinstance(cached_result, tuple) else cached_result
        
        # Hacer predicciones
        try:
            predictions = self._predict_model(X, **kwargs)
            
            result = predictions
            
            # Para clasificación, obtener probabilidades si se solicitan
            if return_proba and hasattr(self._model, 'predict_proba'):
                try:
                    probabilities = self._model.predict_proba(X)
                    result = (predictions, probabilities)
                except:
                    logger.warning("No se pudieron obtener probabilidades")
                    result = predictions
            
            # Guardar en cache
            if cache_key and use_cache:
                self._prediction_cache[cache_key] = result
                
                # Limpiar cache si es muy grande
                if len(self._prediction_cache) > 100:
                    oldest_key = list(self._prediction_cache.keys())[0]
                    del self._prediction_cache[oldest_key]
            
            logger.debug(f"Predicciones generadas para {len(X)} muestras")
            
            return result
            
        except Exception as e:
            logger.error(f"Error en predicciones: {e}")
            raise
    
    def save(self, directory: Path, save_metadata: bool = True) -> bool:
        """
        Guardar modelo completo
        
        Args:
            directory: Directorio donde guardar
            save_metadata: Si guardar metadatos
            
        Returns:
            True si se guardó exitosamente
        """
        logger.info(f"Guardando modelo: {self.info.name}")
        
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        
        try:
            # Rutas de archivos
            model_filename = f"{self.info.name}_v{self.info.version}.pkl"
            metadata_filename = f"{self.info.name}_v{self.info.version}_metadata.json"
            
            model_path = directory / model_filename
            metadata_path = directory / metadata_filename
            
            # Guardar modelo interno
            success = self._save_model(model_path)
            
            if not success:
                logger.error("Error guardando modelo interno")
                return False
            
            # Guardar metadatos
            if save_metadata:
                self.info.model_path = model_path
                self.info.metadata_path = metadata_path
                
                metadata = {
                    'model_info': self.info.to_dict(),
                    'training_history': self.training_history,
                    'hyperparameters': self.info.hyperparameters
                }
                
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2, default=str)
            
            logger.info(f"✅ Modelo guardado en: {model_path}")
            if save_metadata:
                logger.info(f"✅ Metadatos guardados en: {metadata_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error guardando modelo: {e}")
            return False
    
    def load(self, model_path: Path, metadata_path: Optional[Path] = None) -> bool:
        """
        Cargar modelo completo
        
        Args:
            model_path: Ruta del archivo del modelo
            metadata_path: Ruta de metadatos (opcional)
            
        Returns:
            True si se cargó exitosamente
        """
        logger.info(f"Cargando modelo desde: {model_path}")
        
        try:
            # Cargar modelo interno
            success = self._load_model(model_path)
            
            if not success:
                logger.error("Error cargando modelo interno")
                return False
            
            # Cargar metadatos si están disponibles
            if metadata_path and metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                # Reconstruir información del modelo
                model_info_dict = metadata.get('model_info', {})
                
                # Actualizar campos básicos
                self.info.name = model_info_dict.get('name', self.info.name)
                self.info.version = model_info_dict.get('version', self.info.version)
                self.info.description = model_info_dict.get('description', self.info.description)
                self.info.feature_names = model_info_dict.get('feature_names', [])
                self.info.target_name = model_info_dict.get('target_name')
                self.info.data_shape = tuple(model_info_dict['data_shape']) if model_info_dict.get('data_shape') else None
                
                # Reconstruir métricas
                if model_info_dict.get('metrics'):
                    metrics_dict = model_info_dict['metrics']
                    # Convertir timestamp
                    if 'timestamp' in metrics_dict:
                        metrics_dict['timestamp'] = datetime.fromisoformat(metrics_dict['timestamp'])
                    self.info.metrics = ModelMetrics(**{k: v for k, v in metrics_dict.items() if k != 'timestamp'})
                    self.info.metrics.timestamp = datetime.fromisoformat(model_info_dict['metrics']['timestamp'])
                
                # Cargar historial e hiperparámetros
                self.training_history = metadata.get('training_history', [])
                self.info.hyperparameters = metadata.get('hyperparameters', {})
                
                logger.info("✅ Metadatos cargados exitosamente")
            
            self.info.status = ModelStatus.TRAINED
            self.info.model_path = model_path
            self.info.metadata_path = metadata_path
            
            logger.info(f"✅ Modelo cargado: {self.info.name} v{self.info.version}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error cargando modelo: {e}")
            self.info.status = ModelStatus.ERROR
            return False
    
    # ==================== MÉTODOS DE VALIDACIÓN ====================
    
    def _validate_training_data(self, X: pd.DataFrame, y: pd.Series,
                              validation_data: Optional[Tuple[pd.DataFrame, pd.Series]]) -> None:
        """Validar datos de entrenamiento"""
        if X.empty:
            raise ValueError("DataFrame de features está vacío")
        
        if len(y) == 0:
            raise ValueError("Series de target está vacía")
        
        if len(X) != len(y):
            raise ValueError(f"Mismatch en tamaños: X={len(X)}, y={len(y)}")
        
        if X.isnull().any().any():
            logger.warning("Features contienen valores nulos")
        
        if y.isnull().any():
            logger.warning("Target contiene valores nulos")
        
        if validation_data:
            X_val, y_val = validation_data
            if len(X_val) != len(y_val):
                raise ValueError(f"Mismatch en validación: X_val={len(X_val)}, y_val={len(y_val)}")
            
            if list(X.columns) != list(X_val.columns):
                raise ValueError("Columnas de entrenamiento y validación no coinciden")
    
    def _validate_prediction_data(self, X: pd.DataFrame) -> None:
        """Validar datos para predicción"""
        if X.empty:
            raise ValueError("DataFrame de features está vacío")
        
        # Verificar que las features coincidan con las del entrenamiento
        if self.info.feature_names:
            expected_features = set(self.info.feature_names)
            actual_features = set(X.columns)
            
            missing_features = expected_features - actual_features
            if missing_features:
                raise ValueError(f"Features faltantes: {missing_features}")
            
            extra_features = actual_features - expected_features
            if extra_features:
                logger.warning(f"Features adicionales encontradas: {extra_features}")
    
    # ==================== MÉTODOS DE MÉTRICAS ====================
    
    def _calculate_metrics(self, X: pd.DataFrame, y: pd.Series,
                         validation_data: Optional[Tuple[pd.DataFrame, pd.Series]],
                         training_time: float) -> Dict[str, Any]:
        """Calcular métricas del modelo"""
        metrics = {'training_time': training_time, 'sample_size': len(X)}
        
        try:
            # Predicciones en entrenamiento
            y_pred = self._predict_model(X)
            
            if self.info.model_type == ModelType.CLASSIFIER:
                # Métricas de clasificación
                metrics.update(self._calculate_classification_metrics(y, y_pred))
            elif self.info.model_type == ModelType.REGRESSOR:
                # Métricas de regresión
                metrics.update(self._calculate_regression_metrics(y, y_pred))
            
            # Métricas en validación si están disponibles
            if validation_data:
                X_val, y_val = validation_data
                y_val_pred = self._predict_model(X_val)
                
                if self.info.model_type == ModelType.CLASSIFIER:
                    val_metrics = self._calculate_classification_metrics(y_val, y_val_pred)
                    metrics.update({f"val_{k}": v for k, v in val_metrics.items()})
                elif self.info.model_type == ModelType.REGRESSOR:
                    val_metrics = self._calculate_regression_metrics(y_val, y_val_pred)
                    metrics.update({f"val_{k}": v for k, v in val_metrics.items()})
        
        except Exception as e:
            logger.warning(f"Error calculando métricas: {e}")
        
        return metrics
    
    def _calculate_classification_metrics(self, y_true: pd.Series, 
                                        y_pred: np.ndarray) -> Dict[str, float]:
        """Calcular métricas de clasificación"""
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                
                metrics = {
                    'accuracy': accuracy_score(y_true, y_pred),
                    'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
                    'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
                    'f1_score': f1_score(y_true, y_pred, average='weighted', zero_division=0)
                }
                
                return {k: float(v) for k, v in metrics.items()}
        except Exception as e:
            logger.warning(f"Error en métricas de clasificación: {e}")
            return {}
    
    def _calculate_regression_metrics(self, y_true: pd.Series, 
                                    y_pred: np.ndarray) -> Dict[str, float]:
        """Calcular métricas de regresión"""
        try:
            metrics = {
                'mse': mean_squared_error(y_true, y_pred),
                'mae': mean_absolute_error(y_true, y_pred),
                'r2_score': r2_score(y_true, y_pred)
            }
            
            return {k: float(v) for k, v in metrics.items()}
        except Exception as e:
            logger.warning(f"Error en métricas de regresión: {e}")
            return {}
    
    def _log_metrics(self, metrics: Dict[str, Any]) -> None:
        """Log de métricas de entrenamiento"""
        logger.info("\n📊 MÉTRICAS DEL MODELO:")
        logger.info("-" * 40)
        
        for key, value in metrics.items():
            if value is not None and isinstance(value, (int, float)):
                if 'time' in key.lower():
                    logger.info(f"{key:20}: {value:.2f}s")
                elif 'loss' in key.lower() or 'error' in key.lower():
                    logger.info(f"{key:20}: {value:.6f}")
                elif value < 1:
                    logger.info(f"{key:20}: {value:.4f}")
                else:
                    logger.info(f"{key:20}: {value:.2f}")
        
        logger.info("-" * 40)
    
    # ==================== MÉTODOS AUXILIARES ====================
    
    def _generate_cache_key(self, X: pd.DataFrame, kwargs: Dict) -> str:
        """Generar clave única para cache de predicciones"""
        # Hash simple basado en shape y algunos valores
        data_hash = hash((
            X.shape,
            tuple(X.columns),
            str(X.iloc[0].to_dict()) if len(X) > 0 else "",
            str(kwargs)
        ))
        return f"pred_{abs(data_hash)}"
    
    def get_info(self) -> ModelInfo:
        """Obtener información completa del modelo"""
        return self.info
    
    def get_metrics(self) -> Optional[ModelMetrics]:
        """Obtener métricas del modelo"""
        return self.info.metrics
    
    def get_hyperparameters(self) -> Dict[str, Any]:
        """Obtener hiperparámetros"""
        return self.info.hyperparameters.copy()
    
    def set_hyperparameters(self, **kwargs) -> None:
        """Establecer hiperparámetros"""
        self.info.hyperparameters.update(kwargs)
        self.info.updated_at = datetime.now()
        logger.info(f"Hiperparámetros actualizados: {list(kwargs.keys())}")
    
    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        """Obtener importancia de features si está disponible"""
        if hasattr(self._model, 'feature_importances_') and self.info.feature_names:
            importance_df = pd.DataFrame({
                'feature': self.info.feature_names,
                'importance': self._model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            return importance_df
        
        logger.warning("Importancia de features no disponible para este modelo")
        return None
    
    def clear_cache(self) -> None:
        """Limpiar cache de predicciones"""
        self._prediction_cache.clear()
        logger.debug("Cache de predicciones limpiado")
    
    def __str__(self) -> str:
        """Representación string del modelo"""
        status_icon = "✅" if self.info.status == ModelStatus.TRAINED else "❌"
        return (f"{status_icon} {self.info.name} v{self.info.version} "
                f"({self.info.model_type.value}) - {self.info.status.value}")
    
    def __repr__(self) -> str:
        """Representación detallada del modelo"""
        return (f"BaseModel(name='{self.info.name}', "
                f"type={self.info.model_type.value}, "
                f"version='{self.info.version}', "
                f"status={self.info.status.value})")


# Funciones de utilidad
def load_model_from_directory(directory: Path, model_name: str) -> Optional[BaseModel]:
    """
    Cargar modelo desde directorio usando nombre
    
    Args:
        directory: Directorio donde buscar
        model_name: Nombre del modelo a cargar
        
    Returns:
        Modelo cargado o None si no se encuentra
    """
    directory = Path(directory)
    
    # Buscar archivos del modelo
    model_files = list(directory.glob(f"{model_name}_v*.pkl"))
    
    if not model_files:
        logger.error(f"No se encontraron archivos para el modelo: {model_name}")
        return None
    
    # Tomar el más reciente
    latest_model = max(model_files, key=lambda x: x.stat().st_mtime)
    
    # Buscar archivo de metadatos correspondiente
    metadata_file = latest_model.with_name(
        latest_model.stem + "_metadata.json"
    )
    
    logger.info(f"Intentando cargar modelo desde: {latest_model}")
    
    # Nota: Esta función requerirá conocer el tipo específico de modelo
    # En la práctica, esto se manejará en el ModelHub
    logger.warning("Esta función requiere implementación específica por tipo de modelo")
    
    return None
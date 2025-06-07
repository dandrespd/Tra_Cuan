'''
13. models/model_hub.py
Ruta: TradingBot_Cuantitative_MT5/models/model_hub.py
Resumen:

Sistema centralizado de gestión de modelos ML/DL con ciclo de vida completo
Maneja creación, entrenamiento, evaluación, versionado, selección automática y deployment
Proporciona interfaz unificada para todos los tipos de modelos con capacidades de auto-ML
Incluye comparación de modelos, optimización de hiperparámetros y selección inteligente
'''
# models/model_hub.py
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple, Callable
from datetime import datetime, timedelta
import json
import shutil
import pickle
from dataclasses import dataclass, field
import threading
import time
import warnings
warnings.filterwarnings('ignore')

# Local imports
from models.base_model import BaseModel, ModelType, ModelStatus, ModelMetrics
from models.ml_models import MLModelFactory
from models.ensemble_models import EnsembleFactory, VotingEnsembleModel, StackingEnsembleModel
from utils.log_config import get_logger, log_performance

# Optional imports for deep learning
try:
    from models.deep_models import DeepModelFactory
    DEEP_LEARNING_AVAILABLE = True
except ImportError:
    DEEP_LEARNING_AVAILABLE = False

logger = get_logger('models')


@dataclass
class ModelRegistration:
    """Registro de un modelo en el hub"""
    model: BaseModel
    registered_at: datetime
    last_used: Optional[datetime] = None
    usage_count: int = 0
    performance_history: List[Dict[str, Any]] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    is_production: bool = False
    is_backup: bool = False
    auto_retrain: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario para serialización"""
        return {
            'model_info': self.model.to_dict() if hasattr(self.model, 'to_dict') else str(self.model),
            'registered_at': self.registered_at.isoformat(),
            'last_used': self.last_used.isoformat() if self.last_used else None,
            'usage_count': self.usage_count,
            'performance_history': self.performance_history,
            'tags': self.tags,
            'is_production': self.is_production,
            'is_backup': self.is_backup,
            'auto_retrain': self.auto_retrain
        }


@dataclass
class ModelEvaluation:
    """Resultado de evaluación de modelo"""
    model_name: str
    metrics: Dict[str, float]
    evaluation_time: datetime
    dataset_info: Dict[str, Any]
    cross_validation_scores: Optional[List[float]] = None
    feature_importance: Optional[pd.DataFrame] = None
    
    @property
    def primary_score(self) -> float:
        """Score principal para ranking"""
        # Priorizar métricas específicas de trading
        priority_metrics = ['sharpe_ratio', 'profit_factor', 'win_rate', 'f1_score', 'accuracy', 'r2_score']
        
        for metric in priority_metrics:
            if metric in self.metrics:
                return self.metrics[metric]
        
        # Fallback a primera métrica disponible
        return list(self.metrics.values())[0] if self.metrics else 0.0


class ModelHub:
    """
    Hub centralizado para gestión de modelos de ML/DL
    
    Funcionalidades:
    - Registro y versionado de modelos
    - Evaluación y comparación automática
    - Selección del mejor modelo
    - Auto-ML y optimización
    - Deployment y fallback
    """
    
    def __init__(self, base_directory: Path, auto_cleanup: bool = True):
        """
        Inicializar ModelHub
        
        Args:
            base_directory: Directorio base para almacenar modelos
            auto_cleanup: Si limpiar modelos antiguos automáticamente
        """
        self.base_dir = Path(base_directory)
        self.models_dir = self.base_dir / "models"
        self.evaluations_dir = self.base_dir / "evaluations"
        self.experiments_dir = self.base_dir / "experiments"
        
        # Crear directorios
        for dir_path in [self.models_dir, self.evaluations_dir, self.experiments_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Registro de modelos
        self.registered_models: Dict[str, ModelRegistration] = {}
        self.evaluations: Dict[str, List[ModelEvaluation]] = {}
        
        # Modelo activo
        self.active_model: Optional[BaseModel] = None
        self.backup_models: List[BaseModel] = []
        
        # Configuración
        self.auto_cleanup = auto_cleanup
        self.max_models_per_type = 5
        self.evaluation_cache: Dict[str, ModelEvaluation] = {}
        
        # Threading para tareas en background
        self._cleanup_thread: Optional[threading.Thread] = None
        self._should_stop = threading.Event()
        
        # Auto-ML configuración
        self.automl_config = {
            'enabled': True,
            'max_experiments': 10,
            'time_budget_minutes': 60,
            'metric_to_optimize': 'f1_score',
            'cv_folds': 5
        }
        
        # Cargar estado previo
        self._load_hub_state()
        
        # Iniciar cleanup automático
        if self.auto_cleanup:
            self._start_cleanup_thread()
        
        logger.info(f"ModelHub inicializado en: {self.base_dir}")
        logger.info(f"Modelos registrados: {len(self.registered_models)}")
    
    # ==================== REGISTRO DE MODELOS ====================
    
    def register_model(self, model: BaseModel, tags: List[str] = None,
                      is_production: bool = False, is_backup: bool = False,
                      auto_retrain: bool = True) -> bool:
        """
        Registrar modelo en el hub
        
        Args:
            model: Modelo a registrar
            tags: Etiquetas para clasificar el modelo
            is_production: Si es modelo de producción
            is_backup: Si es modelo de respaldo
            auto_retrain: Si debe re-entrenarse automáticamente
            
        Returns:
            True si se registró exitosamente
        """
        try:
            model_name = model.info.name
            
            # Verificar si ya existe
            if model_name in self.registered_models:
                logger.warning(f"Modelo {model_name} ya está registrado. Actualizando...")
            
            # Crear registro
            registration = ModelRegistration(
                model=model,
                registered_at=datetime.now(),
                tags=tags or [],
                is_production=is_production,
                is_backup=is_backup,
                auto_retrain=auto_retrain
            )
            
            # Registrar
            self.registered_models[model_name] = registration
            
            # Guardar modelo en disco
            model_path = self.models_dir / f"{model_name}_v{model.info.version}"
            success = model.save(model_path)
            
            if success:
                logger.info(f"✅ Modelo registrado: {model_name}")
                
                # Actualizar modelo activo si es de producción
                if is_production:
                    self.set_active_model(model_name)
                
                # Agregar a backups si corresponde
                if is_backup:
                    self.backup_models.append(model)
                
                # Guardar estado del hub
                self._save_hub_state()
                
                return True
            else:
                logger.error(f"Error guardando modelo {model_name}")
                return False
                
        except Exception as e:
            logger.error(f"Error registrando modelo {model.info.name}: {e}")
            return False
    
    def unregister_model(self, model_name: str, delete_files: bool = True) -> bool:
        """Desregistrar modelo del hub"""
        if model_name not in self.registered_models:
            logger.warning(f"Modelo {model_name} no está registrado")
            return False
        
        try:
            # Remover del registro
            registration = self.registered_models.pop(model_name)
            
            # Remover de evaluaciones
            if model_name in self.evaluations:
                del self.evaluations[model_name]
            
            # Limpiar modelo activo si es necesario
            if self.active_model and self.active_model.info.name == model_name:
                self.active_model = None
                logger.warning("Modelo activo fue desregistrado")
            
            # Remover archivos si se solicita
            if delete_files:
                model_pattern = f"{model_name}_v*"
                for model_file in self.models_dir.glob(model_pattern):
                    if model_file.is_dir():
                        shutil.rmtree(model_file)
                    else:
                        model_file.unlink()
            
            logger.info(f"Modelo {model_name} desregistrado")
            self._save_hub_state()
            
            return True
            
        except Exception as e:
            logger.error(f"Error desregistrando modelo {model_name}: {e}")
            return False
    
    def list_models(self, filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Listar modelos registrados con filtros opcionales
        
        Args:
            filters: Filtros como {'type': 'CLASSIFIER', 'tag': 'production'}
            
        Returns:
            Lista de información de modelos
        """
        models_info = []
        
        for name, registration in self.registered_models.items():
            model_info = {
                'name': name,
                'type': registration.model.info.model_type.value,
                'status': registration.model.info.status.value,
                'version': registration.model.info.version,
                'registered_at': registration.registered_at.isoformat(),
                'last_used': registration.last_used.isoformat() if registration.last_used else None,
                'usage_count': registration.usage_count,
                'tags': registration.tags,
                'is_production': registration.is_production,
                'is_backup': registration.is_backup,
                'metrics': registration.model.get_metrics().to_dict() if registration.model.get_metrics() else {}
            }
            
            # Aplicar filtros
            if filters:
                include = True
                for key, value in filters.items():
                    if key == 'type' and model_info['type'] != value:
                        include = False
                        break
                    elif key == 'tag' and value not in model_info['tags']:
                        include = False
                        break
                    elif key == 'status' and model_info['status'] != value:
                        include = False
                        break
                
                if include:
                    models_info.append(model_info)
            else:
                models_info.append(model_info)
        
        # Ordenar por última vez usado
        models_info.sort(key=lambda x: x['last_used'] or '1900-01-01', reverse=True)
        
        return models_info
    
    def get_model(self, model_name: str) -> Optional[BaseModel]:
        """Obtener modelo por nombre"""
        if model_name not in self.registered_models:
            logger.warning(f"Modelo {model_name} no encontrado")
            return None
        
        registration = self.registered_models[model_name]
        
        # Actualizar estadísticas de uso
        registration.last_used = datetime.now()
        registration.usage_count += 1
        
        return registration.model
    
    # ==================== EVALUACIÓN DE MODELOS ====================
    
    def evaluate_model(self, model_name: str, X_test: pd.DataFrame, y_test: pd.Series,
                      cv_folds: int = 5, cache_results: bool = True) -> Optional[ModelEvaluation]:
        """
        Evaluar modelo específico
        
        Args:
            model_name: Nombre del modelo a evaluar
            X_test: Features de test
            y_test: Target de test
            cv_folds: Folds para validación cruzada
            cache_results: Si cachear resultados
            
        Returns:
            Evaluación del modelo
        """
        model = self.get_model(model_name)
        if not model:
            return None
        
        # Verificar cache
        cache_key = f"{model_name}_{hash(str(X_test.shape))}"
        if cache_results and cache_key in self.evaluation_cache:
            logger.debug(f"Usando evaluación cacheada para {model_name}")
            return self.evaluation_cache[cache_key]
        
        logger.info(f"Evaluando modelo: {model_name}")
        
        try:
            # Métricas básicas
            predictions = model.predict(X_test)
            metrics = {}
            
            if model.info.model_type == ModelType.CLASSIFIER:
                from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
                from sklearn.metrics import classification_report, confusion_matrix
                
                metrics.update({
                    'accuracy': accuracy_score(y_test, predictions),
                    'precision': precision_score(y_test, predictions, average='weighted', zero_division=0),
                    'recall': recall_score(y_test, predictions, average='weighted', zero_division=0),
                    'f1_score': f1_score(y_test, predictions, average='weighted', zero_division=0)
                })
                
                # Métricas adicionales para trading
                if hasattr(model, 'predict_proba'):
                    try:
                        from sklearn.metrics import roc_auc_score
                        probas = model.predict_proba(X_test)
                        if probas.shape[1] == 2:  # Clasificación binaria
                            metrics['auc_score'] = roc_auc_score(y_test, probas[:, 1])
                    except:
                        pass
            
            else:  # Regresión
                from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
                
                metrics.update({
                    'mse': mean_squared_error(y_test, predictions),
                    'mae': mean_absolute_error(y_test, predictions),
                    'r2_score': r2_score(y_test, predictions)
                })
                
                # Métricas específicas de trading para regresión
                returns_pred = predictions
                returns_actual = y_test.values
                
                # Sharpe ratio simulado
                if len(returns_pred) > 1:
                    sharpe = np.mean(returns_pred) / (np.std(returns_pred) + 1e-8) * np.sqrt(252)
                    metrics['sharpe_ratio'] = sharpe
                
                # Correlación direccional
                direction_pred = np.sign(returns_pred)
                direction_actual = np.sign(returns_actual)
                directional_accuracy = np.mean(direction_pred == direction_actual)
                metrics['directional_accuracy'] = directional_accuracy
            
            # Validación cruzada
            cv_scores = []
            if cv_folds > 1:
                try:
                    from sklearn.model_selection import cross_val_score, TimeSeriesSplit
                    
                    # Usar TimeSeriesSplit para datos financieros
                    cv = TimeSeriesSplit(n_splits=cv_folds)
                    
                    # Para modelos de sklearn nativos
                    if hasattr(model, '_model') and hasattr(model._model, 'fit'):
                        scoring = 'accuracy' if model.info.model_type == ModelType.CLASSIFIER else 'r2'
                        cv_scores = cross_val_score(model._model, X_test, y_test, cv=cv, scoring=scoring)
                        metrics['cv_mean'] = np.mean(cv_scores)
                        metrics['cv_std'] = np.std(cv_scores)
                
                except Exception as e:
                    logger.warning(f"Error en validación cruzada: {e}")
            
            # Feature importance si está disponible
            feature_importance = None
            try:
                feature_importance = model.get_feature_importance()
            except:
                pass
            
            # Crear evaluación
            evaluation = ModelEvaluation(
                model_name=model_name,
                metrics=metrics,
                evaluation_time=datetime.now(),
                dataset_info={
                    'samples': len(X_test),
                    'features': len(X_test.columns),
                    'target_name': y_test.name if hasattr(y_test, 'name') else 'target'
                },
                cross_validation_scores=cv_scores.tolist() if len(cv_scores) > 0 else None,
                feature_importance=feature_importance
            )
            
            # Guardar evaluación
            if model_name not in self.evaluations:
                self.evaluations[model_name] = []
            
            self.evaluations[model_name].append(evaluation)
            
            # Limitar historial de evaluaciones
            if len(self.evaluations[model_name]) > 10:
                self.evaluations[model_name] = self.evaluations[model_name][-10:]
            
            # Cachear
            if cache_results:
                self.evaluation_cache[cache_key] = evaluation
            
            # Actualizar historial de performance
            registration = self.registered_models[model_name]
            registration.performance_history.append({
                'timestamp': evaluation.evaluation_time.isoformat(),
                'primary_score': evaluation.primary_score,
                'metrics': metrics
            })
            
            logger.info(f"✅ Evaluación completada: {model_name}")
            logger.info(f"  Score principal: {evaluation.primary_score:.4f}")
            
            return evaluation
            
        except Exception as e:
            logger.error(f"Error evaluando modelo {model_name}: {e}")
            return None
    
    def evaluate_all_models(self, X_test: pd.DataFrame, y_test: pd.Series,
                          model_filter: Dict[str, Any] = None) -> Dict[str, ModelEvaluation]:
        """Evaluar todos los modelos registrados"""
        logger.info(f"Evaluando todos los modelos...")
        
        models_to_evaluate = self.list_models(model_filter)
        evaluations = {}
        
        for model_info in models_to_evaluate:
            model_name = model_info['name']
            evaluation = self.evaluate_model(model_name, X_test, y_test)
            
            if evaluation:
                evaluations[model_name] = evaluation
        
        # Log resumen
        if evaluations:
            sorted_evaluations = sorted(evaluations.items(), 
                                      key=lambda x: x[1].primary_score, reverse=True)
            
            logger.info("Resultados de evaluación:")
            for name, eval_result in sorted_evaluations[:5]:  # Top 5
                logger.info(f"  {name}: {eval_result.primary_score:.4f}")
        
        return evaluations
    
    def get_best_model(self, model_type: ModelType = None, 
                      metric: str = None) -> Optional[BaseModel]:
        """
        Obtener el mejor modelo según métricas
        
        Args:
            model_type: Filtrar por tipo de modelo
            metric: Métrica específica para comparar
            
        Returns:
            Mejor modelo encontrado
        """
        if not self.registered_models:
            logger.warning("No hay modelos registrados")
            return None
        
        best_model = None
        best_score = -float('inf')
        
        for name, registration in self.registered_models.items():
            model = registration.model
            
            # Filtrar por tipo si se especifica
            if model_type and model.info.model_type != model_type:
                continue
            
            # Obtener score
            score = 0.0
            
            if metric and model.get_metrics():
                metrics_dict = model.get_metrics().to_dict()
                score = metrics_dict.get(metric, 0.0)
            else:
                # Usar evaluaciones si están disponibles
                if name in self.evaluations and self.evaluations[name]:
                    latest_eval = self.evaluations[name][-1]
                    score = latest_eval.primary_score
                elif model.get_metrics():
                    # Usar métricas del modelo
                    metrics_dict = model.get_metrics().to_dict()
                    # Buscar métrica relevante
                    for key in ['f1_score', 'accuracy', 'r2_score', 'sharpe_ratio']:
                        if key in metrics_dict and metrics_dict[key]:
                            score = metrics_dict[key]
                            break
            
            if score > best_score:
                best_score = score
                best_model = model
        
        if best_model:
            logger.info(f"Mejor modelo: {best_model.info.name} (score: {best_score:.4f})")
        
        return best_model
    
    # ==================== AUTO-ML Y OPTIMIZACIÓN ====================
    
    def auto_train_models(self, X_train: pd.DataFrame, y_train: pd.Series,
                         X_val: pd.DataFrame = None, y_val: pd.Series = None,
                         model_types: List[str] = None, time_budget: int = None) -> List[BaseModel]:
        """
        Auto-entrenamiento de múltiples modelos con diferentes configuraciones
        
        Args:
            X_train: Features de entrenamiento
            y_train: Target de entrenamiento
            X_val: Features de validación
            y_val: Target de validación
            model_types: Tipos de modelos a entrenar
            time_budget: Tiempo máximo en minutos
            
        Returns:
            Lista de modelos entrenados
        """
        logger.info("="*60)
        logger.info("INICIANDO AUTO-ML")
        logger.info("="*60)
        
        # Configuración por defecto
        if model_types is None:
            model_types = ['random_forest', 'xgboost', 'lightgbm', 'logistic']
        
        if time_budget is None:
            time_budget = self.automl_config['time_budget_minutes']
        
        # Determinar tipo de tarea
        task_type = 'classification' if len(np.unique(y_train)) <= 10 else 'regression'
        
        start_time = datetime.now()
        end_time = start_time + timedelta(minutes=time_budget)
        
        trained_models = []
        experiment_count = 0
        
        logger.info(f"Tarea: {task_type}")
        logger.info(f"Modelos a probar: {model_types}")
        logger.info(f"Tiempo límite: {time_budget} minutos")
        logger.info(f"Datos: {X_train.shape[0]} muestras, {X_train.shape[1]} features")
        
        # Configuraciones de hiperparámetros para cada modelo
        hyperparameter_configs = {
            'random_forest': [
                {'n_estimators': 50, 'max_depth': 10, 'min_samples_split': 5},
                {'n_estimators': 100, 'max_depth': 20, 'min_samples_split': 2},
                {'n_estimators': 200, 'max_depth': None, 'min_samples_split': 10},
            ],
            'xgboost': [
                {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 6},
                {'n_estimators': 200, 'learning_rate': 0.05, 'max_depth': 8},
                {'n_estimators': 300, 'learning_rate': 0.02, 'max_depth': 10},
            ],
            'lightgbm': [
                {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': -1},
                {'n_estimators': 200, 'learning_rate': 0.05, 'max_depth': 10},
            ],
            'logistic': [
                {'C': 0.1, 'penalty': 'l2'},
                {'C': 1.0, 'penalty': 'l2'},
                {'C': 10.0, 'penalty': 'l1', 'solver': 'liblinear'},
            ]
        }
        
        # Entrenar modelos
        for model_type in model_types:
            if datetime.now() >= end_time:
                logger.warning("Tiempo límite alcanzado")
                break
            
            configs = hyperparameter_configs.get(model_type, [{}])
            
            for config in configs:
                if datetime.now() >= end_time:
                    break
                
                experiment_count += 1
                experiment_name = f"automl_{model_type}_{experiment_count}"
                
                logger.info(f"\nExperimento {experiment_count}: {model_type}")
                logger.info(f"Configuración: {config}")
                
                try:
                    # Crear modelo
                    model = MLModelFactory.create_model(
                        model_type, experiment_name, task_type, **config
                    )
                    
                    # Entrenar
                    validation_data = (X_val, y_val) if X_val is not None else None
                    model.fit(X_train, y_train, validation_data)
                    
                    # Registrar en hub
                    self.register_model(
                        model, 
                        tags=['automl', model_type, f'experiment_{experiment_count}'],
                        auto_retrain=False
                    )
                    
                    trained_models.append(model)
                    
                    # Log progreso
                    metrics = model.get_metrics()
                    if metrics:
                        primary_metric = getattr(metrics, 'f1_score', None) or \
                                       getattr(metrics, 'accuracy', None) or \
                                       getattr(metrics, 'r2_score', 0)
                        logger.info(f"  ✅ Score: {primary_metric:.4f}")
                    
                except Exception as e:
                    logger.error(f"  ❌ Error: {e}")
                    continue
        
        # Crear ensamble si hay múltiples modelos exitosos
        if len(trained_models) >= 2:
            try:
                logger.info("\nCreando ensamble...")
                
                ensemble = EnsembleFactory.create_voting_ensemble(
                    base_models=trained_models[:5],  # Máximo 5 modelos
                    name="automl_ensemble",
                    voting_type='soft' if task_type == 'classification' else None
                )
                
                validation_data = (X_val, y_val) if X_val is not None else None
                ensemble.fit(X_train, y_train, validation_data)
                
                self.register_model(
                    ensemble,
                    tags=['automl', 'ensemble', 'voting'],
                    is_production=True  # Marcar como producción
                )
                
                trained_models.append(ensemble)
                logger.info("✅ Ensamble creado y registrado")
                
            except Exception as e:
                logger.error(f"Error creando ensamble: {e}")
        
        # Resumen final
        elapsed_time = datetime.now() - start_time
        
        logger.info("\n" + "="*60)
        logger.info("AUTO-ML COMPLETADO")
        logger.info("="*60)
        logger.info(f"Tiempo transcurrido: {elapsed_time}")
        logger.info(f"Modelos entrenados: {len(trained_models)}")
        logger.info(f"Experimentos realizados: {experiment_count}")
        
        # Log performance del auto-ML
        log_performance({
            'automl_models_trained': len(trained_models),
            'automl_experiments': experiment_count,
            'automl_time_minutes': elapsed_time.total_seconds() / 60,
            'automl_task_type': task_type
        })
        
        return trained_models
    
    def optimize_best_model(self, X_train: pd.DataFrame, y_train: pd.Series,
                           X_val: pd.DataFrame = None, y_val: pd.Series = None) -> Optional[BaseModel]:
        """Optimizar hiperparámetros del mejor modelo actual"""
        best_model = self.get_best_model()
        
        if not best_model:
            logger.warning("No hay modelo para optimizar")
            return None
        
        logger.info(f"Optimizando hiperparámetros de: {best_model.info.name}")
        
        # Verificar si el modelo soporta optimización
        if hasattr(best_model, 'optimize_hyperparameters'):
            try:
                validation_data = (X_val, y_val) if X_val is not None else None
                optimized_params = best_model.optimize_hyperparameters(
                    X_train, y_train, validation_data
                )
                
                # Crear nuevo modelo optimizado
                optimized_name = f"{best_model.info.name}_optimized"
                
                # Re-entrenar con parámetros optimizados
                best_model.fit(X_train, y_train, validation_data)
                
                # Registrar como nueva versión
                best_model.info.name = optimized_name
                best_model.info.version = "optimized"
                
                self.register_model(
                    best_model,
                    tags=['optimized', 'hyperparameters'],
                    is_production=True
                )
                
                logger.info(f"✅ Modelo optimizado: {optimized_name}")
                return best_model
                
            except Exception as e:
                logger.error(f"Error en optimización: {e}")
                return None
        else:
            logger.warning(f"Modelo {best_model.info.name} no soporta optimización automática")
            return None
    
    # ==================== GESTIÓN DE PRODUCCIÓN ====================
    
    def set_active_model(self, model_name: str) -> bool:
        """Establecer modelo activo para producción"""
        model = self.get_model(model_name)
        
        if not model:
            logger.error(f"Modelo {model_name} no encontrado")
            return False
        
        if model.info.status != ModelStatus.TRAINED:
            logger.error(f"Modelo {model_name} no está entrenado")
            return False
        
        # Mover modelo activo anterior a backup
        if self.active_model:
            old_name = self.active_model.info.name
            self.backup_models.append(self.active_model)
            logger.info(f"Modelo anterior {old_name} movido a backup")
        
        # Establecer nuevo modelo activo
        self.active_model = model
        
        # Actualizar registro
        registration = self.registered_models[model_name]
        registration.is_production = True
        
        logger.info(f"✅ Modelo activo establecido: {model_name}")
        
        # Limpiar backups antiguos
        if len(self.backup_models) > 3:
            self.backup_models = self.backup_models[-3:]
        
        self._save_hub_state()
        
        return True
    
    def get_active_model(self) -> Optional[BaseModel]:
        """Obtener modelo activo"""
        if not self.active_model:
            # Intentar establecer el mejor modelo como activo
            best_model = self.get_best_model()
            if best_model:
                self.set_active_model(best_model.info.name)
                return self.active_model
        
        return self.active_model
    
    def predict_with_fallback(self, X: pd.DataFrame, **kwargs) -> Optional[np.ndarray]:
        """Hacer predicciones con fallback automático"""
        # Intentar con modelo activo
        if self.active_model:
            try:
                predictions = self.active_model.predict(X, **kwargs)
                
                # Actualizar estadísticas de uso
                model_name = self.active_model.info.name
                if model_name in self.registered_models:
                    self.registered_models[model_name].usage_count += 1
                    self.registered_models[model_name].last_used = datetime.now()
                
                return predictions
                
            except Exception as e:
                logger.error(f"Error con modelo activo {self.active_model.info.name}: {e}")
        
        # Intentar con modelos de backup
        for backup_model in self.backup_models:
            try:
                logger.warning(f"Usando modelo de backup: {backup_model.info.name}")
                predictions = backup_model.predict(X, **kwargs)
                return predictions
                
            except Exception as e:
                logger.error(f"Error con backup {backup_model.info.name}: {e}")
                continue
        
        # Si todo falla, intentar con cualquier modelo disponible
        for name, registration in self.registered_models.items():
            if registration.model.info.status == ModelStatus.TRAINED:
                try:
                    logger.warning(f"Usando modelo de emergencia: {name}")
                    predictions = registration.model.predict(X, **kwargs)
                    return predictions
                    
                except Exception as e:
                    logger.error(f"Error con modelo de emergencia {name}: {e}")
                    continue
        
        logger.critical("❌ TODOS LOS MODELOS FALLARON")
        return None
    
    # ==================== PERSISTENCIA Y LIMPIEZA ====================
    
    def _save_hub_state(self):
        """Guardar estado del hub"""
        try:
            state = {
                'registered_models': {
                    name: reg.to_dict() for name, reg in self.registered_models.items()
                },
                'active_model': self.active_model.info.name if self.active_model else None,
                'backup_models': [m.info.name for m in self.backup_models],
                'automl_config': self.automl_config,
                'saved_at': datetime.now().isoformat()
            }
            
            state_file = self.base_dir / 'hub_state.json'
            with open(state_file, 'w') as f:
                json.dump(state, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error guardando estado del hub: {e}")
    
    def _load_hub_state(self):
        """Cargar estado previo del hub"""
        state_file = self.base_dir / 'hub_state.json'
        
        if not state_file.exists():
            logger.info("No hay estado previo del hub")
            return
        
        try:
            with open(state_file, 'r') as f:
                state = json.load(f)
            
            # Cargar configuración
            self.automl_config.update(state.get('automl_config', {}))
            
            logger.info(f"Estado del hub cargado desde: {state.get('saved_at', 'fecha desconocida')}")
            
        except Exception as e:
            logger.error(f"Error cargando estado del hub: {e}")
    
    def _start_cleanup_thread(self):
        """Iniciar hilo de limpieza automática"""
        def cleanup_worker():
            while not self._should_stop.wait(3600):  # Cada hora
                try:
                    self._cleanup_old_models()
                    self._cleanup_old_evaluations()
                except Exception as e:
                    logger.error(f"Error en limpieza automática: {e}")
        
        self._cleanup_thread = threading.Thread(target=cleanup_worker, daemon=True)
        self._cleanup_thread.start()
        logger.debug("Hilo de limpieza iniciado")
    
    def _cleanup_old_models(self):
        """Limpiar modelos antiguos"""
        cutoff_date = datetime.now() - timedelta(days=30)
        
        models_to_remove = []
        
        for name, registration in self.registered_models.items():
            # No remover modelos de producción o backup
            if registration.is_production or registration.is_backup:
                continue
            
            # No remover modelos usados recientemente
            if registration.last_used and registration.last_used > cutoff_date:
                continue
            
            # No remover si no hay suficientes modelos del mismo tipo
            same_type_count = sum(
                1 for reg in self.registered_models.values()
                if reg.model.info.model_type == registration.model.info.model_type
            )
            
            if same_type_count <= 2:
                continue
            
            models_to_remove.append(name)
        
        # Remover modelos seleccionados
        for model_name in models_to_remove:
            logger.info(f"Limpieza automática: removiendo {model_name}")
            self.unregister_model(model_name, delete_files=True)
    
    def _cleanup_old_evaluations(self):
        """Limpiar evaluaciones antiguas"""
        for model_name, evaluations in self.evaluations.items():
            if len(evaluations) > 20:
                # Mantener solo las 15 más recientes
                self.evaluations[model_name] = sorted(
                    evaluations, 
                    key=lambda e: e.evaluation_time
                )[-15:]
    
    def cleanup_hub(self, force: bool = False):
        """Limpieza manual del hub"""
        logger.info("Iniciando limpieza manual del hub...")
        
        if force:
            logger.warning("Limpieza forzada - removiendo todos los modelos no críticos")
        
        self._cleanup_old_models()
        self._cleanup_old_evaluations()
        
        # Limpiar cache
        self.evaluation_cache.clear()
        
        logger.info("Limpieza del hub completada")
    
    def get_hub_statistics(self) -> Dict[str, Any]:
        """Obtener estadísticas del hub"""
        stats = {
            'total_models': len(self.registered_models),
            'models_by_type': {},
            'models_by_status': {},
            'active_model': self.active_model.info.name if self.active_model else None,
            'backup_models': len(self.backup_models),
            'total_evaluations': sum(len(evals) for evals in self.evaluations.values()),
            'disk_usage_mb': self._calculate_disk_usage(),
            'most_used_models': self._get_most_used_models(),
            'automl_config': self.automl_config
        }
        
        # Contar por tipo y estado
        for registration in self.registered_models.values():
            model_type = registration.model.info.model_type.value
            model_status = registration.model.info.status.value
            
            stats['models_by_type'][model_type] = stats['models_by_type'].get(model_type, 0) + 1
            stats['models_by_status'][model_status] = stats['models_by_status'].get(model_status, 0) + 1
        
        return stats
    
    def _calculate_disk_usage(self) -> float:
        """Calcular uso de disco del hub"""
        total_size = 0
        
        for path in [self.models_dir, self.evaluations_dir, self.experiments_dir]:
            for file_path in path.rglob('*'):
                if file_path.is_file():
                    total_size += file_path.stat().st_size
        
        return total_size / (1024 * 1024)  # MB
    
    def _get_most_used_models(self, top_n: int = 5) -> List[Dict[str, Any]]:
        """Obtener modelos más usados"""
        usage_list = [
            {
                'name': name,
                'usage_count': reg.usage_count,
                'last_used': reg.last_used.isoformat() if reg.last_used else None
            }
            for name, reg in self.registered_models.items()
        ]
        
        return sorted(usage_list, key=lambda x: x['usage_count'], reverse=True)[:top_n]
    
    def __del__(self):
        """Cleanup al destruir la instancia"""
        if hasattr(self, '_should_stop'):
            self._should_stop.set()
        
        if hasattr(self, '_cleanup_thread') and self._cleanup_thread:
            self._cleanup_thread.join(timeout=1)


# Funciones de utilidad
def create_model_hub(base_directory: Path = None) -> ModelHub:
    """Crear instancia de ModelHub con configuración por defecto"""
    if base_directory is None:
        base_directory = Path.cwd() / 'model_hub'
    
    return ModelHub(base_directory)


def compare_model_performance(hub: ModelHub, X_test: pd.DataFrame, y_test: pd.Series,
                            models_to_compare: List[str] = None) -> pd.DataFrame:
    """
    Comparar performance de múltiples modelos
    
    Args:
        hub: Instancia de ModelHub
        X_test: Features de test
        y_test: Target de test
        models_to_compare: Lista de nombres de modelos (None = todos)
        
    Returns:
        DataFrame con comparación de métricas
    """
    if models_to_compare is None:
        models_to_compare = list(hub.registered_models.keys())
    
    comparison_data = []
    
    for model_name in models_to_compare:
        evaluation = hub.evaluate_model(model_name, X_test, y_test)
        
        if evaluation:
            row = {
                'model_name': model_name,
                'primary_score': evaluation.primary_score,
                **evaluation.metrics
            }
            comparison_data.append(row)
    
    comparison_df = pd.DataFrame(comparison_data)
    
    if not comparison_df.empty:
        # Ordenar por score principal
        comparison_df = comparison_df.sort_values('primary_score', ascending=False)
        
        logger.info("Comparación de modelos:")
        logger.info("\n" + comparison_df.to_string(index=False))
    
    return comparison_df
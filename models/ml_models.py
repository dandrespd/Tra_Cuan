'''
10. models/ml_models.py
Ruta: TradingBot_Cuantitative_MT5/models/ml_models.py
Resumen:

Implementa modelos tradicionales de Machine Learning que heredan de BaseModel
Incluye Random Forest, XGBoost, LightGBM, SVM, Regresión Logística y más
Proporciona factory pattern para crear modelos y optimización de hiperparámetros
Métricas específicas de trading e integración completa con sistema de logging
'''
# models/ml_models.py
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime
import warnings
import joblib
from pathlib import Path
import json
warnings.filterwarnings('ignore')

# Local imports
from models.base_model import BaseModel, ModelType, ModelStatus, ModelMetrics
from utils.log_config import get_logger, log_performance

# ML imports
try:
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
    from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor
    from sklearn.svm import SVC, SVR
    from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
    from sklearn.naive_bayes import GaussianNB
    from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
    from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
    from sklearn.metrics import make_scorer
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

logger = get_logger('models')


class SklearnModel(BaseModel):
    """Wrapper para modelos de scikit-learn"""
    
    def __init__(self, model_class, name: str, model_type: ModelType, 
                 version: str = "1.0.0", **model_params):
        """
        Inicializar modelo sklearn
        
        Args:
            model_class: Clase del modelo de sklearn
            name: Nombre del modelo
            model_type: Tipo de modelo (CLASSIFIER/REGRESSOR)
            version: Versión del modelo
            **model_params: Parámetros específicos del modelo
        """
        super().__init__(name, model_type, version, f"{model_class.__name__} wrapper")
        
        self.model_class = model_class
        self.model_params = model_params
        self.info.hyperparameters.update(model_params)
        
        logger.info(f"SklearnModel inicializado: {name} ({model_class.__name__})")
    
    def _build_model(self, **kwargs) -> Any:
        """Construir modelo sklearn"""
        # Combinar parámetros por defecto con kwargs
        final_params = {**self.model_params, **kwargs}
        
        # Establecer random_state si el modelo lo soporta
        if 'random_state' in self.model_class().get_params():
            final_params.setdefault('random_state', self.random_state)
        
        model = self.model_class(**final_params)
        
        logger.info(f"Modelo {self.model_class.__name__} construido con parámetros: {final_params}")
        
        return model
    
    def _fit_model(self, X: pd.DataFrame, y: pd.Series, 
                   validation_data: Optional[Tuple[pd.DataFrame, pd.Series]] = None,
                   **kwargs) -> Dict[str, Any]:
        """Entrenar modelo sklearn"""
        start_time = datetime.now()
        
        # Entrenar modelo
        self._model.fit(X, y)
        
        training_time = (datetime.now() - start_time).total_seconds()
        
        # Métricas básicas
        train_score = self._model.score(X, y)
        
        metrics = {
            'training_score': train_score,
            'training_time': training_time
        }
        
        # Métricas de validación si están disponibles
        if validation_data:
            X_val, y_val = validation_data
            val_score = self._model.score(X_val, y_val)
            metrics['validation_score'] = val_score
            
            logger.info(f"Score entrenamiento: {train_score:.4f}")
            logger.info(f"Score validación: {val_score:.4f}")
        else:
            logger.info(f"Score entrenamiento: {train_score:.4f}")
        
        return metrics
    
    def _predict_model(self, X: pd.DataFrame, **kwargs) -> np.ndarray:
        """Hacer predicciones con modelo sklearn"""
        return self._model.predict(X)
    
    def _save_model(self, filepath: Path) -> bool:
        """Guardar modelo sklearn"""
        try:
            joblib.dump(self._model, filepath)
            return True
        except Exception as e:
            logger.error(f"Error guardando modelo sklearn: {e}")
            return False
    
    def _load_model(self, filepath: Path) -> bool:
        """Cargar modelo sklearn"""
        try:
            self._model = joblib.load(filepath)
            return True
        except Exception as e:
            logger.error(f"Error cargando modelo sklearn: {e}")
            return False
    
    def optimize_hyperparameters(self, X: pd.DataFrame, y: pd.Series,
                                validation_data: Optional[Tuple[pd.DataFrame, pd.Series]] = None,
                                param_grid: Dict[str, List] = None,
                                cv: int = 5, n_iter: int = 50) -> Dict[str, Any]:
        """
        Optimizar hiperparámetros usando GridSearchCV o RandomizedSearchCV
        
        Args:
            X: Features de entrenamiento
            y: Target de entrenamiento
            validation_data: Datos de validación
            param_grid: Grid de parámetros a probar
            cv: Número de folds para cross-validation
            n_iter: Número de iteraciones para RandomizedSearchCV
            
        Returns:
            Mejores parámetros encontrados
        """
        if param_grid is None:
            param_grid = self._get_default_param_grid()
        
        logger.info(f"Optimizando hiperparámetros para {self.info.name}")
        logger.info(f"Grid de parámetros: {param_grid}")
        
        # Crear modelo base
        base_model = self.model_class()
        
        # Scoring específico para trading
        if self.info.model_type == ModelType.CLASSIFIER:
            scoring = 'f1_weighted'
        else:
            scoring = 'r2'
        
        # Decidir entre GridSearch o RandomizedSearch
        param_combinations = 1
        for values in param_grid.values():
            param_combinations *= len(values)
        
        if param_combinations <= 100:
            # Usar GridSearchCV para grids pequeños
            search = GridSearchCV(
                base_model, param_grid, cv=cv, scoring=scoring,
                n_jobs=-1, verbose=1
            )
        else:
            # Usar RandomizedSearchCV para grids grandes
            search = RandomizedSearchCV(
                base_model, param_grid, n_iter=n_iter, cv=cv,
                scoring=scoring, n_jobs=-1, verbose=1,
                random_state=self.random_state
            )
        
        # Ejecutar búsqueda
        search.fit(X, y)
        
        # Actualizar modelo con mejores parámetros
        self._model = search.best_estimator_
        self.info.hyperparameters.update(search.best_params_)
        
        # Log resultados
        logger.info(f"✅ Optimización completada")
        logger.info(f"Mejor score: {search.best_score_:.4f}")
        logger.info(f"Mejores parámetros: {search.best_params_}")
        
        return search.best_params_
    
    def _get_default_param_grid(self) -> Dict[str, List]:
        """Obtener grid de parámetros por defecto según el tipo de modelo"""
        model_name = self.model_class.__name__
        
        if 'RandomForest' in model_name:
            return {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
        elif 'SVC' in model_name or 'SVR' in model_name:
            return {
                'C': [0.1, 1, 10, 100],
                'kernel': ['rbf', 'poly', 'sigmoid'],
                'gamma': ['scale', 'auto', 0.1, 1]
            }
        elif 'LogisticRegression' in model_name:
            return {
                'C': [0.1, 1, 10, 100],
                'penalty': ['l1', 'l2', 'elasticnet'],
                'solver': ['liblinear', 'lbfgs', 'saga']
            }
        elif 'KNeighbors' in model_name:
            return {
                'n_neighbors': [3, 5, 7, 9, 11],
                'weights': ['uniform', 'distance'],
                'metric': ['euclidean', 'manhattan', 'minkowski']
            }
        else:
            # Grid genérico
            return {}


class XGBoostModel(BaseModel):
    """Wrapper para modelos XGBoost"""
    
    def __init__(self, name: str, model_type: ModelType, 
                 version: str = "1.0.0", **model_params):
        
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost no está disponible")
        
        super().__init__(name, model_type, version, "XGBoost model")
        
        # Parámetros por defecto para XGBoost
        default_params = {
            'objective': 'binary:logistic' if model_type == ModelType.CLASSIFIER else 'reg:squarederror',
            'eval_metric': 'logloss' if model_type == ModelType.CLASSIFIER else 'rmse',
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': self.random_state,
            'verbosity': 0
        }
        
        # Actualizar con parámetros proporcionados
        default_params.update(model_params)
        self.model_params = default_params
        self.info.hyperparameters.update(default_params)
    
    def _build_model(self, **kwargs) -> Any:
        """Construir modelo XGBoost"""
        final_params = {**self.model_params, **kwargs}
        
        if self.info.model_type == ModelType.CLASSIFIER:
            model = xgb.XGBClassifier(**final_params)
        else:
            model = xgb.XGBRegressor(**final_params)
        
        logger.info(f"Modelo XGBoost construido: {self.info.model_type.value}")
        
        return model
    
    def _fit_model(self, X: pd.DataFrame, y: pd.Series,
                   validation_data: Optional[Tuple[pd.DataFrame, pd.Series]] = None,
                   **kwargs) -> Dict[str, Any]:
        """Entrenar modelo XGBoost"""
        start_time = datetime.now()
        
        # Preparar parámetros de entrenamiento
        fit_params = {
            'verbose': False
        }
        
        # Agregar validation set si está disponible
        if validation_data:
            X_val, y_val = validation_data
            fit_params['eval_set'] = [(X, y), (X_val, y_val)]
            fit_params['eval_names'] = ['train', 'valid']
            fit_params['early_stopping_rounds'] = 50
        
        # Entrenar
        self._model.fit(X, y, **fit_params)
        
        training_time = (datetime.now() - start_time).total_seconds()
        
        # Obtener métricas
        metrics = {
            'training_time': training_time,
            'n_estimators_used': self._model.n_estimators
        }
        
        # Scores de entrenamiento
        train_score = self._model.score(X, y)
        metrics['training_score'] = train_score
        
        if validation_data:
            val_score = self._model.score(X_val, y_val)
            metrics['validation_score'] = val_score
            
            # Obtener métricas del historial de entrenamiento
            if hasattr(self._model, 'evals_result_'):
                evals_result = self._model.evals_result_
                
                if 'train' in evals_result:
                    train_metric = list(evals_result['train'].values())[0]
                    metrics['final_train_metric'] = train_metric[-1]
                
                if 'valid' in evals_result:
                    val_metric = list(evals_result['valid'].values())[0]
                    metrics['final_valid_metric'] = val_metric[-1]
                    metrics['best_iteration'] = self._model.best_iteration
            
            logger.info(f"Score entrenamiento: {train_score:.4f}")
            logger.info(f"Score validación: {val_score:.4f}")
        
        return metrics
    
    def _predict_model(self, X: pd.DataFrame, **kwargs) -> np.ndarray:
        """Hacer predicciones con XGBoost"""
        return self._model.predict(X)
    
    def _save_model(self, filepath: Path) -> bool:
        """Guardar modelo XGBoost"""
        try:
            self._model.save_model(str(filepath))
            return True
        except Exception as e:
            logger.error(f"Error guardando modelo XGBoost: {e}")
            return False
    
    def _load_model(self, filepath: Path) -> bool:
        """Cargar modelo XGBoost"""
        try:
            if self.info.model_type == ModelType.CLASSIFIER:
                self._model = xgb.XGBClassifier()
            else:
                self._model = xgb.XGBRegressor()
            
            self._model.load_model(str(filepath))
            return True
        except Exception as e:
            logger.error(f"Error cargando modelo XGBoost: {e}")
            return False


class LightGBMModel(BaseModel):
    """Wrapper para modelos LightGBM"""
    
    def __init__(self, name: str, model_type: ModelType,
                 version: str = "1.0.0", **model_params):
        
        if not LIGHTGBM_AVAILABLE:
            raise ImportError("LightGBM no está disponible")
        
        super().__init__(name, model_type, version, "LightGBM model")
        
        # Parámetros por defecto
        default_params = {
            'objective': 'binary' if model_type == ModelType.CLASSIFIER else 'regression',
            'metric': 'binary_logloss' if model_type == ModelType.CLASSIFIER else 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': self.random_state,
            'verbosity': -1
        }
        
        default_params.update(model_params)
        self.model_params = default_params
        self.info.hyperparameters.update(default_params)
    
    def _build_model(self, **kwargs) -> Any:
        """Construir modelo LightGBM"""
        final_params = {**self.model_params, **kwargs}
        
        if self.info.model_type == ModelType.CLASSIFIER:
            model = lgb.LGBMClassifier(**final_params)
        else:
            model = lgb.LGBMRegressor(**final_params)
        
        logger.info(f"Modelo LightGBM construido: {self.info.model_type.value}")
        
        return model
    
    def _fit_model(self, X: pd.DataFrame, y: pd.Series,
                   validation_data: Optional[Tuple[pd.DataFrame, pd.Series]] = None,
                   **kwargs) -> Dict[str, Any]:
        """Entrenar modelo LightGBM"""
        start_time = datetime.now()
        
        fit_params = {}
        
        if validation_data:
            X_val, y_val = validation_data
            fit_params['eval_set'] = [(X_val, y_val)]
            fit_params['eval_names'] = ['valid']
            fit_params['early_stopping_rounds'] = 50
        
        # Entrenar
        self._model.fit(X, y, **fit_params)
        
        training_time = (datetime.now() - start_time).total_seconds()
        
        # Métricas
        metrics = {
            'training_time': training_time,
            'n_estimators_used': self._model.n_estimators
        }
        
        train_score = self._model.score(X, y)
        metrics['training_score'] = train_score
        
        if validation_data:
            val_score = self._model.score(X_val, y_val)
            metrics['validation_score'] = val_score
            
            if hasattr(self._model, 'best_iteration_'):
                metrics['best_iteration'] = self._model.best_iteration_
        
        return metrics
    
    def _predict_model(self, X: pd.DataFrame, **kwargs) -> np.ndarray:
        """Hacer predicciones con LightGBM"""
        return self._model.predict(X)
    
    def _save_model(self, filepath: Path) -> bool:
        """Guardar modelo LightGBM"""
        try:
            joblib.dump(self._model, filepath)
            return True
        except Exception as e:
            logger.error(f"Error guardando modelo LightGBM: {e}")
            return False
    
    def _load_model(self, filepath: Path) -> bool:
        """Cargar modelo LightGBM"""
        try:
            self._model = joblib.load(filepath)
            return True
        except Exception as e:
            logger.error(f"Error cargando modelo LightGBM: {e}")
            return False


class CatBoostModel(BaseModel):
    """Wrapper para modelos CatBoost"""
    
    def __init__(self, name: str, model_type: ModelType,
                 version: str = "1.0.0", **model_params):
        
        if not CATBOOST_AVAILABLE:
            raise ImportError("CatBoost no está disponible")
        
        super().__init__(name, model_type, version, "CatBoost model")
        
        # Parámetros por defecto
        default_params = {
            'iterations': 100,
            'learning_rate': 0.1,
            'depth': 6,
            'random_seed': self.random_state,
            'verbose': False
        }
        
        if model_type == ModelType.CLASSIFIER:
            default_params['loss_function'] = 'Logloss'
        else:
            default_params['loss_function'] = 'RMSE'
        
        default_params.update(model_params)
        self.model_params = default_params
        self.info.hyperparameters.update(default_params)
    
    def _build_model(self, **kwargs) -> Any:
        """Construir modelo CatBoost"""
        final_params = {**self.model_params, **kwargs}
        
        if self.info.model_type == ModelType.CLASSIFIER:
            model = cb.CatBoostClassifier(**final_params)
        else:
            model = cb.CatBoostRegressor(**final_params)
        
        logger.info(f"Modelo CatBoost construido: {self.info.model_type.value}")
        
        return model
    
    def _fit_model(self, X: pd.DataFrame, y: pd.Series,
                   validation_data: Optional[Tuple[pd.DataFrame, pd.Series]] = None,
                   **kwargs) -> Dict[str, Any]:
        """Entrenar modelo CatBoost"""
        start_time = datetime.now()
        
        fit_params = {}
        
        if validation_data:
            X_val, y_val = validation_data
            fit_params['eval_set'] = (X_val, y_val)
            fit_params['early_stopping_rounds'] = 50
        
        # Entrenar
        self._model.fit(X, y, **fit_params)
        
        training_time = (datetime.now() - start_time).total_seconds()
        
        # Métricas
        metrics = {
            'training_time': training_time,
            'iterations_used': self._model.tree_count_
        }
        
        train_score = self._model.score(X, y)
        metrics['training_score'] = train_score
        
        if validation_data:
            val_score = self._model.score(X_val, y_val)
            metrics['validation_score'] = val_score
        
        return metrics
    
    def _predict_model(self, X: pd.DataFrame, **kwargs) -> np.ndarray:
        """Hacer predicciones con CatBoost"""
        return self._model.predict(X)
    
    def _save_model(self, filepath: Path) -> bool:
        """Guardar modelo CatBoost"""
        try:
            self._model.save_model(str(filepath))
            return True
        except Exception as e:
            logger.error(f"Error guardando modelo CatBoost: {e}")
            return False
    
    def _load_model(self, filepath: Path) -> bool:
        """Cargar modelo CatBoost"""
        try:
            if self.info.model_type == ModelType.CLASSIFIER:
                self._model = cb.CatBoostClassifier()
            else:
                self._model = cb.CatBoostRegressor()
            
            self._model.load_model(str(filepath))
            return True
        except Exception as e:
            logger.error(f"Error cargando modelo CatBoost: {e}")
            return False


class MLModelFactory:
    """Factory para crear diferentes tipos de modelos ML"""
    
    # Registro de modelos disponibles
    _models = {
        # Scikit-learn models
        'random_forest': {
            'classifier': RandomForestClassifier,
            'regressor': RandomForestRegressor,
            'wrapper': SklearnModel
        },
        'extra_trees': {
            'classifier': ExtraTreesClassifier,
            'regressor': ExtraTreesRegressor,
            'wrapper': SklearnModel
        },
        'svm': {
            'classifier': SVC,
            'regressor': SVR,
            'wrapper': SklearnModel
        },
        'logistic': {
            'classifier': LogisticRegression,
            'regressor': LinearRegression,
            'wrapper': SklearnModel
        },
        'ridge': {
            'classifier': LogisticRegression,
            'regressor': Ridge,
            'wrapper': SklearnModel
        },
        'lasso': {
            'classifier': LogisticRegression,
            'regressor': Lasso,
            'wrapper': SklearnModel
        },
        'naive_bayes': {
            'classifier': GaussianNB,
            'regressor': None,
            'wrapper': SklearnModel
        },
        'knn': {
            'classifier': KNeighborsClassifier,
            'regressor': KNeighborsRegressor,
            'wrapper': SklearnModel
        },
        'adaboost': {
            'classifier': AdaBoostClassifier,
            'regressor': AdaBoostRegressor,
            'wrapper': SklearnModel
        },
        
        # Gradient boosting models
        'xgboost': {
            'wrapper': XGBoostModel,
            'available': XGBOOST_AVAILABLE
        },
        'lightgbm': {
            'wrapper': LightGBMModel,
            'available': LIGHTGBM_AVAILABLE
        },
        'catboost': {
            'wrapper': CatBoostModel,
            'available': CATBOOST_AVAILABLE
        }
    }
    
    @classmethod
    def create_model(cls, model_type: str, name: str, task_type: str,
                    version: str = "1.0.0", **model_params) -> BaseModel:
        """
        Crear modelo específico
        
        Args:
            model_type: Tipo de modelo ('random_forest', 'xgboost', etc.)
            name: Nombre del modelo
            task_type: 'classification' o 'regression'
            version: Versión del modelo
            **model_params: Parámetros específicos del modelo
            
        Returns:
            Instancia del modelo
        """
        if model_type not in cls._models:
            available_models = list(cls._models.keys())
            raise ValueError(f"Modelo '{model_type}' no disponible. "
                           f"Modelos disponibles: {available_models}")
        
        model_info = cls._models[model_type]
        
        # Verificar disponibilidad para modelos externos
        if 'available' in model_info and not model_info['available']:
            raise ImportError(f"Modelo '{model_type}' requiere librería adicional no instalada")
        
        # Determinar ModelType
        if task_type.lower() in ['classification', 'classifier']:
            ml_model_type = ModelType.CLASSIFIER
            sklearn_key = 'classifier'
        elif task_type.lower() in ['regression', 'regressor']:
            ml_model_type = ModelType.REGRESSOR
            sklearn_key = 'regressor'
        else:
            raise ValueError(f"task_type debe ser 'classification' o 'regression', got '{task_type}'")
        
        # Crear modelo según el wrapper
        wrapper_class = model_info['wrapper']
        
        if wrapper_class == SklearnModel:
            # Modelos sklearn
            if sklearn_key not in model_info or model_info[sklearn_key] is None:
                raise ValueError(f"Modelo '{model_type}' no soporta task_type '{task_type}'")
            
            sklearn_class = model_info[sklearn_key]
            model = SklearnModel(
                sklearn_class, name, ml_model_type, version, **model_params
            )
        
        else:
            # Modelos con wrapper específico (XGBoost, LightGBM, etc.)
            model = wrapper_class(name, ml_model_type, version, **model_params)
        
        logger.info(f"✅ Modelo creado: {name} ({model_type}, {task_type})")
        
        return model
    
    @classmethod
    def get_available_models(cls) -> Dict[str, Dict[str, Any]]:
        """Obtener lista de modelos disponibles"""
        available = {}
        
        for model_name, model_info in cls._models.items():
            available[model_name] = {
                'available': model_info.get('available', SKLEARN_AVAILABLE),
                'supports_classification': 'classifier' in model_info or model_info.get('wrapper') not in [SklearnModel],
                'supports_regression': 'regressor' in model_info or model_info.get('wrapper') not in [SklearnModel],
                'wrapper': model_info['wrapper'].__name__
            }
        
        return available
    
    @classmethod
    def create_ensemble_models(cls, base_model_configs: List[Dict[str, Any]], 
                             ensemble_name: str, task_type: str) -> List[BaseModel]:
        """
        Crear múltiples modelos para ensemble
        
        Args:
            base_model_configs: Lista de configuraciones de modelos
            ensemble_name: Nombre base para el ensemble
            task_type: Tipo de tarea
            
        Returns:
            Lista de modelos entrenados
        """
        models = []
        
        for i, config in enumerate(base_model_configs):
            model_type = config.pop('model_type')
            model_name = f"{ensemble_name}_{model_type}_{i}"
            
            try:
                model = cls.create_model(
                    model_type=model_type,
                    name=model_name,
                    task_type=task_type,
                    **config
                )
                models.append(model)
                
            except Exception as e:
                logger.error(f"Error creando modelo {model_name}: {e}")
        
        logger.info(f"Ensemble creado: {len(models)} modelos base")
        
        return models
    
    @classmethod
    def get_default_configs(cls, model_type: str, task_type: str) -> Dict[str, Any]:
        """Obtener configuración por defecto para un modelo"""
        
        configs = {
            'random_forest': {
                'n_estimators': 100,
                'max_depth': None,
                'min_samples_split': 2,
                'min_samples_leaf': 1,
                'random_state': 42
            },
            'xgboost': {
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8
            },
            'lightgbm': {
                'n_estimators': 100,
                'num_leaves': 31,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8
            },
            'svm': {
                'C': 1.0,
                'kernel': 'rbf',
                'gamma': 'scale'
            },
            'logistic': {
                'C': 1.0,
                'penalty': 'l2',
                'solver': 'lbfgs'
            }
        }
        
        return configs.get(model_type, {})


# Funciones de utilidad
def create_trading_models_suite(task_type: str = 'classification') -> List[BaseModel]:
    """Crear suite completa de modelos para trading"""
    
    models_configs = [
        {'model_type': 'random_forest', 'n_estimators': 100},
        {'model_type': 'extra_trees', 'n_estimators': 100},
        {'model_type': 'logistic', 'C': 1.0}
    ]
    
    # Agregar modelos avanzados si están disponibles
    if XGBOOST_AVAILABLE:
        models_configs.append({'model_type': 'xgboost', 'n_estimators': 100})
    
    if LIGHTGBM_AVAILABLE:
        models_configs.append({'model_type': 'lightgbm', 'n_estimators': 100})
    
    if CATBOOST_AVAILABLE:
        models_configs.append({'model_type': 'catboost', 'iterations': 100})
    
    models = []
    
    for config in models_configs:
        model_type = config.pop('model_type')
        model_name = f"trading_{model_type}"
        
        try:
            model = MLModelFactory.create_model(
                model_type=model_type,
                name=model_name,
                task_type=task_type,
                **config
            )
            models.append(model)
            
        except Exception as e:
            logger.warning(f"No se pudo crear modelo {model_type}: {e}")
    
    logger.info(f"Suite de modelos creada: {len(models)} modelos")
    
    return models


def benchmark_models(models: List[BaseModel], X: pd.DataFrame, y: pd.Series,
                    test_size: float = 0.2) -> pd.DataFrame:
    """
    Comparar performance de múltiples modelos
    
    Args:
        models: Lista de modelos a comparar
        X: Features
        y: Target
        test_size: Proporción para test
        
    Returns:
        DataFrame con resultados de benchmark
    """
    from sklearn.model_selection import train_test_split
    
    # Dividir datos
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y if len(np.unique(y)) > 1 else None
    )
    
    results = []
    
    for model in models:
        logger.info(f"Benchmarking modelo: {model.info.name}")
        
        try:
            # Entrenar
            start_time = datetime.now()
            model.fit(X_train, y_train, validation_data=(X_test, y_test))
            training_time = (datetime.now() - start_time).total_seconds()
            
            # Evaluar
            train_score = model._model.score(X_train, y_train)
            test_score = model._model.score(X_test, y_test)
            
            results.append({
                'model_name': model.info.name,
                'model_type': model.info.model_type.value,
                'train_score': train_score,
                'test_score': test_score,
                'training_time': training_time,
                'overfitting': train_score - test_score
            })
            
        except Exception as e:
            logger.error(f"Error en benchmark de {model.info.name}: {e}")
            results.append({
                'model_name': model.info.name,
                'model_type': model.info.model_type.value,
                'train_score': 0,
                'test_score': 0,
                'training_time': 0,
                'overfitting': 0,
                'error': str(e)
            })
    
    results_df = pd.DataFrame(results)
    
    if not results_df.empty:
        # Ordenar por test score
        results_df = results_df.sort_values('test_score', ascending=False)
        
        logger.info("Resultados del benchmark:")
        logger.info("\n" + results_df.to_string(index=False))
    
    return results_df
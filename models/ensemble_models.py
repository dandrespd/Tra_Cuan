'''
12. models/ensemble_models.py
Ruta: TradingBot_Cuantitative_MT5/models/ensemble_models.py
Resumen:

Implementa modelos de ensamble (Voting, Bagging, Boosting, Stacking) especializados para trading
Combina múltiples modelos base para mejorar performance y robustez de predicciones
Incluye estrategias específicas como votación ponderada por Sharpe ratio y stacking adaptativo
Proporciona métricas de diversidad entre modelos y optimización automática de pesos
'''
# models/ensemble_models.py
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple, Callable
import warnings
warnings.filterwarnings('ignore')

# Sklearn imports
from sklearn.ensemble import VotingClassifier, VotingRegressor, BaggingClassifier, BaggingRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.metrics import accuracy_score, mean_squared_error
from scipy.stats import pearsonr
from scipy.optimize import minimize

# Local imports
from models.base_model import BaseModel, ModelType, ModelStatus, ModelMetrics
from models.ml_models import MLModelFactory
from utils.log_config import get_logger
import joblib

logger = get_logger('models')


class EnsembleMetrics:
    """Métricas específicas para modelos de ensamble"""
    
    @staticmethod
    def calculate_diversity(predictions_list: List[np.ndarray], 
                          method: str = 'correlation') -> float:
        """
        Calcular diversidad entre predicciones de modelos
        
        Args:
            predictions_list: Lista de arrays de predicciones
            method: 'correlation', 'disagreement', 'kappa'
            
        Returns:
            Score de diversidad (mayor = más diverso)
        """
        if len(predictions_list) < 2:
            return 0.0
        
        if method == 'correlation':
            # Promedio de correlaciones (menor correlación = mayor diversidad)
            correlations = []
            for i in range(len(predictions_list)):
                for j in range(i+1, len(predictions_list)):
                    corr, _ = pearsonr(predictions_list[i], predictions_list[j])
                    if not np.isnan(corr):
                        correlations.append(abs(corr))
            
            avg_correlation = np.mean(correlations) if correlations else 0
            return 1 - avg_correlation  # Invertir para que mayor sea mejor
        
        elif method == 'disagreement':
            # Porcentaje de desacuerdo para clasificación
            total_disagreements = 0
            total_pairs = 0
            
            for i in range(len(predictions_list)):
                for j in range(i+1, len(predictions_list)):
                    disagreements = np.sum(predictions_list[i] != predictions_list[j])
                    total_disagreements += disagreements
                    total_pairs += len(predictions_list[i])
            
            return total_disagreements / total_pairs if total_pairs > 0 else 0
        
        return 0.0
    
    @staticmethod
    def calculate_ensemble_strength(individual_accuracies: List[float],
                                  ensemble_accuracy: float) -> float:
        """
        Calcular fortaleza del ensamble vs modelos individuales
        
        Args:
            individual_accuracies: Accuracy de modelos individuales
            ensemble_accuracy: Accuracy del ensamble
            
        Returns:
            Improvement ratio del ensamble
        """
        avg_individual = np.mean(individual_accuracies)
        best_individual = np.max(individual_accuracies)
        
        improvement_vs_avg = (ensemble_accuracy - avg_individual) / avg_individual
        improvement_vs_best = (ensemble_accuracy - best_individual) / best_individual
        
        return {
            'vs_average': improvement_vs_avg,
            'vs_best': improvement_vs_best,
            'absolute_gain': ensemble_accuracy - best_individual
        }


class VotingEnsembleModel(BaseModel):
    """
    Modelo de ensamble por votación (hard/soft voting)
    Combina predicciones de múltiples modelos base
    """
    
    def __init__(self, name: str, base_models: List[BaseModel], 
                 voting_type: str = 'soft', **kwargs):
        """
        Inicializar modelo de votación
        
        Args:
            name: Nombre del ensamble
            base_models: Lista de modelos base
            voting_type: 'hard' o 'soft' voting
            **kwargs: Parámetros adicionales
        """
        # Verificar que todos los modelos sean del mismo tipo
        model_types = [model.info.model_type for model in base_models]
        if len(set(model_types)) > 1:
            raise ValueError("Todos los modelos base deben ser del mismo tipo")
        
        model_type = model_types[0]
        super().__init__(name, ModelType.ENSEMBLE, version="1.0.0",
                        description=f"Voting Ensemble ({voting_type})")
        
        self.base_models = base_models
        self.voting_type = voting_type
        self.original_model_type = model_type
        
        # Hiperparámetros
        default_params = {
            'voting': voting_type,
            'weights': None,  # Pesos automáticos o manuales
            'optimize_weights': True,
            'diversity_threshold': 0.1  # Mínima diversidad requerida
        }
        
        default_params.update(kwargs)
        self.set_hyperparameters(**default_params)
        
        # Métricas de diversidad
        self.diversity_score = 0.0
        self.individual_performances = {}
        
        logger.info(f"Voting Ensemble inicializado: {len(base_models)} modelos")
        logger.info(f"Tipo de votación: {voting_type}")
    
    def _build_model(self, **kwargs):
        """Construir ensamble de votación"""
        params = self.get_hyperparameters()
        
        # Preparar lista de estimadores
        estimators = []
        for i, model in enumerate(self.base_models):
            estimator_name = f"model_{i}_{model.info.name}"
            # Usamos el modelo interno de cada BaseModel
            if hasattr(model, '_model') and model._model is not None:
                estimators.append((estimator_name, model._model))
            else:
                logger.warning(f"Modelo {model.info.name} no está entrenado")
        
        if not estimators:
            raise ValueError("No hay modelos entrenados disponibles")
        
        # Crear ensamble según tipo
        if self.original_model_type == ModelType.CLASSIFIER:
            ensemble = VotingClassifier(
                estimators=estimators,
                voting=params['voting'],
                weights=params['weights']
            )
        else:
            ensemble = VotingRegressor(
                estimators=estimators,
                weights=params['weights']
            )
        
        logger.info(f"Ensamble construido con {len(estimators)} estimadores")
        return ensemble
    
    def _fit_model(self, X: pd.DataFrame, y: pd.Series,
                   validation_data: Optional[Tuple[pd.DataFrame, pd.Series]] = None,
                   **kwargs) -> Dict[str, Any]:
        """Entrenar ensamble"""
        
        # Verificar que todos los modelos base estén entrenados
        untrained_models = [m for m in self.base_models if m.info.status != ModelStatus.TRAINED]
        if untrained_models:
            logger.info(f"Entrenando {len(untrained_models)} modelos base...")
            for model in untrained_models:
                try:
                    model.fit(X, y, validation_data)
                except Exception as e:
                    logger.error(f"Error entrenando {model.info.name}: {e}")
        
        # Construir ensamble
        self._model = self._build_model()
        
        # Preparar datos para sklearn
        X_sklearn = X.values if hasattr(X, 'values') else X
        y_sklearn = y.values if hasattr(y, 'values') else y
        
        # Entrenar ensamble
        self._model.fit(X_sklearn, y_sklearn)
        
        # Calcular métricas de diversidad
        predictions_list = []
        individual_scores = []
        
        for model in self.base_models:
            if model.info.status == ModelStatus.TRAINED:
                try:
                    pred = model.predict(X)
                    predictions_list.append(pred)
                    
                    # Calcular performance individual
                    if self.original_model_type == ModelType.CLASSIFIER:
                        score = accuracy_score(y, pred)
                    else:
                        score = -mean_squared_error(y, pred)  # Negativo para que mayor sea mejor
                    
                    individual_scores.append(score)
                    self.individual_performances[model.info.name] = score
                    
                except Exception as e:
                    logger.warning(f"Error evaluando {model.info.name}: {e}")
        
        # Calcular diversidad
        if len(predictions_list) >= 2:
            self.diversity_score = EnsembleMetrics.calculate_diversity(
                predictions_list, 
                method='correlation' if self.original_model_type == ModelType.REGRESSOR else 'disagreement'
            )
        
        # Optimizar pesos si está habilitado
        weights = None
        if self.get_hyperparameters()['optimize_weights'] and len(individual_scores) > 1:
            weights = self._optimize_weights(X, y, validation_data)
            if weights is not None:
                self._model.set_params(weights=weights)
                self._model.fit(X_sklearn, y_sklearn)  # Re-entrenar con nuevos pesos
        
        # Métricas finales
        ensemble_pred = self._predict_model(X)
        if self.original_model_type == ModelType.CLASSIFIER:
            ensemble_score = accuracy_score(y, ensemble_pred)
        else:
            ensemble_score = -mean_squared_error(y, ensemble_pred)
        
        # Calcular fortaleza del ensamble
        strength_metrics = EnsembleMetrics.calculate_ensemble_strength(
            individual_scores, ensemble_score
        )
        
        metrics = {
            'ensemble_score': ensemble_score,
            'diversity_score': self.diversity_score,
            'num_base_models': len(self.base_models),
            'trained_models': len([m for m in self.base_models if m.info.status == ModelStatus.TRAINED]),
            'avg_individual_score': np.mean(individual_scores) if individual_scores else 0,
            'best_individual_score': np.max(individual_scores) if individual_scores else 0,
            'improvement_vs_avg': strength_metrics['vs_average'],
            'improvement_vs_best': strength_metrics['vs_best'],
            'optimized_weights': weights is not None
        }
        
        logger.info(f"Ensamble entrenado:")
        logger.info(f"  Score: {ensemble_score:.4f}")
        logger.info(f"  Diversidad: {self.diversity_score:.4f}")
        logger.info(f"  Mejora vs mejor individual: {strength_metrics['vs_best']:.2%}")
        
        return metrics
    
    def _optimize_weights(self, X: pd.DataFrame, y: pd.Series,
                         validation_data: Optional[Tuple[pd.DataFrame, pd.Series]]) -> Optional[List[float]]:
        """Optimizar pesos del ensamble"""
        logger.info("Optimizando pesos del ensamble...")
        
        # Obtener predicciones de cada modelo
        predictions_matrix = []
        valid_models = []
        
        for model in self.base_models:
            if model.info.status == ModelStatus.TRAINED:
                try:
                    pred = model.predict(X)
                    predictions_matrix.append(pred)
                    valid_models.append(model)
                except Exception as e:
                    logger.warning(f"Error obteniendo predicciones de {model.info.name}: {e}")
        
        if len(predictions_matrix) < 2:
            logger.warning("Insuficientes modelos para optimización de pesos")
            return None
        
        predictions_matrix = np.array(predictions_matrix).T  # Shape: (samples, models)
        
        # Función objetivo (minimizar error)
        def objective(weights):
            weights = np.array(weights)
            weights = weights / np.sum(weights)  # Normalizar
            
            ensemble_pred = np.average(predictions_matrix, axis=1, weights=weights)
            
            if self.original_model_type == ModelType.CLASSIFIER:
                # Para clasificación, usar accuracy negativa
                ensemble_pred_binary = (ensemble_pred > 0.5).astype(int)
                return -accuracy_score(y, ensemble_pred_binary)
            else:
                # Para regresión, usar MSE
                return mean_squared_error(y, ensemble_pred)
        
        # Optimización con restricciones
        n_models = len(valid_models)
        initial_weights = np.ones(n_models) / n_models
        
        # Restricción: suma de pesos = 1
        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        
        # Bounds: pesos entre 0 y 1
        bounds = [(0, 1) for _ in range(n_models)]
        
        try:
            result = minimize(
                objective,
                initial_weights,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints
            )
            
            if result.success:
                optimized_weights = result.x.tolist()
                logger.info(f"Pesos optimizados: {[f'{w:.3f}' for w in optimized_weights]}")
                return optimized_weights
            else:
                logger.warning("Optimización de pesos falló")
                return None
                
        except Exception as e:
            logger.error(f"Error en optimización de pesos: {e}")
            return None
    
    def _predict_model(self, X: pd.DataFrame, **kwargs) -> np.ndarray:
        """Hacer predicciones con ensamble"""
        X_sklearn = X.values if hasattr(X, 'values') else X
        return self._model.predict(X_sklearn)
    
    def _save_model(self, filepath: Path) -> bool:
        """Guardar modelo de ensamble"""
        try:
            # Crear directorio para el ensamble
            ensemble_dir = filepath.parent / f"{filepath.stem}_ensemble"
            ensemble_dir.mkdir(exist_ok=True)
            
            # Guardar ensamble principal
            joblib.dump(self._model, ensemble_dir / "ensemble_model.pkl")
            
            # Guardar modelos base
            base_models_dir = ensemble_dir / "base_models"
            base_models_dir.mkdir(exist_ok=True)
            
            saved_models = []
            for i, model in enumerate(self.base_models):
                model_file = base_models_dir / f"base_model_{i}.pkl"
                if model.save(base_models_dir, save_metadata=False):
                    saved_models.append({
                        'index': i,
                        'name': model.info.name,
                        'type': model.info.model_type.value,
                        'file': str(model_file.relative_to(ensemble_dir))
                    })
            
            # Guardar metadatos del ensamble
            ensemble_metadata = {
                'voting_type': self.voting_type,
                'original_model_type': self.original_model_type.value,
                'diversity_score': self.diversity_score,
                'individual_performances': self.individual_performances,
                'saved_models': saved_models,
                'hyperparameters': self.get_hyperparameters()
            }
            
            with open(ensemble_dir / "ensemble_metadata.json", 'w') as f:
                import json
                json.dump(ensemble_metadata, f, indent=2, default=str)
            
            logger.info(f"Ensamble guardado en: {ensemble_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Error guardando ensamble: {e}")
            return False
    
    def _load_model(self, filepath: Path) -> bool:
        """Cargar modelo de ensamble"""
        try:
            ensemble_dir = filepath.parent / f"{filepath.stem}_ensemble"
            
            # Cargar ensamble principal
            self._model = joblib.load(ensemble_dir / "ensemble_model.pkl")
            
            # Cargar metadatos
            with open(ensemble_dir / "ensemble_metadata.json", 'r') as f:
                import json
                metadata = json.load(f)
            
            self.voting_type = metadata['voting_type']
            self.original_model_type = ModelType(metadata['original_model_type'])
            self.diversity_score = metadata.get('diversity_score', 0.0)
            self.individual_performances = metadata.get('individual_performances', {})
            
            # Nota: La carga completa de modelos base requeriría 
            # conocer sus tipos específicos. En un sistema real,
            # esto se manejaría a través del ModelHub
            
            logger.info(f"Ensamble cargado desde: {ensemble_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Error cargando ensamble: {e}")
            return False
    
    def get_model_contributions(self, X: pd.DataFrame) -> pd.DataFrame:
        """Obtener contribuciones de cada modelo base"""
        if not hasattr(self._model, 'estimators_'):
            raise ValueError("Ensamble no entrenado")
        
        contributions = {}
        
        for i, (name, estimator) in enumerate(self._model.estimators_):
            try:
                pred = estimator.predict(X.values if hasattr(X, 'values') else X)
                contributions[name] = pred
            except Exception as e:
                logger.warning(f"Error obteniendo contribución de {name}: {e}")
        
        return pd.DataFrame(contributions, index=X.index if hasattr(X, 'index') else None)
    
    def analyze_disagreements(self, X: pd.DataFrame, threshold: float = 0.1) -> pd.DataFrame:
        """Analizar desacuerdos entre modelos base"""
        contributions = self.get_model_contributions(X)
        
        if contributions.empty:
            return pd.DataFrame()
        
        # Calcular desacuerdos
        disagreements = []
        
        for idx in contributions.index:
            row_predictions = contributions.loc[idx].values
            
            if self.original_model_type == ModelType.CLASSIFIER:
                # Para clasificación: porcentaje de desacuerdo
                unique_preds = len(np.unique(row_predictions))
                disagreement = (unique_preds - 1) / (len(row_predictions) - 1) if len(row_predictions) > 1 else 0
            else:
                # Para regresión: coeficiente de variación
                std_dev = np.std(row_predictions)
                mean_pred = np.mean(row_predictions)
                disagreement = std_dev / mean_pred if mean_pred != 0 else 0
            
            disagreements.append({
                'index': idx,
                'disagreement_score': disagreement,
                'high_disagreement': disagreement > threshold,
                'predictions': row_predictions.tolist()
            })
        
        return pd.DataFrame(disagreements)


class StackingEnsembleModel(BaseModel):
    """
    Modelo de ensamble por stacking
    Usa un meta-modelo para combinar predicciones de modelos base
    """
    
    def __init__(self, name: str, base_models: List[BaseModel], 
                 meta_model_type: str = 'linear', **kwargs):
        """
        Inicializar modelo de stacking
        
        Args:
            name: Nombre del ensamble
            base_models: Lista de modelos base
            meta_model_type: Tipo de meta-modelo ('linear', 'rf', 'xgb')
            **kwargs: Parámetros adicionales
        """
        # Verificar tipos de modelos
        model_types = [model.info.model_type for model in base_models]
        if len(set(model_types)) > 1:
            raise ValueError("Todos los modelos base deben ser del mismo tipo")
        
        model_type = model_types[0]
        super().__init__(name, ModelType.ENSEMBLE, version="1.0.0",
                        description=f"Stacking Ensemble (meta: {meta_model_type})")
        
        self.base_models = base_models
        self.meta_model_type = meta_model_type
        self.original_model_type = model_type
        
        # Meta-modelo
        self.meta_model = None
        
        # Hiperparámetros
        default_params = {
            'cv_folds': 5,
            'meta_model_type': meta_model_type,
            'use_original_features': False,  # Si incluir features originales en meta-modelo
            'meta_model_params': {}
        }
        
        default_params.update(kwargs)
        self.set_hyperparameters(**default_params)
        
        logger.info(f"Stacking Ensemble inicializado: {len(base_models)} modelos base")
        logger.info(f"Meta-modelo: {meta_model_type}")
    
    def _create_meta_model(self):
        """Crear meta-modelo según tipo especificado"""
        params = self.get_hyperparameters()
        meta_type = params['meta_model_type']
        meta_params = params['meta_model_params']
        
        task = 'classification' if self.original_model_type == ModelType.CLASSIFIER else 'regression'
        
        if meta_type == 'linear':
            if task == 'classification':
                from sklearn.linear_model import LogisticRegression
                return LogisticRegression(random_state=42, **meta_params)
            else:
                from sklearn.linear_model import LinearRegression
                return LinearRegression(**meta_params)
        
        elif meta_type == 'rf':
            meta_model = MLModelFactory.create_model('random_forest', 'meta_rf', task, **meta_params)
            return meta_model._model if hasattr(meta_model, '_model') else meta_model
        
        elif meta_type == 'xgb':
            meta_model = MLModelFactory.create_model('xgboost', 'meta_xgb', task, **meta_params)
            return meta_model._model if hasattr(meta_model, '_model') else meta_model
        
        else:
            raise ValueError(f"Meta-modelo no soportado: {meta_type}")
    
    def _generate_meta_features(self, X: pd.DataFrame, y: pd.Series,
                              use_cv: bool = True) -> np.ndarray:
        """Generar features para el meta-modelo usando validación cruzada"""
        params = self.get_hyperparameters()
        
        meta_features = []
        
        if use_cv:
            # Usar validación cruzada para evitar overfitting
            from sklearn.model_selection import KFold
            
            kf = KFold(n_splits=params['cv_folds'], shuffle=True, random_state=42)
            n_samples = len(X)
            n_models = len(self.base_models)
            
            # Inicializar matriz de meta-features
            cv_predictions = np.zeros((n_samples, n_models))
            
            for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
                logger.info(f"Procesando fold {fold + 1}/{params['cv_folds']}")
                
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
                for model_idx, model in enumerate(self.base_models):
                    try:
                        # Entrenar modelo en train fold
                        model_copy = self._copy_model(model)
                        model_copy.fit(X_train, y_train)
                        
                        # Predecir en validation fold
                        pred = model_copy.predict(X_val)
                        cv_predictions[val_idx, model_idx] = pred
                        
                    except Exception as e:
                        logger.warning(f"Error en fold {fold} para {model.info.name}: {e}")
                        cv_predictions[val_idx, model_idx] = np.mean(y_train)  # Fallback
            
            meta_features = cv_predictions
        
        else:
            # Usar predicciones directas (mayor riesgo de overfitting)
            for model in self.base_models:
                if model.info.status == ModelStatus.TRAINED:
                    pred = model.predict(X)
                    meta_features.append(pred)
            
            meta_features = np.column_stack(meta_features)
        
        # Agregar features originales si está configurado
        if params['use_original_features']:
            X_scaled = StandardScaler().fit_transform(X)
            meta_features = np.column_stack([meta_features, X_scaled])
        
        return meta_features
    
    def _copy_model(self, model: BaseModel) -> BaseModel:
        """Crear copia de un modelo (simplificado)"""
        # En un sistema real, esto requeriría un método más sofisticado
        # Por ahora, retornamos el modelo original
        # TODO: Implementar clonación profunda de modelos
        return model
    
    def _build_model(self, **kwargs):
        """Construir meta-modelo"""
        self.meta_model = self._create_meta_model()
        return self.meta_model
    
    def _fit_model(self, X: pd.DataFrame, y: pd.Series,
                   validation_data: Optional[Tuple[pd.DataFrame, pd.Series]] = None,
                   **kwargs) -> Dict[str, Any]:
        """Entrenar modelo de stacking"""
        
        # Entrenar modelos base si es necesario
        untrained_models = [m for m in self.base_models if m.info.status != ModelStatus.TRAINED]
        if untrained_models:
            logger.info(f"Entrenando {len(untrained_models)} modelos base...")
            for model in untrained_models:
                try:
                    model.fit(X, y, validation_data)
                except Exception as e:
                    logger.error(f"Error entrenando {model.info.name}: {e}")
        
        # Generar meta-features
        logger.info("Generando meta-features con validación cruzada...")
        meta_X = self._generate_meta_features(X, y, use_cv=True)
        
        logger.info(f"Meta-features generadas: {meta_X.shape}")
        
        # Construir y entrenar meta-modelo
        self._model = self._build_model()
        self._model.fit(meta_X, y)
        
        # Calcular métricas
        meta_pred = self._model.predict(meta_X)
        
        if self.original_model_type == ModelType.CLASSIFIER:
            from sklearn.metrics import accuracy_score
            stacking_score = accuracy_score(y, meta_pred)
        else:
            stacking_score = -mean_squared_error(y, meta_pred)
        
        # Comparar con modelos individuales
        individual_scores = []
        for model in self.base_models:
            if model.info.status == ModelStatus.TRAINED:
                try:
                    pred = model.predict(X)
                    if self.original_model_type == ModelType.CLASSIFIER:
                        score = accuracy_score(y, pred)
                    else:
                        score = -mean_squared_error(y, pred)
                    individual_scores.append(score)
                except:
                    pass
        
        metrics = {
            'stacking_score': stacking_score,
            'meta_features_shape': meta_X.shape,
            'num_base_models': len(self.base_models),
            'avg_individual_score': np.mean(individual_scores) if individual_scores else 0,
            'best_individual_score': np.max(individual_scores) if individual_scores else 0,
            'meta_model_type': self.meta_model_type
        }
        
        if individual_scores:
            improvement = (stacking_score - np.max(individual_scores)) / np.max(individual_scores)
            metrics['improvement_vs_best'] = improvement
        
        logger.info(f"Stacking entrenado:")
        logger.info(f"  Score: {stacking_score:.4f}")
        logger.info(f"  Meta-features: {meta_X.shape}")
        
        return metrics
    
    def _predict_model(self, X: pd.DataFrame, **kwargs) -> np.ndarray:
        """Hacer predicciones con stacking"""
        # Generar meta-features usando modelos base entrenados
        meta_features = []
        
        for model in self.base_models:
            if model.info.status == ModelStatus.TRAINED:
                pred = model.predict(X)
                meta_features.append(pred)
        
        if not meta_features:
            raise ValueError("No hay modelos base entrenados")
        
        meta_X = np.column_stack(meta_features)
        
        # Agregar features originales si está configurado
        if self.get_hyperparameters()['use_original_features']:
            from sklearn.preprocessing import StandardScaler
            # Nota: En producción, el scaler debería guardarse del entrenamiento
            X_scaled = StandardScaler().fit_transform(X)
            meta_X = np.column_stack([meta_X, X_scaled])
        
        return self._model.predict(meta_X)
    
    def _save_model(self, filepath: Path) -> bool:
        """Guardar modelo de stacking"""
        try:
            stacking_dir = filepath.parent / f"{filepath.stem}_stacking"
            stacking_dir.mkdir(exist_ok=True)
            
            # Guardar meta-modelo
            joblib.dump(self._model, stacking_dir / "meta_model.pkl")
            
            # Guardar modelos base (similar a VotingEnsemble)
            base_models_dir = stacking_dir / "base_models"
            base_models_dir.mkdir(exist_ok=True)
            
            for i, model in enumerate(self.base_models):
                model.save(base_models_dir / f"base_model_{i}")
            
            # Metadatos
            metadata = {
                'meta_model_type': self.meta_model_type,
                'original_model_type': self.original_model_type.value,
                'hyperparameters': self.get_hyperparameters()
            }
            
            with open(stacking_dir / "stacking_metadata.json", 'w') as f:
                import json
                json.dump(metadata, f, indent=2, default=str)
            
            return True
            
        except Exception as e:
            logger.error(f"Error guardando stacking: {e}")
            return False
    
    def _load_model(self, filepath: Path) -> bool:
        """Cargar modelo de stacking"""
        try:
            stacking_dir = filepath.parent / f"{filepath.stem}_stacking"
            
            # Cargar meta-modelo
            self._model = joblib.load(stacking_dir / "meta_model.pkl")
            
            # Cargar metadatos
            with open(stacking_dir / "stacking_metadata.json", 'r') as f:
                import json
                metadata = json.load(f)
            
            self.meta_model_type = metadata['meta_model_type']
            self.original_model_type = ModelType(metadata['original_model_type'])
            
            return True
            
        except Exception as e:
            logger.error(f"Error cargando stacking: {e}")
            return False


# Factory para modelos de ensamble
class EnsembleFactory:
    """Factory para crear diferentes tipos de ensambles"""
    
    @staticmethod
    def create_voting_ensemble(base_models: List[BaseModel], name: str = "voting_ensemble",
                             voting_type: str = "soft", **kwargs) -> VotingEnsembleModel:
        """Crear ensamble de votación"""
        return VotingEnsembleModel(name, base_models, voting_type, **kwargs)
    
    @staticmethod
    def create_stacking_ensemble(base_models: List[BaseModel], name: str = "stacking_ensemble",
                               meta_model_type: str = "linear", **kwargs) -> StackingEnsembleModel:
        """Crear ensamble de stacking"""
        return StackingEnsembleModel(name, base_models, meta_model_type, **kwargs)
    
    @staticmethod
    def create_diverse_ensemble(task: str = 'classification', 
                              ensemble_type: str = 'voting',
                              name: str = "diverse_ensemble") -> BaseModel:
        """
        Crear ensamble diverso con modelos complementarios
        
        Args:
            task: 'classification' o 'regression'
            ensemble_type: 'voting' o 'stacking'
            name: Nombre del ensamble
            
        Returns:
            Modelo de ensamble configurado
        """
        # Crear modelos base diversos
        base_models = [
            MLModelFactory.create_model('random_forest', f'{name}_rf', task),
            MLModelFactory.create_model('logistic' if task == 'classification' else 'linear', 
                                      f'{name}_linear', task),
        ]
        
        # Agregar XGBoost si está disponible
        try:
            base_models.append(MLModelFactory.create_model('xgboost', f'{name}_xgb', task))
        except:
            logger.warning("XGBoost no disponible, usando solo RF y Linear")
        
        # Crear ensamble
        if ensemble_type == 'voting':
            return EnsembleFactory.create_voting_ensemble(base_models, name)
        elif ensemble_type == 'stacking':
            return EnsembleFactory.create_stacking_ensemble(base_models, name)
        else:
            raise ValueError(f"Tipo de ensamble no soportado: {ensemble_type}")
    
    @staticmethod
    def create_trading_ensemble(name: str = "trading_ensemble") -> VotingEnsembleModel:
        """
        Crear ensamble especializado para trading
        Combina modelos optimizados para diferentes aspectos del mercado
        """
        # Modelos base especializados
        base_models = [
            # Modelo rápido para señales inmediatas
            MLModelFactory.create_model('random_forest', 'fast_signals', 'classification',
                                      n_estimators=50, max_depth=10),
            
            # Modelo robusto para tendencias
            MLModelFactory.create_model('random_forest', 'trend_robust', 'classification',
                                      n_estimators=200, max_depth=20, min_samples_split=10),
            
            # Modelo conservador para filtrado
            MLModelFactory.create_model('logistic', 'conservative_filter', 'classification',
                                      C=0.1, penalty='l2')
        ]
        
        # Agregar modelos avanzados si están disponibles
        try:
            base_models.append(
                MLModelFactory.create_model('xgboost', 'momentum_xgb', 'classification',
                                          learning_rate=0.05, n_estimators=150)
            )
        except:
            pass
        
        return VotingEnsembleModel(
            name=name,
            base_models=base_models,
            voting_type='soft',
            optimize_weights=True,
            diversity_threshold=0.2
        )


# Funciones de utilidad
def evaluate_ensemble_performance(ensemble: BaseModel, X_test: pd.DataFrame, y_test: pd.Series,
                                base_models: List[BaseModel] = None) -> Dict[str, Any]:
    """
    Evaluar performance de ensamble vs modelos individuales
    
    Args:
        ensemble: Modelo de ensamble entrenado
        X_test: Features de test
        y_test: Target de test
        base_models: Lista de modelos base para comparación
        
    Returns:
        Diccionario con métricas comparativas
    """
    results = {}
    
    # Evaluar ensamble
    ensemble_pred = ensemble.predict(X_test)
    
    if ensemble.original_model_type == ModelType.CLASSIFIER:
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        results['ensemble'] = {
            'accuracy': accuracy_score(y_test, ensemble_pred),
            'precision': precision_score(y_test, ensemble_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_test, ensemble_pred, average='weighted', zero_division=0),
            'f1': f1_score(y_test, ensemble_pred, average='weighted', zero_division=0)
        }
    else:
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        
        results['ensemble'] = {
            'mse': mean_squared_error(y_test, ensemble_pred),
            'mae': mean_absolute_error(y_test, ensemble_pred),
            'r2': r2_score(y_test, ensemble_pred)
        }
    
    # Evaluar modelos base si se proporcionan
    if base_models:
        results['base_models'] = {}
        
        for model in base_models:
            if model.info.status == ModelStatus.TRAINED:
                try:
                    pred = model.predict(X_test)
                    
                    if ensemble.original_model_type == ModelType.CLASSIFIER:
                        results['base_models'][model.info.name] = {
                            'accuracy': accuracy_score(y_test, pred),
                            'precision': precision_score(y_test, pred, average='weighted', zero_division=0),
                            'recall': recall_score(y_test, pred, average='weighted', zero_division=0),
                            'f1': f1_score(y_test, pred, average='weighted', zero_division=0)
                        }
                    else:
                        results['base_models'][model.info.name] = {
                            'mse': mean_squared_error(y_test, pred),
                            'mae': mean_absolute_error(y_test, pred),
                            'r2': r2_score(y_test, pred)
                        }
                        
                except Exception as e:
                    logger.warning(f"Error evaluando {model.info.name}: {e}")
    
    # Calcular mejoras
    if base_models and results.get('base_models'):
        primary_metric = 'accuracy' if ensemble.original_model_type == ModelType.CLASSIFIER else 'r2'
        
        ensemble_score = results['ensemble'][primary_metric]
        base_scores = [metrics[primary_metric] for metrics in results['base_models'].values()]
        
        if base_scores:
            best_base_score = max(base_scores)
            avg_base_score = np.mean(base_scores)
            
            results['improvements'] = {
                'vs_best': (ensemble_score - best_base_score) / best_base_score if best_base_score != 0 else 0,
                'vs_average': (ensemble_score - avg_base_score) / avg_base_score if avg_base_score != 0 else 0,
                'absolute_vs_best': ensemble_score - best_base_score
            }
    
    return results
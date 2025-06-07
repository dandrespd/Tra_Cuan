from typing import Dict, List, Any
import optuna
from sklearn.model_selection import TimeSeriesSplit
from models.model_hub import ModelHub
from utils.log_config import get_logger

@dataclass
class OptimizationObjective:
    """Objetivo de optimización"""
    primary_metric: str  # 'sharpe_ratio', 'accuracy', 'profit_factor'
    secondary_metrics: List[str]
    constraints: Dict[str, Any]  # {'max_drawdown': 0.15, 'min_win_rate': 0.55}
    optimization_type: str  # 'single', 'multi_objective'
    time_budget_minutes: int = 60
    resource_budget: Dict[str, Any] = field(default_factory=dict)  # CPU, memory limits

class MLOptimizer:
    """Optimizador principal de modelos ML"""
    
    def __init__(self, model_hub: ModelHub, 
                objective: OptimizationObjective):
        self.model_hub = model_hub
        self.objective = objective
        
        # Optuna para optimización Bayesiana
        self.study = optuna.create_study(
            direction='maximize' if objective.optimization_type == 'single' else None,
            study_name=f"ml_optimization_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            load_if_exists=True
        )
        
        # Histórico de optimizaciones
        self.optimization_history = []
        
        # Configuración de búsqueda
        self.search_space = SearchSpaceDefinition()
        
        # Evaluador de modelos
        self.model_evaluator = ModelEvaluator()
        
        # Cache de evaluaciones
        self.evaluation_cache = {}
        
    def optimize_model(self, model_type: str, 
                      training_data: Tuple[pd.DataFrame, pd.Series],
                      validation_data: Tuple[pd.DataFrame, pd.Series],
                      n_trials: int = 50,
                      timeout: Optional[int] = None) -> Dict[str, Any]:
        """Optimiza hiperparámetros de un modelo específico"""
        
        X_train, y_train = training_data
        X_val, y_val = validation_data
        
        def objective_function(trial):
            # Hash para cache
            param_hash = self._hash_params(trial.params)
            if param_hash in self.evaluation_cache:
                return self.evaluation_cache[param_hash]
            
            # Sugerir hiperparámetros
            params = self._suggest_hyperparameters(trial, model_type)
            
            # Crear y entrenar modelo
            model = self._create_model_with_params(model_type, params)
            
            # Cross-validation temporal
            cv_scores = self._temporal_cross_validation(
                model, X_train, y_train, n_splits=5
            )
            
            # Evaluar en validación
            model.fit(X_train, y_train)
            val_score = self._evaluate_model_objective(model, X_val, y_val)
            
            # Score combinado
            score = 0.7 * cv_scores.mean() + 0.3 * val_score
            
            # Verificar constraints
            constraint_penalty = self._calculate_constraint_penalty(model, X_val, y_val)
            score -= constraint_penalty
            
            # Cachear resultado
            self.evaluation_cache[param_hash] = score
            
            # Pruning para ahorrar tiempo
            trial.report(score, trial.number)
            if trial.should_prune():
                raise optuna.TrialPruned()
            
            return score
        
        # Callbacks para monitoreo
        callbacks = [
            self._optimization_callback,
            optuna.study.MaxTrialsCallback(n_trials),
            self._early_stopping_callback
        ]
        
        # Ejecutar optimización
        self.study.optimize(
            objective_function,
            n_trials=n_trials,
            timeout=timeout,
            callbacks=callbacks,
            n_jobs=1  # Para evitar problemas con MT5
        )
        
        # Obtener mejores parámetros
        best_params = self.study.best_params
        best_score = self.study.best_value
        
        # Análisis de importancia de hiperparámetros
        param_importance = self._analyze_parameter_importance()
        
        # Crear modelo final con mejores parámetros
        best_model = self._create_model_with_params(model_type, best_params)
        
        # Entrenamiento final con todos los datos
        X_full = pd.concat([X_train, X_val])
        y_full = pd.concat([y_train, y_val])
        best_model.fit(X_full, y_full)
        
        # Registrar en model hub
        model_name = f"{model_type}_optimized_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.model_hub.register_model(
            best_model,
            tags=['optimized', model_type, f'score_{best_score:.4f}']
        )
        
        # Guardar resultados
        optimization_result = {
            'best_params': best_params,
            'best_score': best_score,
            'model_name': model_name,
            'n_trials': len(self.study.trials),
            'param_importance': param_importance,
            'optimization_history': self._get_optimization_history(),
            'convergence_analysis': self._analyze_convergence()
        }
        
        self.optimization_history.append(optimization_result)
        
        return optimization_result
    
    def _suggest_hyperparameters(self, trial: optuna.Trial, 
                                model_type: str) -> Dict[str, Any]:
        """Sugiere hiperparámetros según el tipo de modelo"""
        
        search_space = self.search_space.get_space(model_type)
        params = {}
        
        for param_name, param_config in search_space.items():
            if param_config['type'] == 'int':
                params[param_name] = trial.suggest_int(
                    param_name, 
                    param_config['low'], 
                    param_config['high'],
                    step=param_config.get('step', 1)
                )
            elif param_config['type'] == 'float':
                params[param_name] = trial.suggest_float(
                    param_name,
                    param_config['low'],
                    param_config['high'],
                    log=param_config.get('log', False)
                )
            elif param_config['type'] == 'categorical':
                params[param_name] = trial.suggest_categorical(
                    param_name,
                    param_config['choices']
                )
        
        return params
    
    def _temporal_cross_validation(self, model, X: pd.DataFrame, y: pd.Series,
                                 n_splits: int = 5) -> np.ndarray:
        """Cross-validation respetando orden temporal"""
        tscv = TimeSeriesSplit(n_splits=n_splits)
        scores = []
        
        for train_idx, val_idx in tscv.split(X):
            X_train_cv = X.iloc[train_idx]
            y_train_cv = y.iloc[train_idx]
            X_val_cv = X.iloc[val_idx]
            y_val_cv = y.iloc[val_idx]
            
            # Clonar modelo para evitar contaminación
            model_clone = clone(model)
            model_clone.fit(X_train_cv, y_train_cv)
            
            score = self._evaluate_model_objective(model_clone, X_val_cv, y_val_cv)
            scores.append(score)
        
        return np.array(scores)

class AutoMLPipeline:
    """Pipeline completo de AutoML para trading"""
    
    def __init__(self, optimization_config: Dict[str, Any]):
        self.config = optimization_config
        self.pipeline_stages = {
            'data_preprocessing': DataPreprocessingOptimizer(),
            'feature_engineering': FeatureEngineeringOptimizer(),
            'feature_selection': FeatureSelectionOptimizer(),
            'model_selection': ModelSelectionOptimizer(),
            'hyperparameter_tuning': HyperparameterOptimizer(),
            'ensemble_creation': EnsembleOptimizer()
        }
        
        # Resultados de cada etapa
        self.stage_results = {}
        
    def run_automl(self, raw_data: pd.DataFrame, 
                   target: pd.Series,
                   time_budget_minutes: int = 60,
                   validation_strategy: str = 'temporal_split') -> Dict[str, Any]:
        """Ejecuta pipeline completo de AutoML"""
        
        start_time = datetime.now()
        time_allocator = TimeAllocator(time_budget_minutes)
        
        # Dividir datos según estrategia
        train_data, val_data = self._split_data(
            raw_data, target, strategy=validation_strategy
        )
        
        results = {'start_time': start_time}
        
        # 1. Optimizar preprocesamiento
        with time_allocator.allocate_time('preprocessing', 0.15):
            preprocessing_result = self._optimize_preprocessing(train_data)
            self.stage_results['preprocessing'] = preprocessing_result
            processed_data = preprocessing_result['transformed_data']
        
        # 2. Ingeniería de features
        with time_allocator.allocate_time('feature_engineering', 0.20):
            feature_result = self._optimize_features(processed_data, target)
            self.stage_results['feature_engineering'] = feature_result
            engineered_data = feature_result['engineered_data']
        
        # 3. Selección de features
        with time_allocator.allocate_time('feature_selection', 0.15):
            selection_result = self._select_features(engineered_data, target)
            self.stage_results['feature_selection'] = selection_result
            selected_data = engineered_data[selection_result['selected_features']]
        
        # 4. Selección de modelos
        with time_allocator.allocate_time('model_selection', 0.20):
            model_selection_result = self._select_models(selected_data, target)
            self.stage_results['model_selection'] = model_selection_result
            candidate_models = model_selection_result['top_models']
        
        # 5. Optimización de hiperparámetros
        with time_allocator.allocate_time('hyperparameter_tuning', 0.25):
            optimized_models = self._optimize_hyperparameters(
                candidate_models, selected_data, target
            )
            self.stage_results['hyperparameter_tuning'] = optimized_models
        
        # 6. Crear ensemble
        with time_allocator.allocate_time('ensemble_creation', 0.05):
            if len(optimized_models) > 1:
                ensemble_result = self._create_ensemble(
                    optimized_models, selected_data, target
                )
                self.stage_results['ensemble'] = ensemble_result
        
        # Compilar resultados finales
        results.update({
            'pipeline_stages': self.stage_results,
            'final_model': self._select_final_model(),
            'total_time': (datetime.now() - start_time).total_seconds(),
            'feature_importance': self._get_global_feature_importance(),
            'performance_summary': self._compile_performance_summary()
        })
        
        return results

class NeuralArchitectureSearch:
    """Búsqueda de arquitectura neural para modelos deep learning"""
    
    def __init__(self, search_space: Dict[str, Any]):
        self.search_space = search_space
        self.architecture_evaluator = ArchitectureEvaluator()
        self.search_history = []
        
    def search_architecture(self, input_shape: Tuple, 
                          output_shape: Tuple,
                          training_data: Tuple[np.ndarray, np.ndarray],
                          search_strategy: str = 'evolutionary',
                          max_trials: int = 50) -> Dict[str, Any]:
        """Busca arquitectura óptima de red neural"""
        
        # Inicializar estrategia de búsqueda
        if search_strategy == 'evolutionary':
            searcher = self._create_evolutionary_searcher()
        elif search_strategy == 'reinforcement_learning':
            searcher = self._create_rl_searcher()
        elif search_strategy == 'bayesian':
            searcher = self._create_bayesian_searcher()
        else:
            searcher = RandomSearcher()
        
        best_architecture = None
        best_score = float('-inf')
        
        # Progress tracking
        progress_tracker = ProgressTracker(max_trials)
        
        for trial in range(max_trials):
            # Generar arquitectura candidata
            architecture = searcher.generate_architecture(
                self.search_space, 
                input_shape, 
                output_shape
            )
            
            # Early stopping si no hay mejora
            if progress_tracker.should_stop():
                logger.info("Early stopping: No mejora en últimas iteraciones")
                break
            
            # Construir y evaluar modelo
            try:
                model = self._build_model_from_architecture(
                    architecture, input_shape, output_shape
                )
                
                # Evaluación rápida con subset de datos
                score = self.architecture_evaluator.quick_evaluate(
                    model, training_data, epochs=5
                )
                
                # Si es prometedor, evaluación completa
                if score > best_score * 0.9:
                    full_score = self.architecture_evaluator.full_evaluate(
                        model, training_data, epochs=20
                    )
                    score = full_score
                
            except Exception as e:
                logger.warning(f"Error evaluando arquitectura: {e}")
                score = float('-inf')
            
            # Actualizar mejor arquitectura
            if score > best_score:
                best_score = score
                best_architecture = architecture
                progress_tracker.update_best(score)
            
            # Actualizar searcher con feedback
            searcher.update(architecture, score)
            
            # Guardar en historial
            self.search_history.append({
                'trial': trial,
                'architecture': architecture,
                'score': score,
                'timestamp': datetime.now()
            })
            
            # Log progress
            if trial % 10 == 0:
                logger.info(f"NAS Trial {trial + 1}/{max_trials}: "
                          f"Best Score = {best_score:.4f}")
        
        # Análisis post-búsqueda
        analysis = self._analyze_search_results()
        
        return {
            'best_architecture': best_architecture,
            'best_score': best_score,
            'total_trials': len(self.search_history),
            'search_history': self.search_history,
            'analysis': analysis,
            'final_model': self._create_final_model(best_architecture)
        }
    
    def _build_model_from_architecture(self, architecture: Dict[str, Any],
                                     input_shape: Tuple,
                                     output_shape: Tuple) -> tf.keras.Model:
        """Construye modelo Keras desde descripción de arquitectura"""
        
        model = tf.keras.Sequential()
        
        # Input layer
        model.add(tf.keras.layers.Input(shape=input_shape))
        
        # Hidden layers según arquitectura
        for layer_config in architecture['layers']:
            layer_type = layer_config['type']
            
            if layer_type == 'dense':
                model.add(tf.keras.layers.Dense(
                    units=layer_config['units'],
                    activation=layer_config.get('activation', 'relu')
                ))
            
            elif layer_type == 'lstm':
                model.add(tf.keras.layers.LSTM(
                    units=layer_config['units'],
                    return_sequences=layer_config.get('return_sequences', False),
                    dropout=layer_config.get('dropout', 0.0)
                ))
            
            elif layer_type == 'conv1d':
                model.add(tf.keras.layers.Conv1D(
                    filters=layer_config['filters'],
                    kernel_size=layer_config['kernel_size'],
                    activation=layer_config.get('activation', 'relu'),
                    padding=layer_config.get('padding', 'same')
                ))
            
            elif layer_type == 'dropout':
                model.add(tf.keras.layers.Dropout(
                    rate=layer_config['rate']
                ))
            
            elif layer_type == 'batch_norm':
                model.add(tf.keras.layers.BatchNormalization())
        
        # Output layer
        model.add(tf.keras.layers.Dense(
            units=output_shape[0],
            activation=architecture.get('output_activation', 'sigmoid')
        ))
        
        # Compilar modelo
        model.compile(
            optimizer=self._create_optimizer(architecture.get('optimizer', {})),
            loss=architecture.get('loss', 'binary_crossentropy'),
            metrics=['accuracy']
        )
        
        return model

class FeatureSelectionOptimizer:
    """Optimiza selección de features"""
    
    def __init__(self):
        self.selection_methods = {
            'mutual_information': self._mutual_information_selection,
            'recursive_elimination': self._recursive_feature_elimination,
            'lasso': self._lasso_selection,
            'genetic_algorithm': self._genetic_algorithm_selection,
            'boruta': self._boruta_selection,
            'shap_selection': self._shap_based_selection
        }
        
        self.ensemble_selector = EnsembleFeatureSelector()
        
    def optimize_feature_selection(self, X: pd.DataFrame, y: pd.Series,
                                 method: str = 'auto',
                                 target_features: Optional[int] = None) -> Dict[str, Any]:
        """Optimiza selección de features"""
        
        if method == 'auto':
            # Probar múltiples métodos y combinar
            return self._auto_select_features(X, y, target_features)
        
        elif method in self.selection_methods:
            selector_func = self.selection_methods[method]
            return selector_func(X, y, target_features)
        
        else:
            raise ValueError(f"Método no soportado: {method}")
    
    def _auto_select_features(self, X: pd.DataFrame, y: pd.Series,
                            target_features: Optional[int] = None) -> Dict[str, Any]:
        """Selección automática usando múltiples métodos"""
        
        # Determinar número objetivo de features
        if target_features is None:
            target_features = min(int(np.sqrt(len(X.columns))), 50)
        
        # Aplicar diferentes métodos
        selection_results = {}
        
        for method_name, method_func in self.selection_methods.items():
            try:
                result = method_func(X, y, target_features)
                selection_results[method_name] = result
            except Exception as e:
                logger.warning(f"Error en {method_name}: {e}")
        
        # Combinar resultados con ensemble
        final_selection = self.ensemble_selector.combine_selections(
            selection_results, X, y
        )
        
        # Validar selección
        validation_score = self._validate_feature_selection(
            X[final_selection['features']], y
        )
        
        return {
            'selected_features': final_selection['features'],
            'n_features': len(final_selection['features']),
            'validation_score': validation_score,
            'method_scores': selection_results,
            'feature_ranking': final_selection['ranking']
        }
    
    def _shap_based_selection(self, X: pd.DataFrame, y: pd.Series,
                            target_features: int) -> Dict[str, Any]:
        """Selección basada en valores SHAP"""
        
        # Entrenar modelo para SHAP
        model = XGBClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)
        
        # Calcular valores SHAP
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        
        # Feature importance basada en SHAP
        shap_importance = np.abs(shap_values).mean(axis=0)
        
        # Seleccionar top features
        indices = np.argsort(shap_importance)[-target_features:]
        selected_features = X.columns[indices].tolist()
        
        return {
            'features': selected_features,
            'importance_scores': shap_importance[indices],
            'method': 'shap'
        }

class HyperparameterOptimizer:
    """Optimización avanzada de hiperparámetros"""
    
    def __init__(self):
        self.optimization_strategies = {
            'bayesian': self._bayesian_optimization,
            'genetic': self._genetic_optimization,
            'hyperband': self._hyperband_optimization,
            'population_based': self._population_based_training
        }
        
    def optimize(self, model_class, param_space: Dict[str, Any],
                X_train: pd.DataFrame, y_train: pd.Series,
                X_val: pd.DataFrame, y_val: pd.Series,
                strategy: str = 'bayesian',
                budget: int = 100) -> Dict[str, Any]:
        """Optimiza hiperparámetros con estrategia especificada"""
        
        optimizer_func = self.optimization_strategies.get(strategy)
        
        if not optimizer_func:
            raise ValueError(f"Estrategia no soportada: {strategy}")
        
        return optimizer_func(
            model_class, param_space, 
            X_train, y_train, X_val, y_val, 
            budget
        )
    
    def _hyperband_optimization(self, model_class, param_space,
                              X_train, y_train, X_val, y_val,
                              budget) -> Dict[str, Any]:
        """Optimización usando Hyperband (eficiente para deep learning)"""
        
        # Configuración de Hyperband
        max_iter = 81  # Máximo número de épocas
        eta = 3  # Factor de reducción
        
        # Calcular número de configuraciones
        logeta = lambda x: np.log(x) / np.log(eta)
        s_max = int(logeta(max_iter))
        B = (s_max + 1) * budget
        
        best_config = None
        best_score = float('-inf')
        
        for s in reversed(range(s_max + 1)):
            # Número de configuraciones
            n = int(np.ceil(B / max_iter / (s + 1) * eta ** s))
            # Recursos iniciales
            r = max_iter * eta ** (-s)
            
            # Generar configuraciones aleatorias
            configs = []
            for _ in range(n):
                config = self._sample_configuration(param_space)
                configs.append(config)
            
            # Successive halving
            for i in range(s + 1):
                # Entrenar cada configuración
                n_configs = len(configs)
                scores = []
                
                for config in configs:
                    model = model_class(**config)
                    score = self._train_and_evaluate(
                        model, X_train, y_train, X_val, y_val,
                        epochs=int(r)
                    )
                    scores.append((score, config))
                
                # Seleccionar mejores configuraciones
                scores.sort(reverse=True)
                n_keep = int(n_configs / eta)
                configs = [config for _, config in scores[:n_keep]]
                
                # Actualizar mejor configuración
                if scores[0][0] > best_score:
                    best_score = scores[0][0]
                    best_config = scores[0][1]
                
                # Aumentar recursos
                r *= eta
        
        return {
            'best_params': best_config,
            'best_score': best_score,
            'optimization_method': 'hyperband'
        }

class SearchSpaceDefinition:
    """Define espacios de búsqueda para diferentes modelos"""
    
    def __init__(self):
        self.spaces = {
            'random_forest': {
                'n_estimators': {'type': 'int', 'low': 50, 'high': 500},
                'max_depth': {'type': 'int', 'low': 5, 'high': 50},
                'min_samples_split': {'type': 'int', 'low': 2, 'high': 100},
                'min_samples_leaf': {'type': 'int', 'low': 1, 'high': 50},
                'max_features': {'type': 'categorical', 'choices': ['sqrt', 'log2', None]},
                'bootstrap': {'type': 'categorical', 'choices': [True, False]}
            },
            
            'xgboost': {
                'n_estimators': {'type': 'int', 'low': 50, 'high': 1000},
                'max_depth': {'type': 'int', 'low': 3, 'high': 12},
                'learning_rate': {'type': 'float', 'low': 0.01, 'high': 0.3, 'log': True},
                'subsample': {'type': 'float', 'low': 0.5, 'high': 1.0},
                'colsample_bytree': {'type': 'float', 'low': 0.5, 'high': 1.0},
                'gamma': {'type': 'float', 'low': 0, 'high': 5},
                'reg_alpha': {'type': 'float', 'low': 0, 'high': 2},
                'reg_lambda': {'type': 'float', 'low': 0, 'high': 2}
            },
            
            'lightgbm': {
                'num_leaves': {'type': 'int', 'low': 20, 'high': 300},
                'learning_rate': {'type': 'float', 'low': 0.01, 'high': 0.3, 'log': True},
                'feature_fraction': {'type': 'float', 'low': 0.5, 'high': 1.0},
                'bagging_fraction': {'type': 'float', 'low': 0.5, 'high': 1.0},
                'bagging_freq': {'type': 'int', 'low': 1, 'high': 10},
                'min_data_in_leaf': {'type': 'int', 'low': 10, 'high': 100},
                'lambda_l1': {'type': 'float', 'low': 0, 'high': 2},
                'lambda_l2': {'type': 'float', 'low': 0, 'high': 2}
            },
            
            'lstm': {
                'lstm_units': {'type': 'categorical', 'choices': [32, 64, 128, 256]},
                'dropout_rate': {'type': 'float', 'low': 0.1, 'high': 0.5},
                'learning_rate': {'type': 'float', 'low': 0.0001, 'high': 0.01, 'log': True},
                'batch_size': {'type': 'categorical', 'choices': [16, 32, 64, 128]},
                'n_layers': {'type': 'int', 'low': 1, 'high': 4},
                'activation': {'type': 'categorical', 'choices': ['relu', 'tanh', 'sigmoid']},
                'optimizer': {'type': 'categorical', 'choices': ['adam', 'rmsprop', 'sgd']}
            }
        }
    
    def get_space(self, model_type: str) -> Dict[str, Any]:
        """Obtiene espacio de búsqueda para un modelo"""
        return self.spaces.get(model_type, {})
    
    def add_custom_space(self, model_type: str, space: Dict[str, Any]):
        """Agrega espacio de búsqueda personalizado"""
        self.spaces[model_type] = space

class ModelEvaluator:
    """Evaluador de modelos con métricas específicas de trading"""
    
    def __init__(self):
        self.metrics_calculators = {
            'accuracy': self._calculate_accuracy,
            'sharpe_ratio': self._calculate_sharpe_ratio,
            'profit_factor': self._calculate_profit_factor,
            'max_drawdown': self._calculate_max_drawdown,
            'calmar_ratio': self._calculate_calmar_ratio,
            'win_rate': self._calculate_win_rate
        }
    
    def evaluate(self, model, X: pd.DataFrame, y: pd.Series,
                metric: str = 'sharpe_ratio') -> float:
        """Evalúa modelo con métrica especificada"""
        
        # Generar predicciones
        predictions = model.predict(X)
        
        # Si es modelo de probabilidad, obtener probabilidades
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(X)
        else:
            probabilities = None
        
        # Calcular métrica
        if metric in self.metrics_calculators:
            return self.metrics_calculators[metric](
                y, predictions, probabilities, X
            )
        else:
            raise ValueError(f"Métrica no soportada: {metric}")
    
    def _calculate_sharpe_ratio(self, y_true, y_pred, probas, X):
        """Calcula Sharpe ratio simulado"""
        # Simular returns basados en predicciones
        returns = []
        
        for i in range(len(y_true)):
            if y_pred[i] == 1 and y_true.iloc[i] == 1:
                returns.append(0.01)  # Win
            elif y_pred[i] == 0 and y_true.iloc[i] == 0:
                returns.append(0.01)  # Win
            else:
                returns.append(-0.01)  # Loss
        
        returns = np.array(returns)
        
        if len(returns) < 2 or returns.std() == 0:
            return 0
        
        sharpe = returns.mean() / returns.std() * np.sqrt(252)
        return sharpe
'''
11. models/deep_models.py
Ruta: TradingBot_Cuantitative_MT5/models/deep_models.py
Resumen:

Implementa modelos de Deep Learning especializados para trading financiero
Incluye LSTM, GRU, CNN, Transformers y arquitecturas híbridas
Optimización automática de arquitecturas y hiperparámetros
Integración completa con sistema de logging y métricas de trading
'''
# models/deep_models.py
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Union, Tuple, Callable
from datetime import datetime
import warnings
import json
from pathlib import Path
import pickle
warnings.filterwarnings('ignore')

# Local imports
from models.base_model import BaseModel, ModelType, ModelStatus, ModelMetrics
from utils.log_config import get_logger, log_performance

# TensorFlow/Keras imports
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, Model, Sequential
    from tensorflow.keras.optimizers import Adam, RMSprop, SGD
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
    from tensorflow.keras.regularizers import l1, l2, l1_l2
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

# PyTorch imports
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False

# Sklearn for preprocessing
try:
    from sklearn.preprocessing import MinMaxScaler, StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

logger = get_logger('models')


class KerasModelBase(BaseModel):
    """Clase base para modelos de Keras/TensorFlow"""
    
    def __init__(self, name: str, model_type: ModelType, version: str = "1.0.0",
                 description: str = "Keras Deep Learning Model"):
        
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow/Keras no está disponible")
        
        super().__init__(name, model_type, version, description)
        
        # Configuración por defecto
        self.sequence_length = 60  # Ventana temporal
        self.batch_size = 32
        self.epochs = 100
        self.patience = 15
        self.learning_rate = 0.001
        self.validation_split = 0.2
        
        # Callbacks
        self.callbacks = []
        self.history = None
        
        # Escaladores
        self.feature_scaler = None
        self.target_scaler = None
        
        logger.info(f"KerasModelBase inicializado: {name}")
    
    def _prepare_sequential_data(self, X: pd.DataFrame, y: pd.Series = None,
                               sequence_length: int = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Preparar datos secuenciales para modelos temporales"""
        
        if sequence_length is None:
            sequence_length = self.sequence_length
        
        # Normalizar features
        if self.feature_scaler is None:
            self.feature_scaler = MinMaxScaler()
            X_scaled = self.feature_scaler.fit_transform(X)
        else:
            X_scaled = self.feature_scaler.transform(X)
        
        # Crear secuencias
        X_sequences = []
        y_sequences = []
        
        for i in range(sequence_length, len(X_scaled)):
            X_sequences.append(X_scaled[i-sequence_length:i])
            
            if y is not None:
                y_sequences.append(y.iloc[i])
        
        X_sequences = np.array(X_sequences)
        
        if y is not None:
            y_sequences = np.array(y_sequences)
            
            # Normalizar target para regresión
            if self.info.model_type == ModelType.REGRESSOR:
                if self.target_scaler is None:
                    self.target_scaler = MinMaxScaler()
                    y_sequences = self.target_scaler.fit_transform(y_sequences.reshape(-1, 1)).flatten()
                else:
                    y_sequences = self.target_scaler.transform(y_sequences.reshape(-1, 1)).flatten()
            
            return X_sequences, y_sequences
        
        return X_sequences, None
    
    def _setup_callbacks(self, validation_data: bool = False):
        """Configurar callbacks para entrenamiento"""
        self.callbacks = []
        
        # Early stopping
        if validation_data:
            early_stopping = EarlyStopping(
                monitor='val_loss',
                patience=self.patience,
                restore_best_weights=True,
                verbose=1
            )
            self.callbacks.append(early_stopping)
            
            # Reduce learning rate on plateau
            reduce_lr = ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=self.patience // 2,
                min_lr=1e-7,
                verbose=1
            )
            self.callbacks.append(reduce_lr)
        
        # Model checkpoint (opcional)
        # checkpoint = ModelCheckpoint(...)
        # self.callbacks.append(checkpoint)
    
    def _predict_model(self, X: pd.DataFrame, **kwargs) -> np.ndarray:
        """Hacer predicciones con modelo Keras"""
        # Preparar datos secuenciales
        X_seq, _ = self._prepare_sequential_data(X)
        
        if len(X_seq) == 0:
            logger.warning("No hay suficientes datos para crear secuencias")
            return np.array([])
        
        # Predicción
        predictions = self._model.predict(X_seq, verbose=0)
        
        # Desnormalizar si es regresión
        if (self.info.model_type == ModelType.REGRESSOR and 
            self.target_scaler is not None):
            predictions = self.target_scaler.inverse_transform(
                predictions.reshape(-1, 1)
            ).flatten()
        
        # Para clasificación, convertir probabilidades a clases
        if self.info.model_type == ModelType.CLASSIFIER:
            if predictions.shape[1] == 1:  # Clasificación binaria
                predictions = (predictions > 0.5).astype(int).flatten()
            else:  # Multiclase
                predictions = np.argmax(predictions, axis=1)
        
        return predictions
    
    def _save_model(self, filepath: Path) -> bool:
        """Guardar modelo Keras"""
        try:
            # Guardar arquitectura y pesos
            self._model.save(str(filepath))
            
            # Guardar escaladores
            scalers_path = filepath.parent / f"{filepath.stem}_scalers.pkl"
            with open(scalers_path, 'wb') as f:
                pickle.dump({
                    'feature_scaler': self.feature_scaler,
                    'target_scaler': self.target_scaler
                }, f)
            
            return True
        except Exception as e:
            logger.error(f"Error guardando modelo Keras: {e}")
            return False
    
    def _load_model(self, filepath: Path) -> bool:
        """Cargar modelo Keras"""
        try:
            # Cargar modelo
            self._model = keras.models.load_model(str(filepath))
            
            # Cargar escaladores
            scalers_path = filepath.parent / f"{filepath.stem}_scalers.pkl"
            if scalers_path.exists():
                with open(scalers_path, 'rb') as f:
                    scalers = pickle.load(f)
                    self.feature_scaler = scalers.get('feature_scaler')
                    self.target_scaler = scalers.get('target_scaler')
            
            return True
        except Exception as e:
            logger.error(f"Error cargando modelo Keras: {e}")
            return False


class LSTMModel(KerasModelBase):
    """Modelo LSTM para predicción de series temporales"""
    
    def __init__(self, name: str, model_type: ModelType, version: str = "1.0.0",
                 lstm_units: List[int] = None, dropout: float = 0.2,
                 recurrent_dropout: float = 0.2, **kwargs):
        
        super().__init__(name, model_type, version, "LSTM Model for Time Series")
        
        # Arquitectura
        self.lstm_units = lstm_units or [50, 30]
        self.dropout = dropout
        self.recurrent_dropout = recurrent_dropout
        
        # Actualizar hiperparámetros
        self.info.hyperparameters.update({
            'lstm_units': self.lstm_units,
            'dropout': dropout,
            'recurrent_dropout': recurrent_dropout,
            'sequence_length': self.sequence_length,
            **kwargs
        })
    
    def _build_model(self, input_shape: Tuple[int, int] = None, **kwargs) -> Model:
        """Construir arquitectura LSTM"""
        
        if input_shape is None:
            # Forma por defecto: (sequence_length, n_features)
            input_shape = (self.sequence_length, 10)  # Se ajustará dinámicamente
        
        model = Sequential(name=f"LSTM_{self.info.name}")
        
        # Capa de entrada
        model.add(layers.Input(shape=input_shape))
        
        # Capas LSTM
        for i, units in enumerate(self.lstm_units):
            return_sequences = i < len(self.lstm_units) - 1  # Última capa no retorna secuencias
            
            model.add(layers.LSTM(
                units=units,
                return_sequences=return_sequences,
                dropout=self.dropout,
                recurrent_dropout=self.recurrent_dropout,
                name=f"lstm_{i+1}"
            ))
        
        # Capa de regularización
        model.add(layers.Dropout(self.dropout))
        
        # Capas densas
        model.add(layers.Dense(32, activation='relu', name='dense_1'))
        model.add(layers.Dropout(self.dropout))
        
        # Capa de salida
        if self.info.model_type == ModelType.CLASSIFIER:
            output_units = 1
            activation = 'sigmoid'
            loss = 'binary_crossentropy'
            metrics = ['accuracy']
        else:
            output_units = 1
            activation = 'linear'
            loss = 'mse'
            metrics = ['mae']
        
        model.add(layers.Dense(output_units, activation=activation, name='output'))
        
        # Compilar
        optimizer = Adam(learning_rate=self.learning_rate)
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        
        logger.info(f"Modelo LSTM construido:")
        logger.info(f"  - Unidades LSTM: {self.lstm_units}")
        logger.info(f"  - Dropout: {self.dropout}")
        logger.info(f"  - Parámetros totales: {model.count_params():,}")
        
        return model
    
    def _fit_model(self, X: pd.DataFrame, y: pd.Series,
                   validation_data: Optional[Tuple[pd.DataFrame, pd.Series]] = None,
                   **kwargs) -> Dict[str, Any]:
        """Entrenar modelo LSTM"""
        
        # Preparar datos secuenciales
        X_seq, y_seq = self._prepare_sequential_data(X, y)
        
        if len(X_seq) == 0:
            raise ValueError("No hay suficientes datos para crear secuencias")
        
        # Actualizar forma de entrada
        input_shape = (X_seq.shape[1], X_seq.shape[2])
        
        # Construir modelo con forma correcta
        self._model = self._build_model(input_shape=input_shape, **kwargs)
        
        # Preparar datos de validación
        validation_data_processed = None
        if validation_data:
            X_val, y_val = validation_data
            X_val_seq, y_val_seq = self._prepare_sequential_data(X_val, y_val)
            if len(X_val_seq) > 0:
                validation_data_processed = (X_val_seq, y_val_seq)
        
        # Configurar callbacks
        self._setup_callbacks(validation_data_processed is not None)
        
        # Entrenar
        start_time = datetime.now()
        
        self.history = self._model.fit(
            X_seq, y_seq,
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_data=validation_data_processed,
            callbacks=self.callbacks,
            verbose=1
        )
        
        training_time = (datetime.now() - start_time).total_seconds()
        
        # Extraer métricas del historial
        metrics = {
            'training_time': training_time,
            'epochs_trained': len(self.history.history['loss']),
            'final_train_loss': self.history.history['loss'][-1]
        }
        
        if validation_data_processed:
            metrics['final_val_loss'] = self.history.history['val_loss'][-1]
            
            # Mejor época basada en validation loss
            best_epoch = np.argmin(self.history.history['val_loss'])
            metrics['best_epoch'] = best_epoch + 1
            metrics['best_val_loss'] = self.history.history['val_loss'][best_epoch]
        
        # Métricas adicionales según el tipo
        if self.info.model_type == ModelType.CLASSIFIER:
            metrics['final_train_accuracy'] = self.history.history['accuracy'][-1]
            if validation_data_processed:
                metrics['final_val_accuracy'] = self.history.history['val_accuracy'][-1]
        
        logger.info(f"Entrenamiento LSTM completado:")
        logger.info(f"  - Épocas: {metrics['epochs_trained']}")
        logger.info(f"  - Tiempo: {training_time:.2f}s")
        logger.info(f"  - Loss final: {metrics['final_train_loss']:.6f}")
        
        return metrics


class GRUModel(KerasModelBase):
    """Modelo GRU para predicción de series temporales"""
    
    def __init__(self, name: str, model_type: ModelType, version: str = "1.0.0",
                 gru_units: List[int] = None, dropout: float = 0.2,
                 recurrent_dropout: float = 0.2, **kwargs):
        
        super().__init__(name, model_type, version, "GRU Model for Time Series")
        
        # Arquitectura
        self.gru_units = gru_units or [50, 30]
        self.dropout = dropout
        self.recurrent_dropout = recurrent_dropout
        
        # Hiperparámetros
        self.info.hyperparameters.update({
            'gru_units': self.gru_units,
            'dropout': dropout,
            'recurrent_dropout': recurrent_dropout,
            **kwargs
        })
    
    def _build_model(self, input_shape: Tuple[int, int] = None, **kwargs) -> Model:
        """Construir arquitectura GRU"""
        
        if input_shape is None:
            input_shape = (self.sequence_length, 10)
        
        model = Sequential(name=f"GRU_{self.info.name}")
        
        # Input
        model.add(layers.Input(shape=input_shape))
        
        # Capas GRU
        for i, units in enumerate(self.gru_units):
            return_sequences = i < len(self.gru_units) - 1
            
            model.add(layers.GRU(
                units=units,
                return_sequences=return_sequences,
                dropout=self.dropout,
                recurrent_dropout=self.recurrent_dropout,
                name=f"gru_{i+1}"
            ))
        
        # Regularización
        model.add(layers.Dropout(self.dropout))
        
        # Capas densas
        model.add(layers.Dense(32, activation='relu'))
        model.add(layers.Dropout(self.dropout))
        
        # Salida
        if self.info.model_type == ModelType.CLASSIFIER:
            output_units = 1
            activation = 'sigmoid'
            loss = 'binary_crossentropy'
            metrics = ['accuracy']
        else:
            output_units = 1
            activation = 'linear'
            loss = 'mse'
            metrics = ['mae']
        
        model.add(layers.Dense(output_units, activation=activation))
        
        # Compilar
        optimizer = Adam(learning_rate=self.learning_rate)
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        
        logger.info(f"Modelo GRU construido con {model.count_params():,} parámetros")
        
        return model
    
    def _fit_model(self, X: pd.DataFrame, y: pd.Series,
                   validation_data: Optional[Tuple[pd.DataFrame, pd.Series]] = None,
                   **kwargs) -> Dict[str, Any]:
        """Entrenar modelo GRU (similar a LSTM)"""
        
        # Preparar datos
        X_seq, y_seq = self._prepare_sequential_data(X, y)
        
        if len(X_seq) == 0:
            raise ValueError("Datos insuficientes para secuencias")
        
        # Construir modelo
        input_shape = (X_seq.shape[1], X_seq.shape[2])
        self._model = self._build_model(input_shape=input_shape, **kwargs)
        
        # Validación
        validation_data_processed = None
        if validation_data:
            X_val, y_val = validation_data
            X_val_seq, y_val_seq = self._prepare_sequential_data(X_val, y_val)
            if len(X_val_seq) > 0:
                validation_data_processed = (X_val_seq, y_val_seq)
        
        # Callbacks
        self._setup_callbacks(validation_data_processed is not None)
        
        # Entrenar
        start_time = datetime.now()
        
        self.history = self._model.fit(
            X_seq, y_seq,
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_data=validation_data_processed,
            callbacks=self.callbacks,
            verbose=1
        )
        
        training_time = (datetime.now() - start_time).total_seconds()
        
        # Métricas
        metrics = {
            'training_time': training_time,
            'epochs_trained': len(self.history.history['loss']),
            'final_train_loss': self.history.history['loss'][-1]
        }
        
        if validation_data_processed:
            metrics['final_val_loss'] = self.history.history['val_loss'][-1]
        
        return metrics


class CNNModel(KerasModelBase):
    """Modelo CNN para detectar patrones en datos financieros"""
    
    def __init__(self, name: str, model_type: ModelType, version: str = "1.0.0",
                 conv_layers: List[Dict[str, int]] = None, pool_size: int = 2,
                 **kwargs):
        
        super().__init__(name, model_type, version, "CNN Model for Pattern Recognition")
        
        # Arquitectura convolucional
        self.conv_layers = conv_layers or [
            {'filters': 32, 'kernel_size': 3},
            {'filters': 64, 'kernel_size': 3},
            {'filters': 128, 'kernel_size': 3}
        ]
        self.pool_size = pool_size
        
        self.info.hyperparameters.update({
            'conv_layers': self.conv_layers,
            'pool_size': pool_size,
            **kwargs
        })
    
    def _build_model(self, input_shape: Tuple[int, int] = None, **kwargs) -> Model:
        """Construir arquitectura CNN"""
        
        if input_shape is None:
            input_shape = (self.sequence_length, 10)
        
        model = Sequential(name=f"CNN_{self.info.name}")
        
        # Input
        model.add(layers.Input(shape=input_shape))
        
        # Capas convolucionales
        for i, layer_config in enumerate(self.conv_layers):
            model.add(layers.Conv1D(
                filters=layer_config['filters'],
                kernel_size=layer_config['kernel_size'],
                activation='relu',
                padding='same',
                name=f"conv1d_{i+1}"
            ))
            
            model.add(layers.MaxPooling1D(pool_size=self.pool_size))
            model.add(layers.Dropout(0.25))
        
        # Aplanar
        model.add(layers.GlobalMaxPooling1D())
        
        # Capas densas
        model.add(layers.Dense(128, activation='relu'))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dropout(0.3))
        
        # Salida
        if self.info.model_type == ModelType.CLASSIFIER:
            output_units = 1
            activation = 'sigmoid'
            loss = 'binary_crossentropy'
            metrics = ['accuracy']
        else:
            output_units = 1
            activation = 'linear'
            loss = 'mse'
            metrics = ['mae']
        
        model.add(layers.Dense(output_units, activation=activation))
        
        # Compilar
        optimizer = Adam(learning_rate=self.learning_rate)
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        
        logger.info(f"Modelo CNN construido con {model.count_params():,} parámetros")
        
        return model
    
    def _fit_model(self, X: pd.DataFrame, y: pd.Series,
                   validation_data: Optional[Tuple[pd.DataFrame, pd.Series]] = None,
                   **kwargs) -> Dict[str, Any]:
        """Entrenar modelo CNN"""
        
        # Preparar datos secuenciales para CNN
        X_seq, y_seq = self._prepare_sequential_data(X, y)
        
        if len(X_seq) == 0:
            raise ValueError("Datos insuficientes")
        
        # Construir modelo
        input_shape = (X_seq.shape[1], X_seq.shape[2])
        self._model = self._build_model(input_shape=input_shape, **kwargs)
        
        # Validación
        validation_data_processed = None
        if validation_data:
            X_val, y_val = validation_data
            X_val_seq, y_val_seq = self._prepare_sequential_data(X_val, y_val)
            if len(X_val_seq) > 0:
                validation_data_processed = (X_val_seq, y_val_seq)
        
        # Callbacks
        self._setup_callbacks(validation_data_processed is not None)
        
        # Entrenar
        start_time = datetime.now()
        
        self.history = self._model.fit(
            X_seq, y_seq,
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_data=validation_data_processed,
            callbacks=self.callbacks,
            verbose=1
        )
        
        training_time = (datetime.now() - start_time).total_seconds()
        
        return {
            'training_time': training_time,
            'epochs_trained': len(self.history.history['loss']),
            'final_train_loss': self.history.history['loss'][-1]
        }


class CNNLSTMModel(KerasModelBase):
    """Modelo híbrido CNN-LSTM para capturar patrones y secuencias"""
    
    def __init__(self, name: str, model_type: ModelType, version: str = "1.0.0",
                 conv_filters: int = 64, lstm_units: int = 50, **kwargs):
        
        super().__init__(name, model_type, version, "CNN-LSTM Hybrid Model")
        
        self.conv_filters = conv_filters
        self.lstm_units = lstm_units
        
        self.info.hyperparameters.update({
            'conv_filters': conv_filters,
            'lstm_units': lstm_units,
            **kwargs
        })
    
    def _build_model(self, input_shape: Tuple[int, int] = None, **kwargs) -> Model:
        """Construir arquitectura híbrida CNN-LSTM"""
        
        if input_shape is None:
            input_shape = (self.sequence_length, 10)
        
        model = Sequential(name=f"CNN_LSTM_{self.info.name}")
        
        # Input
        model.add(layers.Input(shape=input_shape))
        
        # Bloque CNN para extracción de características
        model.add(layers.Conv1D(
            filters=self.conv_filters,
            kernel_size=3,
            activation='relu',
            padding='same'
        ))
        model.add(layers.MaxPooling1D(pool_size=2))
        model.add(layers.Dropout(0.25))
        
        model.add(layers.Conv1D(
            filters=self.conv_filters * 2,
            kernel_size=3,
            activation='relu',
            padding='same'
        ))
        model.add(layers.Dropout(0.25))
        
        # Bloque LSTM para modelado temporal
        model.add(layers.LSTM(
            units=self.lstm_units,
            return_sequences=False,
            dropout=0.2,
            recurrent_dropout=0.2
        ))
        
        # Capas densas finales
        model.add(layers.Dense(50, activation='relu'))
        model.add(layers.Dropout(0.3))
        
        # Salida
        if self.info.model_type == ModelType.CLASSIFIER:
            output_units = 1
            activation = 'sigmoid'
            loss = 'binary_crossentropy'
            metrics = ['accuracy']
        else:
            output_units = 1
            activation = 'linear'
            loss = 'mse'
            metrics = ['mae']
        
        model.add(layers.Dense(output_units, activation=activation))
        
        # Compilar
        optimizer = Adam(learning_rate=self.learning_rate)
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        
        logger.info(f"Modelo CNN-LSTM híbrido construido con {model.count_params():,} parámetros")
        
        return model
    
    def _fit_model(self, X: pd.DataFrame, y: pd.Series,
                   validation_data: Optional[Tuple[pd.DataFrame, pd.Series]] = None,
                   **kwargs) -> Dict[str, Any]:
        """Entrenar modelo híbrido"""
        
        # Preparar datos
        X_seq, y_seq = self._prepare_sequential_data(X, y)
        
        if len(X_seq) == 0:
            raise ValueError("Datos insuficientes")
        
        # Construir modelo
        input_shape = (X_seq.shape[1], X_seq.shape[2])
        self._model = self._build_model(input_shape=input_shape, **kwargs)
        
        # Validación
        validation_data_processed = None
        if validation_data:
            X_val, y_val = validation_data
            X_val_seq, y_val_seq = self._prepare_sequential_data(X_val, y_val)
            if len(X_val_seq) > 0:
                validation_data_processed = (X_val_seq, y_val_seq)
        
        # Callbacks
        self._setup_callbacks(validation_data_processed is not None)
        
        # Entrenar
        start_time = datetime.now()
        
        self.history = self._model.fit(
            X_seq, y_seq,
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_data=validation_data_processed,
            callbacks=self.callbacks,
            verbose=1
        )
        
        training_time = (datetime.now() - start_time).total_seconds()
        
        return {
            'training_time': training_time,
            'epochs_trained': len(self.history.history['loss']),
            'final_train_loss': self.history.history['loss'][-1]
        }


class DeepModelFactory:
    """Factory para crear modelos de Deep Learning"""
    
    _models = {
        'lstm': LSTMModel,
        'gru': GRUModel,
        'cnn': CNNModel,
        'cnn_lstm': CNNLSTMModel
    }
    
    @classmethod
    def create_model(cls, model_type: str, name: str, task_type: str,
                    version: str = "1.0.0", **model_params) -> BaseModel:
        """
        Crear modelo de Deep Learning
        
        Args:
            model_type: Tipo de modelo ('lstm', 'gru', 'cnn', 'cnn_lstm')
            name: Nombre del modelo
            task_type: 'classification' o 'regression'
            version: Versión
            **model_params: Parámetros específicos
            
        Returns:
            Modelo de Deep Learning
        """
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow/Keras requerido para modelos de Deep Learning")
        
        if model_type not in cls._models:
            available = list(cls._models.keys())
            raise ValueError(f"Modelo '{model_type}' no disponible. Disponibles: {available}")
        
        # Determinar ModelType
        if task_type.lower() in ['classification', 'classifier']:
            ml_model_type = ModelType.CLASSIFIER
        elif task_type.lower() in ['regression', 'regressor']:
            ml_model_type = ModelType.REGRESSOR
        else:
            raise ValueError(f"task_type debe ser 'classification' o 'regression'")
        
        # Crear modelo
        model_class = cls._models[model_type]
        model = model_class(name, ml_model_type, version, **model_params)
        
        logger.info(f"✅ Modelo Deep Learning creado: {name} ({model_type})")
        
        return model
    
    @classmethod
    def get_available_models(cls) -> List[str]:
        """Obtener modelos disponibles"""
        if TENSORFLOW_AVAILABLE:
            return list(cls._models.keys())
        else:
            return []
    
    @classmethod
    def create_ensemble_architectures(cls, base_config: Dict[str, Any],
                                    ensemble_name: str, task_type: str) -> List[BaseModel]:
        """Crear múltiples arquitecturas para ensemble"""
        
        architectures = [
            # LSTM variations
            {
                'model_type': 'lstm',
                'lstm_units': [50, 30],
                'dropout': 0.2
            },
            {
                'model_type': 'lstm',
                'lstm_units': [100, 50, 25],
                'dropout': 0.3
            },
            
            # GRU variations
            {
                'model_type': 'gru',
                'gru_units': [64, 32],
                'dropout': 0.25
            },
            
            # CNN
            {
                'model_type': 'cnn',
                'conv_layers': [
                    {'filters': 32, 'kernel_size': 3},
                    {'filters': 64, 'kernel_size': 5}
                ]
            },
            
            # Hybrid
            {
                'model_type': 'cnn_lstm',
                'conv_filters': 64,
                'lstm_units': 50
            }
        ]
        
        models = []
        
        for i, arch_config in enumerate(architectures):
            model_type = arch_config.pop('model_type')
            model_name = f"{ensemble_name}_{model_type}_{i}"
            
            # Combinar configuración base con específica
            final_config = {**base_config, **arch_config}
            
            try:
                model = cls.create_model(
                    model_type=model_type,
                    name=model_name,
                    task_type=task_type,
                    **final_config
                )
                models.append(model)
                
            except Exception as e:
                logger.error(f"Error creando modelo {model_name}: {e}")
        
        logger.info(f"Ensemble de Deep Learning creado: {len(models)} modelos")
        
        return models


# Funciones de utilidad
def create_default_deep_learning_suite(task_type: str = 'classification') -> List[BaseModel]:
    """Crear suite de modelos de Deep Learning con configuración por defecto"""
    
    if not TENSORFLOW_AVAILABLE:
        logger.warning("TensorFlow no disponible - no se pueden crear modelos de Deep Learning")
        return []
    
    models = []
    
    # Configuraciones por defecto
    configs = [
        {
            'model_type': 'lstm',
            'name': 'trading_lstm',
            'lstm_units': [50, 30],
            'sequence_length': 60
        },
        {
            'model_type': 'gru',
            'name': 'trading_gru',
            'gru_units': [50, 30],
            'sequence_length': 60
        },
        {
            'model_type': 'cnn',
            'name': 'trading_cnn',
            'conv_layers': [
                {'filters': 32, 'kernel_size': 3},
                {'filters': 64, 'kernel_size': 3}
            ],
            'sequence_length': 60
        },
        {
            'model_type': 'cnn_lstm',
            'name': 'trading_cnn_lstm',
            'conv_filters': 64,
            'lstm_units': 50,
            'sequence_length': 60
        }
    ]
    
    for config in configs:
        model_type = config.pop('model_type')
        
        try:
            model = DeepModelFactory.create_model(
                model_type=model_type,
                task_type=task_type,
                **config
            )
            models.append(model)
            
        except Exception as e:
            logger.warning(f"No se pudo crear modelo {model_type}: {e}")
    
    logger.info(f"Suite de Deep Learning creada: {len(models)} modelos")
    
    return models


def optimize_deep_learning_architecture(model: KerasModelBase, X: pd.DataFrame, y: pd.Series,
                                       param_grid: Dict[str, List] = None) -> Dict[str, Any]:
    """
    Optimizar arquitectura de modelo de Deep Learning
    
    Args:
        model: Modelo base a optimizar
        X: Features de entrenamiento
        y: Target
        param_grid: Grid de hiperparámetros
        
    Returns:
        Mejores parámetros encontrados
    """
    if param_grid is None:
        # Grid por defecto para modelos LSTM/GRU
        param_grid = {
            'learning_rate': [0.001, 0.01, 0.1],
            'batch_size': [16, 32, 64],
            'dropout': [0.1, 0.2, 0.3]
        }
    
    best_score = -float('inf')
    best_params = {}
    
    logger.info(f"Optimizando arquitectura para {model.info.name}")
    
    # Búsqueda manual (para Deep Learning es más práctico que GridSearch)
    for lr in param_grid.get('learning_rate', [0.001]):
        for batch_size in param_grid.get('batch_size', [32]):
            for dropout in param_grid.get('dropout', [0.2]):
                
                try:
                    # Crear copia del modelo con nuevos parámetros
                    test_model = model.__class__(
                        name=f"{model.info.name}_test",
                        model_type=model.info.model_type,
                        learning_rate=lr,
                        batch_size=batch_size,
                        dropout=dropout,
                        epochs=10  # Pocas épocas para búsqueda rápida
                    )
                    
                    # Dividir datos para validación
                    split_idx = int(len(X) * 0.8)
                    X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
                    y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]
                    
                    # Entrenar
                    test_model.fit(X_train, y_train, validation_data=(X_val, y_val))
                    
                    # Evaluar
                    predictions = test_model.predict(X_val)
                    
                    if model.info.model_type == ModelType.CLASSIFIER:
                        from sklearn.metrics import accuracy_score
                        score = accuracy_score(y_val.values[-len(predictions):], predictions)
                    else:
                        from sklearn.metrics import r2_score
                        score = r2_score(y_val.values[-len(predictions):], predictions)
                    
                    logger.info(f"Config: lr={lr}, batch={batch_size}, dropout={dropout} -> Score: {score:.4f}")
                    
                    if score > best_score:
                        best_score = score
                        best_params = {
                            'learning_rate': lr,
                            'batch_size': batch_size,
                            'dropout': dropout
                        }
                
                except Exception as e:
                    logger.warning(f"Error en configuración lr={lr}, batch={batch_size}: {e}")
                    continue
    
    logger.info(f"Mejores parámetros encontrados: {best_params} (score: {best_score:.4f})")
    
    return best_params
'''
5. data/feature_engineer.py
Ruta: TradingBot_Cuantitative_MT5/data/feature_engineer.py
Resumen:

Sistema modular de ingeniería de características
Arquitectura extensible con plugins para nuevas features
Pipeline configurable de transformaciones
Soporte para features personalizadas y experimentales
Registro automático de nuevos tipos de features
'''
# data/feature_engineer.py
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Callable, Any, Union, Type
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import inspect
import importlib
import json
from pathlib import Path
from sklearn.preprocessing import PolynomialFeatures
from sklearn.decomposition import PCA
import talib

from utils.log_config import get_logger
from data.data_processor import ProcessingConfig

logger = get_logger('data')


@dataclass
class FeatureConfig:
    """Configuración para ingeniería de features"""
    # Features básicas
    calculate_price_features: bool = True
    calculate_volume_features: bool = True
    calculate_time_features: bool = True
    
    # Features avanzadas
    calculate_microstructure: bool = True
    calculate_market_profile: bool = True
    calculate_order_flow: bool = False  # Requiere datos de nivel 2
    
    # Interacciones y transformaciones
    create_polynomial_features: bool = False
    polynomial_degree: int = 2
    create_interaction_features: bool = True
    
    # Reducción de dimensionalidad
    apply_pca: bool = False
    pca_components: int = 50
    
    # Features personalizadas
    custom_features_path: Optional[Path] = None
    enable_experimental: bool = False
    
    # Ventanas temporales
    lookback_windows: List[int] = field(default_factory=lambda: [5, 10, 20, 50, 100])
    
    # Configuración de plugins
    plugins_enabled: bool = True
    plugins_directory: Optional[Path] = None


class FeatureEngineering(ABC):
    """Clase base abstracta para ingeniería de features"""
    
    @abstractmethod
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calcular features"""
        pass
    
    @abstractmethod
    def get_feature_names(self) -> List[str]:
        """Obtener nombres de features generadas"""
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Nombre del conjunto de features"""
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        """Descripción del conjunto de features"""
        pass


class PriceFeatures(FeatureEngineering):
    """Features basadas en precio"""
    
    def __init__(self, windows: List[int] = None):
        self.windows = windows or [5, 10, 20, 50]
        self.features_generated = []
    
    @property
    def name(self) -> str:
        return "Price Features"
    
    @property
    def description(self) -> str:
        return "Features derivadas de movimientos de precio y estructura"
    
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calcular features de precio"""
        logger.info(f"Calculando {self.name}...")
        
        # Precio logarítmico
        df['log_price'] = np.log(df['close'])
        
        # Cambios de precio
        for window in self.windows:
            # Momentum
            df[f'momentum_{window}'] = df['close'] - df['close'].shift(window)
            df[f'momentum_pct_{window}'] = df['close'].pct_change(window)
            
            # Rate of change
            df[f'roc_{window}'] = (df['close'] - df['close'].shift(window)) / df['close'].shift(window)
            
            # Precio relativo a rango
            df[f'price_position_{window}'] = (df['close'] - df['low'].rolling(window).min()) / \
                                            (df['high'].rolling(window).max() - df['low'].rolling(window).min() + 1e-10)
            
            self.features_generated.extend([
                f'momentum_{window}', f'momentum_pct_{window}',
                f'roc_{window}', f'price_position_{window}'
            ])
        
        # Eficiencia de precio (cuán directo es el movimiento)
        for window in [10, 20]:
            net_change = df['close'] - df['close'].shift(window)
            total_change = df['high'].rolling(window).max() - df['low'].rolling(window).min()
            df[f'price_efficiency_{window}'] = net_change / (total_change + 1e-10)
            self.features_generated.append(f'price_efficiency_{window}')
        
        # Divergencia de precio
        df['hl_spread'] = (df['high'] - df['low']) / df['close']
        df['co_spread'] = (df['close'] - df['open']) / df['close']
        self.features_generated.extend(['hl_spread', 'co_spread'])
        
        # Sesgo de precio
        df['price_skew'] = (df['close'] - df['open']) / (df['high'] - df['low'] + 1e-10)
        self.features_generated.append('price_skew')
        
        logger.info(f"  {len(self.features_generated)} features de precio calculadas")
        
        return df
    
    def get_feature_names(self) -> List[str]:
        return self.features_generated


class VolumeFeatures(FeatureEngineering):
    """Features basadas en volumen"""
    
    def __init__(self, windows: List[int] = None):
        self.windows = windows or [5, 10, 20, 50]
        self.features_generated = []
    
    @property
    def name(self) -> str:
        return "Volume Features"
    
    @property
    def description(self) -> str:
        return "Features derivadas de patrones de volumen y liquidez"
    
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calcular features de volumen"""
        logger.info(f"Calculando {self.name}...")
        
        # Volume weighted average price (VWAP)
        df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
        
        for window in self.windows:
            # VWAP
            df[f'vwap_{window}'] = (df['typical_price'] * df['tick_volume']).rolling(window).sum() / \
                                   df['tick_volume'].rolling(window).sum()
            
            # Volume rate of change
            df[f'volume_roc_{window}'] = df['tick_volume'].pct_change(window)
            
            # Volume ratio
            df[f'volume_ratio_{window}'] = df['tick_volume'] / df['tick_volume'].rolling(window).mean()
            
            # Price-Volume correlation
            df[f'pv_corr_{window}'] = df['close'].rolling(window).corr(df['tick_volume'])
            
            self.features_generated.extend([
                f'vwap_{window}', f'volume_roc_{window}',
                f'volume_ratio_{window}', f'pv_corr_{window}'
            ])
        
        # Money Flow Index components
        df['money_flow'] = df['typical_price'] * df['tick_volume']
        df['money_flow_positive'] = np.where(df['typical_price'] > df['typical_price'].shift(1), 
                                            df['money_flow'], 0)
        df['money_flow_negative'] = np.where(df['typical_price'] < df['typical_price'].shift(1), 
                                            df['money_flow'], 0)
        
        # Accumulation/Distribution
        df['ad'] = ((df['close'] - df['low']) - (df['high'] - df['close'])) / \
                   (df['high'] - df['low'] + 1e-10) * df['tick_volume']
        df['ad_line'] = df['ad'].cumsum()
        
        self.features_generated.extend(['typical_price', 'money_flow', 'ad', 'ad_line'])
        
        logger.info(f"  {len(self.features_generated)} features de volumen calculadas")
        
        return df
    
    def get_feature_names(self) -> List[str]:
        return self.features_generated


class TimeFeatures(FeatureEngineering):
    """Features basadas en tiempo y estacionalidad"""
    
    def __init__(self):
        self.features_generated = []
    
    @property
    def name(self) -> str:
        return "Time Features"
    
    @property
    def description(self) -> str:
        return "Features temporales y de estacionalidad del mercado"
    
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calcular features temporales"""
        logger.info(f"Calculando {self.name}...")
        
        # Extraer componentes temporales
        df['hour'] = df.index.hour
        df['day_of_week'] = df.index.dayofweek
        df['day_of_month'] = df.index.day
        df['month'] = df.index.month
        df['quarter'] = df.index.quarter
        
        # Codificación cíclica para hora
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        
        # Codificación cíclica para día de la semana
        df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        # Codificación cíclica para día del mes
        df['dom_sin'] = np.sin(2 * np.pi * df['day_of_month'] / 31)
        df['dom_cos'] = np.cos(2 * np.pi * df['day_of_month'] / 31)
        
        # Sesiones de trading
        df['asian_session'] = ((df['hour'] >= 0) & (df['hour'] < 8)).astype(int)
        df['london_session'] = ((df['hour'] >= 8) & (df['hour'] < 16)).astype(int)
        df['ny_session'] = ((df['hour'] >= 13) & (df['hour'] < 22)).astype(int)
        df['session_overlap'] = ((df['hour'] >= 13) & (df['hour'] < 16)).astype(int)
        
        # Tiempo desde apertura de sesión
        df['minutes_since_open'] = df.index.hour * 60 + df.index.minute
        df['minutes_to_close'] = (24 * 60) - df['minutes_since_open']
        
        # Indicadores de período especial
        df['is_monday'] = (df['day_of_week'] == 0).astype(int)
        df['is_friday'] = (df['day_of_week'] == 4).astype(int)
        df['is_month_start'] = (df['day_of_month'] <= 5).astype(int)
        df['is_month_end'] = (df['day_of_month'] >= 25).astype(int)
        
        self.features_generated = [
            'hour', 'day_of_week', 'day_of_month', 'month', 'quarter',
            'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'dom_sin', 'dom_cos',
            'asian_session', 'london_session', 'ny_session', 'session_overlap',
            'minutes_since_open', 'minutes_to_close',
            'is_monday', 'is_friday', 'is_month_start', 'is_month_end'
        ]
        
        logger.info(f"  {len(self.features_generated)} features temporales calculadas")
        
        return df
    
    def get_feature_names(self) -> List[str]:
        return self.features_generated


class MicrostructureFeatures(FeatureEngineering):
    """Features de microestructura del mercado"""
    
    def __init__(self, windows: List[int] = None):
        self.windows = windows or [5, 10, 20]
        self.features_generated = []
    
    @property
    def name(self) -> str:
        return "Microstructure Features"
    
    @property
    def description(self) -> str:
        return "Features de microestructura y dinámicas de mercado"
    
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calcular features de microestructura"""
        logger.info(f"Calculando {self.name}...")
        
        # Realized volatility
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        
        for window in self.windows:
            # Volatilidad realizada
            df[f'realized_vol_{window}'] = df['log_returns'].rolling(window).std() * np.sqrt(252 * 24 * 4)  # Anualizada
            
            # Parkinson volatility (usando high-low)
            df[f'parkinson_vol_{window}'] = np.sqrt(
                (1 / (4 * np.log(2))) * 
                (np.log(df['high'] / df['low']) ** 2).rolling(window).mean()
            ) * np.sqrt(252 * 24 * 4)
            
            # Garman-Klass volatility
            df[f'gk_vol_{window}'] = np.sqrt(
                0.5 * (np.log(df['high'] / df['low']) ** 2).rolling(window).mean() -
                (2 * np.log(2) - 1) * (np.log(df['close'] / df['open']) ** 2).rolling(window).mean()
            ) * np.sqrt(252 * 24 * 4)
            
            # Amihud illiquidity
            df[f'amihud_{window}'] = (df['log_returns'].abs() / (df['tick_volume'] + 1)).rolling(window).mean()
            
            # Kyle's lambda (impacto de precio)
            returns_abs = df['log_returns'].abs()
            volume_signed = df['tick_volume'] * np.sign(df['log_returns'])
            df[f'kyle_lambda_{window}'] = returns_abs.rolling(window).sum() / \
                                          (volume_signed.rolling(window).sum().abs() + 1e-10)
            
            self.features_generated.extend([
                f'realized_vol_{window}', f'parkinson_vol_{window}', 
                f'gk_vol_{window}', f'amihud_{window}', f'kyle_lambda_{window}'
            ])
        
        # Autocorrelación de retornos (detección de ineficiencias)
        for lag in [1, 2, 5]:
            df[f'returns_autocorr_lag{lag}'] = df['log_returns'].rolling(20).apply(
                lambda x: x.autocorr(lag=lag) if len(x) > lag else np.nan
            )
            self.features_generated.append(f'returns_autocorr_lag{lag}')
        
        # Runs test (aleatoriedad de retornos)
        df['returns_sign'] = np.sign(df['log_returns'])
        df['runs'] = (df['returns_sign'] != df['returns_sign'].shift(1)).astype(int)
        df['runs_ratio'] = df['runs'].rolling(20).sum() / 20
        
        self.features_generated.extend(['log_returns', 'returns_sign', 'runs', 'runs_ratio'])
        
        logger.info(f"  {len(self.features_generated)} features de microestructura calculadas")
        
        return df
    
    def get_feature_names(self) -> List[str]:
        return self.features_generated


class MarketProfileFeatures(FeatureEngineering):
    """Features de perfil de mercado y estructura"""
    
    def __init__(self, profile_periods: List[int] = None):
        self.profile_periods = profile_periods or [20, 50, 100]
        self.features_generated = []
    
    @property
    def name(self) -> str:
        return "Market Profile Features"
    
    @property
    def description(self) -> str:
        return "Features basadas en perfiles de mercado y distribución de precios"
    
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calcular features de perfil de mercado"""
        logger.info(f"Calculando {self.name}...")
        
        for period in self.profile_periods:
            # Value Area (70% del volumen)
            rolling_window = df['close'].rolling(period)
            
            # Calcular POC (Point of Control) - precio con más volumen
            df[f'poc_{period}'] = df['close'].rolling(period).apply(
                lambda x: x.mode()[0] if len(x) > 0 and len(x.mode()) > 0 else x.mean()
            )
            
            # Value Area High/Low (aproximación simplificada)
            df[f'vah_{period}'] = df['high'].rolling(period).quantile(0.85)
            df[f'val_{period}'] = df['low'].rolling(period).quantile(0.15)
            
            # Precio relativo a Value Area
            df[f'price_to_poc_{period}'] = (df['close'] - df[f'poc_{period}']) / df['close']
            df[f'price_in_va_{period}'] = (
                (df['close'] >= df[f'val_{period}']) & 
                (df['close'] <= df[f'vah_{period}'])
            ).astype(int)
            
            # Skew del perfil
            df[f'profile_skew_{period}'] = df['close'].rolling(period).skew()
            
            # Balance del mercado
            df[f'market_balance_{period}'] = (
                df['close'].rolling(period).std() / 
                df['close'].rolling(period).mean()
            )
            
            self.features_generated.extend([
                f'poc_{period}', f'vah_{period}', f'val_{period}',
                f'price_to_poc_{period}', f'price_in_va_{period}',
                f'profile_skew_{period}', f'market_balance_{period}'
            ])
        
        # Distribución de volumen por niveles de precio
        for period in [20, 50]:
            # Volumen en máximos vs mínimos
            high_volume = df.apply(
                lambda row: row['tick_volume'] if row['close'] > df['close'].shift(1).iloc[row.name-1] 
                else 0 if row.name > 0 else 0, axis=1
            )
            low_volume = df.apply(
                lambda row: row['tick_volume'] if row['close'] < df['close'].shift(1).iloc[row.name-1] 
                else 0 if row.name > 0 else 0, axis=1
            )
            
            df[f'volume_imbalance_{period}'] = (
                high_volume.rolling(period).sum() - low_volume.rolling(period).sum()
            ) / (high_volume.rolling(period).sum() + low_volume.rolling(period).sum() + 1e-10)
            
            self.features_generated.append(f'volume_imbalance_{period}')
        
        logger.info(f"  {len(self.features_generated)} features de perfil de mercado calculadas")
        
        return df
    
    def get_feature_names(self) -> List[str]:
        return self.features_generated


class CustomFeaturePlugin(FeatureEngineering):
    """Clase base para plugins de features personalizadas"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.features_generated = []
    
    @property
    @abstractmethod
    def name(self) -> str:
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        pass
    
    @abstractmethod
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        pass
    
    def get_feature_names(self) -> List[str]:
        return self.features_generated


class FeatureEngineer:
    """Orquestador principal de ingeniería de features"""
    
    def __init__(self, config: FeatureConfig = None):
        self.config = config or FeatureConfig()
        self.feature_sets: List[FeatureEngineering] = []
        self.all_features: List[str] = []
        self.processing_history = []
        
        # Registrar feature sets estándar
        self._register_standard_features()
        
        # Cargar plugins si están habilitados
        if self.config.plugins_enabled:
            self._load_plugins()
        
        logger.info("FeatureEngineer inicializado")
        logger.info(f"Feature sets registrados: {len(self.feature_sets)}")
    
    def _register_standard_features(self):
        """Registrar conjuntos de features estándar"""
        if self.config.calculate_price_features:
            self.register_feature_set(PriceFeatures(self.config.lookback_windows))
        
        if self.config.calculate_volume_features:
            self.register_feature_set(VolumeFeatures(self.config.lookback_windows))
        
        if self.config.calculate_time_features:
            self.register_feature_set(TimeFeatures())
        
        if self.config.calculate_microstructure:
            self.register_feature_set(MicrostructureFeatures(self.config.lookback_windows[:3]))
        
        if self.config.calculate_market_profile:
            self.register_feature_set(MarketProfileFeatures())
    
    def register_feature_set(self, feature_set: FeatureEngineering):
        """Registrar un nuevo conjunto de features"""
        if not isinstance(feature_set, FeatureEngineering):
            raise TypeError("Feature set debe heredar de FeatureEngineering")
        
        self.feature_sets.append(feature_set)
        logger.info(f"Feature set registrado: {feature_set.name}")
    
    def _load_plugins(self):
        """Cargar plugins de features desde directorio"""
        plugins_dir = self.config.plugins_directory or Path("plugins/features")
        
        if not plugins_dir.exists():
            logger.warning(f"Directorio de plugins no encontrado: {plugins_dir}")
            return
        
        logger.info(f"Cargando plugins desde: {plugins_dir}")
        
        for plugin_file in plugins_dir.glob("*.py"):
            if plugin_file.name.startswith("_"):
                continue
            
            try:
                # Importar módulo
                spec = importlib.util.spec_from_file_location(
                    plugin_file.stem, plugin_file
                )
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                
                # Buscar clases que hereden de CustomFeaturePlugin
                for name, obj in inspect.getmembers(module, inspect.isclass):
                    if issubclass(obj, CustomFeaturePlugin) and obj != CustomFeaturePlugin:
                        plugin_instance = obj()
                        self.register_feature_set(plugin_instance)
                        logger.info(f"Plugin cargado: {plugin_instance.name}")
                        
            except Exception as e:
                logger.error(f"Error cargando plugin {plugin_file}: {e}")
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aplicar toda la ingeniería de features"""
        logger.info("="*60)
        logger.info("INGENIERÍA DE FEATURES")
        logger.info("="*60)
        logger.info(f"Datos de entrada: {len(df)} filas")
        logger.info(f"Feature sets a aplicar: {len(self.feature_sets)}")
        
        # Registro de proceso
        process_record = {
            'timestamp': pd.Timestamp.now(),
            'input_shape': df.shape,
            'feature_sets_applied': [],
            'features_created': [],
            'processing_time': {}
        }
        
        # Copia para no modificar original
        df_features = df.copy()
        initial_columns = set(df_features.columns)
        
        # Aplicar cada conjunto de features
        for feature_set in self.feature_sets:
            start_time = pd.Timestamp.now()
            
            try:
                logger.info(f"\nAplicando: {feature_set.name}")
                logger.info(f"  {feature_set.description}")
                
                # Calcular features
                df_features = feature_set.calculate(df_features)
                
                # Registrar features creadas
                new_features = feature_set.get_feature_names()
                self.all_features.extend(new_features)
                
                process_record['feature_sets_applied'].append(feature_set.name)
                process_record['features_created'].extend(new_features)
                
                # Tiempo de procesamiento
                elapsed = (pd.Timestamp.now() - start_time).total_seconds()
                process_record['processing_time'][feature_set.name] = elapsed
                
                logger.info(f"  Completado en {elapsed:.2f}s")
                
            except Exception as e:
                logger.error(f"Error en {feature_set.name}: {e}")
                continue
        
        # Aplicar transformaciones adicionales si están configuradas
        if self.config.create_interaction_features:
            df_features = self._create_interaction_features(df_features, initial_columns)
        
        if self.config.create_polynomial_features:
            df_features = self._create_polynomial_features(df_features, initial_columns)
        
        if self.config.apply_pca:
            df_features = self._apply_pca(df_features, initial_columns)
        
        # Registro final
        process_record['output_shape'] = df_features.shape
        process_record['total_features'] = len(df_features.columns) - len(initial_columns)
        self.processing_history.append(process_record)
        
        # Resumen
        logger.info("\n" + "="*60)
        logger.info("RESUMEN DE INGENIERÍA DE FEATURES")
        logger.info("="*60)
        logger.info(f"Features originales: {len(initial_columns)}")
        logger.info(f"Features creadas: {process_record['total_features']}")
        logger.info(f"Total features: {len(df_features.columns)}")
        logger.info(f"Tiempo total: {sum(process_record['processing_time'].values()):.2f}s")
        
        return df_features
    
    def _create_interaction_features(self, df: pd.DataFrame, 
                                   exclude_columns: set) -> pd.DataFrame:
        """Crear features de interacción"""
        logger.info("\nCreando features de interacción...")
        
        # Seleccionar columnas numéricas importantes
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        feature_cols = [col for col in numeric_cols if col not in exclude_columns]
        
        # Limitar número de features para evitar explosión combinatoria
        important_features = self._select_important_features(df[feature_cols], max_features=10)
        
        interactions_created = 0
        
        # Crear interacciones
        for i in range(len(important_features)):
            for j in range(i+1, len(important_features)):
                feat1, feat2 = important_features[i], important_features[j]
                
                # Multiplicación
                interaction_name = f"{feat1}_x_{feat2}"
                df[interaction_name] = df[feat1] * df[feat2]
                
                # Ratio (con protección contra división por cero)
                ratio_name = f"{feat1}_div_{feat2}"
                df[ratio_name] = df[feat1] / (df[feat2] + 1e-10)
                
                interactions_created += 2
        
        logger.info(f"  {interactions_created} interacciones creadas")
        
        return df
    
    def _create_polynomial_features(self, df: pd.DataFrame, 
                                  exclude_columns: set) -> pd.DataFrame:
        """Crear features polinómicas"""
        logger.info(f"\nCreando features polinómicas (grado {self.config.polynomial_degree})...")
        
        # Seleccionar columnas
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        feature_cols = [col for col in numeric_cols if col not in exclude_columns]
        
        # Limitar features
        important_features = self._select_important_features(df[feature_cols], max_features=5)
        
        # Crear polinomios
        poly = PolynomialFeatures(degree=self.config.polynomial_degree, include_bias=False)
        poly_features = poly.fit_transform(df[important_features])
        
        # Obtener nombres
        feature_names = poly.get_feature_names_out(important_features)
        
        # Agregar al DataFrame
        for i, name in enumerate(feature_names):
            if name not in df.columns:  # Evitar duplicados
                df[f"poly_{name}"] = poly_features[:, i]
        
        logger.info(f"  {len(feature_names)} features polinómicas creadas")
        
        return df
    
    def _apply_pca(self, df: pd.DataFrame, exclude_columns: set) -> pd.DataFrame:
        """Aplicar PCA para reducción de dimensionalidad"""
        logger.info(f"\nAplicando PCA (componentes: {self.config.pca_components})...")
        
        # Seleccionar columnas
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        feature_cols = [col for col in numeric_cols if col not in exclude_columns]
        
        # Aplicar PCA
        pca = PCA(n_components=min(self.config.pca_components, len(feature_cols)))
        pca_features = pca.fit_transform(df[feature_cols].fillna(0))
        
        # Agregar componentes al DataFrame
        for i in range(pca_features.shape[1]):
            df[f'pca_{i}'] = pca_features[:, i]
        
        explained_variance = pca.explained_variance_ratio_.sum()
        logger.info(f"  Varianza explicada: {explained_variance:.2%}")
        
        return df
    
    def _select_important_features(self, df: pd.DataFrame, 
                                 max_features: int = 10) -> List[str]:
        """Seleccionar features más importantes basado en varianza"""
        # Usar varianza como proxy de importancia
        variances = df.var()
        
        # Excluir features con varianza cero o muy baja
        variances = variances[variances > 1e-10]
        
        # Seleccionar top features
        top_features = variances.nlargest(max_features).index.tolist()
        
        return top_features
    
    def get_feature_info(self) -> Dict[str, Any]:
        """Obtener información sobre features generadas"""
        info = {
            'total_features': len(self.all_features),
            'feature_sets': [fs.name for fs in self.feature_sets],
            'processing_history': self.processing_history,
            'config': self.config.__dict__
        }
        
        # Agrupar features por tipo
        feature_groups = {}
        for feature in self.all_features:
            # Intentar identificar el tipo de feature
            if any(x in feature for x in ['price', 'momentum', 'roc']):
                group = 'price'
            elif any(x in feature for x in ['volume', 'vwap', 'money_flow']):
                group = 'volume'
            elif any(x in feature for x in ['hour', 'day', 'session', 'time']):
                group = 'time'
            elif any(x in feature for x in ['vol_', 'volatility', 'gk_', 'parkinson']):
                group = 'volatility'
            elif any(x in feature for x in ['poc', 'vah', 'val', 'profile']):
                group = 'market_profile'
            else:
                group = 'other'
            
            if group not in feature_groups:
                feature_groups[group] = []
            feature_groups[group].append(feature)
        
        info['feature_groups'] = {k: len(v) for k, v in feature_groups.items()}
        
        return info
    
    def save_feature_config(self, filepath: Path):
        """Guardar configuración de features"""
        config_dict = {
            'config': self.config.__dict__,
            'feature_sets': [fs.name for fs in self.feature_sets],
            'all_features': self.all_features,
            'timestamp': pd.Timestamp.now().isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=2, default=str)
        
        logger.info(f"Configuración de features guardada en: {filepath}")
    
    @classmethod
    def load_feature_config(cls, filepath: Path) -> 'FeatureEngineer':
        """Cargar configuración de features"""
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        
        # Reconstruir configuración
        config = FeatureConfig(**config_dict['config'])
        engineer = cls(config)
        
        logger.info(f"Configuración de features cargada desde: {filepath}")
        
        return engineer


# Función de utilidad para crear pipeline completo
def create_feature_pipeline(config: Optional[FeatureConfig] = None) -> FeatureEngineer:
    """Crear pipeline de ingeniería de features"""
    return FeatureEngineer(config)


# Ejemplo de plugin personalizado
class ExampleCustomFeature(CustomFeaturePlugin):
    """Ejemplo de plugin de feature personalizada"""
    
    @property
    def name(self) -> str:
        return "Example Custom Feature"
    
    @property
    def description(self) -> str:
        return "Ejemplo de cómo crear features personalizadas"
    
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        # Ejemplo: Detectar divergencias
        df['price_rsi_divergence'] = 0
        
        if 'rsi_14' in df.columns:
            # Detectar divergencia alcista (precio baja, RSI sube)
            price_down = df['close'] < df['close'].shift(20)
            rsi_up = df['rsi_14'] > df['rsi_14'].shift(20)
            df.loc[price_down & rsi_up, 'price_rsi_divergence'] = 1
            
            # Detectar divergencia bajista (precio sube, RSI baja)
            price_up = df['close'] > df['close'].shift(20)
            rsi_down = df['rsi_14'] < df['rsi_14'].shift(20)
            df.loc[price_up & rsi_down, 'price_rsi_divergence'] = -1
            
            self.features_generated.append('price_rsi_divergence')
        
        return df
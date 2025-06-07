'''
4. data/data_processor.py
Ruta: TradingBot_Cuantitative_MT5/data/data_processor.py
Resumen:

Procesa datos crudos para análisis
Calcula indicadores técnicos
Detecta patrones en los datos
Normaliza y escala datos para ML
Maneja outliers y valores anómalos
'''
# data/data_processor.py
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import talib
from scipy import stats
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

from utils.log_config import get_logger

logger = get_logger('data')


@dataclass
class ProcessingConfig:
    """Configuración para procesamiento de datos"""
    # Indicadores técnicos
    use_technical_indicators: bool = True
    indicator_periods: List[int] = None
    
    # Normalización
    scaling_method: str = 'robust'  # 'standard', 'robust', 'minmax'
    
    # Outliers
    remove_outliers: bool = True
    outlier_threshold: float = 4.0  # Desviaciones estándar
    
    # Features estadísticas
    calculate_statistics: bool = True
    statistic_windows: List[int] = None
    
    # Patrones
    detect_patterns: bool = True
    pattern_lookback: int = 20
    
    def __post_init__(self):
        if self.indicator_periods is None:
            self.indicator_periods = [5, 10, 14, 20, 50, 100, 200]
        if self.statistic_windows is None:
            self.statistic_windows = [5, 10, 20, 50]


class DataProcessor:
    """Procesador principal de datos de mercado"""
    
    def __init__(self, config: Optional[ProcessingConfig] = None):
        self.config = config or ProcessingConfig()
        self.scaler = None
        self.feature_names = []
        self.processing_stats = {
            'total_processed': 0,
            'outliers_removed': 0,
            'indicators_calculated': 0,
            'patterns_detected': 0
        }
        
        logger.info("DataProcessor inicializado")
        logger.info(f"Método de escalado: {self.config.scaling_method}")
        logger.info(f"Remover outliers: {self.config.remove_outliers}")
    
    def process(self, df: pd.DataFrame, fit_scaler: bool = True) -> pd.DataFrame:
        """
        Procesar DataFrame completo
        
        Args:
            df: DataFrame con datos OHLCV
            fit_scaler: Si ajustar el scaler (True para entrenamiento)
            
        Returns:
            DataFrame procesado
        """
        logger.info("="*60)
        logger.info("PROCESAMIENTO DE DATOS")
        logger.info("="*60)
        logger.info(f"Datos de entrada: {len(df)} filas")
        logger.info(f"Rango: {df.index[0]} a {df.index[-1]}")
        
        # Copia para no modificar original
        df_processed = df.copy()
        
        # 1. Calcular retornos y features básicas
        logger.info("\n1. Calculando features básicas...")
        df_processed = self._calculate_basic_features(df_processed)
        
        # 2. Indicadores técnicos
        if self.config.use_technical_indicators:
            logger.info("\n2. Calculando indicadores técnicos...")
            df_processed = self._calculate_technical_indicators(df_processed)
        
        # 3. Features estadísticas
        if self.config.calculate_statistics:
            logger.info("\n3. Calculando features estadísticas...")
            df_processed = self._calculate_statistical_features(df_processed)
        
        # 4. Detectar patrones
        if self.config.detect_patterns:
            logger.info("\n4. Detectando patrones...")
            df_processed = self._detect_patterns(df_processed)
        
        # 5. Remover outliers
        if self.config.remove_outliers:
            logger.info("\n5. Removiendo outliers...")
            df_processed = self._remove_outliers(df_processed)
        
        # 6. Normalización
        logger.info("\n6. Normalizando datos...")
        df_processed = self._normalize_data(df_processed, fit_scaler)
        
        # 7. Limpiar NaN
        initial_rows = len(df_processed)
        df_processed = df_processed.dropna()
        dropped_rows = initial_rows - len(df_processed)
        
        if dropped_rows > 0:
            logger.warning(f"Filas eliminadas por NaN: {dropped_rows}")
        
        # Actualizar estadísticas
        self.processing_stats['total_processed'] += len(df_processed)
        
        # Resumen final
        logger.info("\n" + "="*60)
        logger.info("RESUMEN DEL PROCESAMIENTO")
        logger.info("="*60)
        logger.info(f"Filas procesadas: {len(df_processed)}")
        logger.info(f"Features totales: {len(df_processed.columns)}")
        logger.info(f"Features numéricas: {len(df_processed.select_dtypes(include=[np.number]).columns)}")
        logger.info(f"Memoria utilizada: {df_processed.memory_usage().sum() / 1024**2:.2f} MB")
        
        # Guardar nombres de features
        self.feature_names = df_processed.columns.tolist()
        
        return df_processed
    
    def _calculate_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calcular features básicas"""
        # Retornos
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        
        # Retornos de diferentes períodos
        for period in [5, 10, 20]:
            df[f'returns_{period}'] = df['close'].pct_change(period)
        
        # Precio relativo a máximos/mínimos
        for period in [20, 50]:
            df[f'price_to_high_{period}'] = df['close'] / df['high'].rolling(period).max()
            df[f'price_to_low_{period}'] = df['close'] / df['low'].rolling(period).min()
        
        # Rango (High - Low)
        df['range'] = df['high'] - df['low']
        df['range_pct'] = df['range'] / df['close']
        
        # Posición en el rango del día
        df['close_position'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-10)
        
        # Gaps
        df['gap'] = df['open'] - df['close'].shift(1)
        df['gap_pct'] = df['gap'] / df['close'].shift(1)
        
        # Volumen relativo
        df['volume_ratio'] = df['tick_volume'] / df['tick_volume'].rolling(20).mean()
        
        logger.info(f"  Features básicas calculadas: {len([c for c in df.columns if c not in ['open', 'high', 'low', 'close', 'tick_volume']])}")
        
        return df
    
    def _calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calcular indicadores técnicos usando TA-Lib"""
        indicators_calculated = 0
        
        # Preparar datos para TA-Lib
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        volume = df['tick_volume'].values.astype(float)
        
        # RSI
        for period in [9, 14, 21]:
            df[f'rsi_{period}'] = talib.RSI(close, timeperiod=period)
            indicators_calculated += 1
        
        # Moving Averages
        for period in self.config.indicator_periods:
            if period <= len(df):
                df[f'sma_{period}'] = talib.SMA(close, timeperiod=period)
                df[f'ema_{period}'] = talib.EMA(close, timeperiod=period)
                indicators_calculated += 2
        
        # MACD
        macd, macd_signal, macd_hist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
        df['macd'] = macd
        df['macd_signal'] = macd_signal
        df['macd_hist'] = macd_hist
        indicators_calculated += 3
        
        # Bollinger Bands
        for period in [20, 50]:
            upper, middle, lower = talib.BBANDS(close, timeperiod=period, nbdevup=2, nbdevdn=2)
            df[f'bb_upper_{period}'] = upper
            df[f'bb_middle_{period}'] = middle
            df[f'bb_lower_{period}'] = lower
            df[f'bb_width_{period}'] = upper - lower
            df[f'bb_position_{period}'] = (close - lower) / (upper - lower + 1e-10)
            indicators_calculated += 5
        
        # ATR
        for period in [14, 20]:
            df[f'atr_{period}'] = talib.ATR(high, low, close, timeperiod=period)
            df[f'atr_pct_{period}'] = df[f'atr_{period}'] / close
            indicators_calculated += 2
        
        # Stochastic
        slowk, slowd = talib.STOCH(high, low, close, fastk_period=14, slowk_period=3, slowd_period=3)
        df['stoch_k'] = slowk
        df['stoch_d'] = slowd
        indicators_calculated += 2
        
        # ADX
        df['adx'] = talib.ADX(high, low, close, timeperiod=14)
        df['plus_di'] = talib.PLUS_DI(high, low, close, timeperiod=14)
        df['minus_di'] = talib.MINUS_DI(high, low, close, timeperiod=14)
        indicators_calculated += 3
        
        # CCI
        df['cci'] = talib.CCI(high, low, close, timeperiod=14)
        indicators_calculated += 1
        
        # MFI
        df['mfi'] = talib.MFI(high, low, close, volume, timeperiod=14)
        indicators_calculated += 1
        
        # Williams %R
        df['williams_r'] = talib.WILLR(high, low, close, timeperiod=14)
        indicators_calculated += 1
        
        # OBV
        df['obv'] = talib.OBV(close, volume)
        df['obv_ema'] = talib.EMA(df['obv'].values, timeperiod=20)
        indicators_calculated += 2
        
        self.processing_stats['indicators_calculated'] += indicators_calculated
        logger.info(f"  Indicadores técnicos calculados: {indicators_calculated}")
        
        return df
    
    def _calculate_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calcular features estadísticas"""
        features_calculated = 0
        
        # Rolling statistics para retornos
        for window in self.config.statistic_windows:
            if window <= len(df):
                # Media y desviación estándar
                df[f'returns_mean_{window}'] = df['returns'].rolling(window).mean()
                df[f'returns_std_{window}'] = df['returns'].rolling(window).std()
                
                # Skewness y Kurtosis
                df[f'returns_skew_{window}'] = df['returns'].rolling(window).skew()
                df[f'returns_kurt_{window}'] = df['returns'].rolling(window).kurt()
                
                # Percentiles
                df[f'returns_q25_{window}'] = df['returns'].rolling(window).quantile(0.25)
                df[f'returns_q75_{window}'] = df['returns'].rolling(window).quantile(0.75)
                
                features_calculated += 6
        
        # Volatilidad realizada
        for window in [5, 20, 60]:
            df[f'realized_vol_{window}'] = df['returns'].rolling(window).std() * np.sqrt(252)
            features_calculated += 1
        
        # Ratio Sharpe rolling
        for window in [20, 60]:
            returns_mean = df['returns'].rolling(window).mean()
            returns_std = df['returns'].rolling(window).std()
            df[f'sharpe_{window}'] = (returns_mean / returns_std) * np.sqrt(252)
            features_calculated += 1
        
        # Autocorrelación
        for lag in [1, 5, 10]:
            df[f'returns_autocorr_{lag}'] = df['returns'].rolling(20).apply(
                lambda x: x.autocorr(lag=lag) if len(x) > lag else np.nan
            )
            features_calculated += 1
        
        logger.info(f"  Features estadísticas calculadas: {features_calculated}")
        
        return df
    
    def _detect_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detectar patrones en los datos"""
        patterns_detected = 0
        
        # Patrones de velas japonesas usando TA-Lib
        pattern_functions = {
            'doji': talib.CDLDOJI,
            'hammer': talib.CDLHAMMER,
            'shooting_star': talib.CDLSHOOTINGSTAR,
            'engulfing': talib.CDLENGULFING,
            'morning_star': talib.CDLMORNINGSTAR,
            'evening_star': talib.CDLEVENINGSTAR,
            'three_white_soldiers': talib.CDL3WHITESOLDIERS,
            'three_black_crows': talib.CDL3BLACKCROWS
        }
        
        o = df['open'].values
        h = df['high'].values
        l = df['low'].values
        c = df['close'].values
        
        for name, func in pattern_functions.items():
            try:
                df[f'pattern_{name}'] = func(o, h, l, c)
                patterns_detected += 1
            except:
                logger.warning(f"No se pudo calcular patrón: {name}")
        
        # Patrones personalizados
        # Soporte/Resistencia
        window = self.config.pattern_lookback
        df['is_resistance'] = df['high'] == df['high'].rolling(window, center=True).max()
        df['is_support'] = df['low'] == df['low'].rolling(window, center=True).min()
        patterns_detected += 2
        
        # Breakouts
        df['breakout_up'] = (df['close'] > df['high'].rolling(window).max().shift(1)).astype(int)
        df['breakout_down'] = (df['close'] < df['low'].rolling(window).min().shift(1)).astype(int)
        patterns_detected += 2
        
        # Tendencia basada en pendiente de SMA
        if 'sma_20' in df.columns:
            sma_slope = df['sma_20'].diff(5) / 5
            df['trend_strength'] = sma_slope / df['close'] * 1000  # Normalizado
            patterns_detected += 1
        
        self.processing_stats['patterns_detected'] += patterns_detected
        logger.info(f"  Patrones detectados: {patterns_detected}")
        
        return df
    
    def _remove_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remover outliers usando método IQR"""
        initial_len = len(df)
        
        # Columnas numéricas para verificar outliers
        numeric_cols = ['returns', 'volume_ratio', 'range_pct']
        
        for col in numeric_cols:
            if col in df.columns:
                # Método IQR
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - self.config.outlier_threshold * IQR
                upper_bound = Q3 + self.config.outlier_threshold * IQR
                
                # Marcar outliers
                outliers = (df[col] < lower_bound) | (df[col] > upper_bound)
                
                if outliers.any():
                    logger.info(f"  Outliers en {col}: {outliers.sum()}")
                    # En lugar de eliminar, podemos recortar
                    df.loc[df[col] < lower_bound, col] = lower_bound
                    df.loc[df[col] > upper_bound, col] = upper_bound
        
        # Z-score para retornos extremos
        if 'returns' in df.columns:
            z_scores = np.abs(stats.zscore(df['returns'].dropna()))
            extreme_returns = z_scores > self.config.outlier_threshold
            
            if extreme_returns.any():
                logger.warning(f"  Retornos extremos detectados: {extreme_returns.sum()}")
        
        outliers_removed = initial_len - len(df)
        self.processing_stats['outliers_removed'] += outliers_removed
        
        if outliers_removed > 0:
            logger.info(f"  Total outliers removidos: {outliers_removed}")
        
        return df
    
    def _normalize_data(self, df: pd.DataFrame, fit_scaler: bool) -> pd.DataFrame:
        """Normalizar datos numéricos"""
        # Seleccionar columnas numéricas
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        # Excluir columnas que no deben ser normalizadas
        exclude_cols = ['open', 'high', 'low', 'close', 'tick_volume']
        columns_to_scale = [col for col in numeric_columns if col not in exclude_cols]
        
        if not columns_to_scale:
            logger.warning("No hay columnas para normalizar")
            return df
        
        # Seleccionar scaler
        if fit_scaler or self.scaler is None:
            if self.config.scaling_method == 'standard':
                self.scaler = StandardScaler()
            elif self.config.scaling_method == 'robust':
                self.scaler = RobustScaler()
            elif self.config.scaling_method == 'minmax':
                self.scaler = MinMaxScaler()
            else:
                raise ValueError(f"Método de escalado no válido: {self.config.scaling_method}")
            
            # Ajustar scaler
            self.scaler.fit(df[columns_to_scale])
            logger.info(f"  Scaler ajustado con {len(columns_to_scale)} columnas")
        
        # Transformar datos
        df_scaled = df.copy()
        df_scaled[columns_to_scale] = self.scaler.transform(df[columns_to_scale])
        
        logger.info(f"  Datos normalizados usando {self.config.scaling_method}")
        
        return df_scaled
    
    def get_feature_importance(self, model=None) -> pd.DataFrame:
        """Obtener importancia de features si hay un modelo disponible"""
        if model is None or not hasattr(model, 'feature_importances_'):
            logger.warning("No hay modelo disponible para calcular importancia")
            return pd.DataFrame()
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Log top features
        logger.info("Top 10 features más importantes:")
        for idx, row in importance_df.head(10).iterrows():
            logger.info(f"  {row['feature']}: {row['importance']:.4f}")
        
        return importance_df
    
    def get_processing_stats(self) -> Dict:
        """Obtener estadísticas de procesamiento"""
        stats = self.processing_stats.copy()
        stats['total_features'] = len(self.feature_names)
        
        # Categorizar features
        feature_categories = {
            'basic': len([f for f in self.feature_names if any(x in f for x in ['returns', 'range', 'gap', 'volume_ratio'])]),
            'technical': len([f for f in self.feature_names if any(x in f for x in ['rsi', 'sma', 'ema', 'macd', 'bb_', 'atr', 'stoch', 'adx'])]),
            'statistical': len([f for f in self.feature_names if any(x in f for x in ['mean', 'std', 'skew', 'kurt', 'sharpe'])]),
            'patterns': len([f for f in self.feature_names if 'pattern' in f or 'breakout' in f])
        }
        
        stats['feature_categories'] = feature_categories
        
        logger.info("="*60)
        logger.info("ESTADÍSTICAS DE PROCESAMIENTO")
        logger.info("="*60)
        logger.info(f"Total procesado: {stats['total_processed']:,} filas")
        logger.info(f"Outliers removidos: {stats['outliers_removed']}")
        logger.info(f"Features totales: {stats['total_features']}")
        logger.info("\nCategorías de features:")
        for cat, count in feature_categories.items():
            logger.info(f"  {cat}: {count}")
        logger.info("="*60)
        
        return stats
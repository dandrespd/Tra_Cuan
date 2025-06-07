'''
3. data/data_collector.py
Ruta: TradingBot_Cuantitative_MT5/data/data_collector.py
Resumen:

Recolecta datos históricos y en tiempo real desde MT5
Valida la integridad de los datos
Maneja gaps y datos faltantes
Almacena datos en caché para optimizar rendimiento
Proporciona datos limpios y listos para el procesamiento
'''
# data/data_collector.py
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
import json
import pickle
from pathlib import Path
import time
from collections import deque

from core.mt5_connector import MT5Connector
from config.settings import settings
from utils.log_config import get_logger, log_risk_alert

logger = get_logger('data')
main_logger = get_logger('main')


class DataCache:
    """Cache para optimizar acceso a datos"""
    
    def __init__(self, max_size: int = 100, ttl_seconds: int = 300):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache: Dict[str, Dict] = {}
        self.access_times: Dict[str, datetime] = {}
        self.hit_count = 0
        self.miss_count = 0
        
    def get(self, key: str) -> Optional[pd.DataFrame]:
        """Obtener datos del cache"""
        if key in self.cache:
            # Verificar TTL
            if (datetime.now() - self.access_times[key]).total_seconds() < self.ttl_seconds:
                self.hit_count += 1
                self.access_times[key] = datetime.now()
                logger.debug(f"Cache hit para {key} - Hit rate: {self.hit_rate:.1%}")
                return self.cache[key]['data'].copy()
            else:
                # Datos expirados
                del self.cache[key]
                del self.access_times[key]
        
        self.miss_count += 1
        logger.debug(f"Cache miss para {key}")
        return None
    
    def put(self, key: str, data: pd.DataFrame):
        """Guardar datos en cache"""
        # Limpiar cache si está lleno
        if len(self.cache) >= self.max_size:
            self._evict_oldest()
        
        self.cache[key] = {
            'data': data.copy(),
            'timestamp': datetime.now()
        }
        self.access_times[key] = datetime.now()
        logger.debug(f"Datos guardados en cache: {key}")
    
    def _evict_oldest(self):
        """Eliminar entrada más antigua"""
        if not self.access_times:
            return
        
        oldest_key = min(self.access_times.items(), key=lambda x: x[1])[0]
        del self.cache[oldest_key]
        del self.access_times[oldest_key]
        logger.debug(f"Entrada eliminada del cache: {oldest_key}")
    
    @property
    def hit_rate(self) -> float:
        """Calcular tasa de aciertos"""
        total = self.hit_count + self.miss_count
        return self.hit_count / total if total > 0 else 0
    
    def get_stats(self) -> Dict:
        """Obtener estadísticas del cache"""
        return {
            'size': len(self.cache),
            'max_size': self.max_size,
            'hit_count': self.hit_count,
            'miss_count': self.miss_count,
            'hit_rate': self.hit_rate,
            'ttl_seconds': self.ttl_seconds
        }


class DataCollector:
    """Recolector principal de datos de mercado"""
    
    def __init__(self, connector: MT5Connector, cache_enabled: bool = True):
        self.connector = connector
        self.cache = DataCache() if cache_enabled else None
        self.collection_stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'data_gaps_detected': 0,
            'data_points_collected': 0,
            'last_collection_time': None
        }
        
        # Buffer para datos en tiempo real
        self.realtime_buffer = deque(maxlen=1000)
        
        # Directorio para almacenamiento
        self.storage_dir = settings.DATA_DIR / 'market_data'
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("DataCollector inicializado")
        logger.info(f"Cache: {'Habilitado' if cache_enabled else 'Deshabilitado'}")
        logger.info(f"Directorio de almacenamiento: {self.storage_dir}")
    
    def collect_training_data(self, symbol: str, timeframe: int, 
                            bars: int = 5000, 
                            end_date: Optional[datetime] = None) -> Optional[pd.DataFrame]:
        """
        Recolectar datos históricos para entrenamiento
        
        Args:
            symbol: Símbolo a recolectar
            timeframe: Timeframe de los datos
            bars: Número de barras a recolectar
            end_date: Fecha final (por defecto: ahora)
            
        Returns:
            DataFrame con datos históricos o None si falla
        """
        logger.info("="*60)
        logger.info("RECOLECCIÓN DE DATOS DE ENTRENAMIENTO")
        logger.info("="*60)
        logger.info(f"Símbolo: {symbol}")
        logger.info(f"Timeframe: {self._timeframe_to_string(timeframe)}")
        logger.info(f"Barras solicitadas: {bars}")
        logger.info(f"Fecha final: {end_date or 'Actual'}")
        
        self.collection_stats['total_requests'] += 1
        start_time = time.time()
        
        try:
            # Intentar obtener del cache primero
            cache_key = f"{symbol}_{timeframe}_{bars}_{end_date}"
            if self.cache:
                cached_data = self.cache.get(cache_key)
                if cached_data is not None:
                    logger.info(f"✅ Datos obtenidos del cache ({len(cached_data)} barras)")
                    return cached_data
            
            # Recolectar en chunks para datasets grandes
            all_data = []
            chunk_size = 1000  # MT5 tiene límites
            remaining_bars = bars
            current_end = end_date or datetime.now()
            
            logger.info(f"Recolectando en chunks de {chunk_size} barras...")
            
            while remaining_bars > 0:
                bars_to_fetch = min(chunk_size, remaining_bars)
                
                # Obtener datos
                if end_date:
                    rates = mt5.copy_rates_from(
                        symbol, timeframe, current_end, bars_to_fetch
                    )
                else:
                    rates = mt5.copy_rates_from_pos(
                        symbol, timeframe, 0, bars_to_fetch
                    )
                
                if rates is None or len(rates) == 0:
                    logger.error(f"No se pudieron obtener datos para {symbol}")
                    self.collection_stats['failed_requests'] += 1
                    return None
                
                # Convertir a DataFrame
                df_chunk = pd.DataFrame(rates)
                all_data.append(df_chunk)
                
                logger.info(f"  Chunk recolectado: {len(df_chunk)} barras")
                
                # Actualizar para siguiente chunk
                remaining_bars -= len(df_chunk)
                if remaining_bars > 0 and len(df_chunk) > 0:
                    current_end = datetime.fromtimestamp(df_chunk.iloc[0]['time'])
                
                # Evitar sobrecarga del servidor
                time.sleep(0.1)
            
            # Combinar todos los chunks
            df = pd.concat(all_data, ignore_index=True)
            
            # Eliminar duplicados
            df = df.drop_duplicates(subset=['time'])
            
            # Convertir tiempo y establecer índice
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df.set_index('time', inplace=True)
            df.sort_index(inplace=True)
            
            # Validar integridad
            validation_result = self._validate_data_integrity(df, timeframe)
            
            if not validation_result['is_valid']:
                logger.warning(f"⚠️ Problemas de integridad detectados: {validation_result['issues']}")
                
                # Intentar reparar
                df = self._repair_data(df, timeframe)
            
            # Estadísticas finales
            elapsed_time = time.time() - start_time
            self.collection_stats['successful_requests'] += 1
            self.collection_stats['data_points_collected'] += len(df)
            self.collection_stats['last_collection_time'] = datetime.now()
            
            logger.info(f"✅ Recolección completada:")
            logger.info(f"  - Barras obtenidas: {len(df)}")
            logger.info(f"  - Rango: {df.index[0]} a {df.index[-1]}")
            logger.info(f"  - Tiempo: {elapsed_time:.2f} segundos")
            logger.info(f"  - Gaps detectados: {validation_result.get('gaps_count', 0)}")
            
            # Guardar en cache
            if self.cache and len(df) > 0:
                self.cache.put(cache_key, df)
            
            # Guardar en disco si es un dataset grande
            if len(df) > 1000:
                self._save_to_disk(df, symbol, timeframe)
            
            return df
            
        except Exception as e:
            logger.error(f"❌ Error recolectando datos: {e}")
            self.collection_stats['failed_requests'] += 1
            return None
    
    def get_latest_data(self, symbol: str, timeframe: int, 
                       bars: int = 100) -> Optional[pd.DataFrame]:
        """Obtener datos más recientes para análisis"""
        logger.debug(f"Obteniendo últimos {bars} datos para {symbol}")
        
        try:
            # Usar cache con TTL más corto para datos recientes
            cache_key = f"latest_{symbol}_{timeframe}_{bars}"
            
            if self.cache:
                # TTL más corto para datos recientes
                old_ttl = self.cache.ttl_seconds
                self.cache.ttl_seconds = 60  # 1 minuto
                
                cached_data = self.cache.get(cache_key)
                
                self.cache.ttl_seconds = old_ttl  # Restaurar
                
                if cached_data is not None:
                    return cached_data
            
            # Obtener datos frescos
            rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, bars)
            
            if rates is None:
                logger.error(f"No se pudieron obtener datos recientes para {symbol}")
                return None
            
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df.set_index('time', inplace=True)
            
            # Agregar al buffer de tiempo real
            self._update_realtime_buffer(df)
            
            # Cache con TTL corto
            if self.cache:
                old_ttl = self.cache.ttl_seconds
                self.cache.ttl_seconds = 60
                self.cache.put(cache_key, df)
                self.cache.ttl_seconds = old_ttl
            
            return df
            
        except Exception as e:
            logger.error(f"Error obteniendo datos recientes: {e}")
            return None
    
    def stream_realtime_data(self, symbol: str, callback=None):
        """
        Stream de datos en tiempo real
        
        Args:
            symbol: Símbolo a monitorear
            callback: Función a llamar con cada nuevo tick
        """
        logger.info(f"Iniciando stream de datos para {symbol}")
        
        last_tick_time = 0
        error_count = 0
        max_errors = 10
        
        try:
            while error_count < max_errors:
                try:
                    # Obtener último tick
                    tick = mt5.symbol_info_tick(symbol)
                    
                    if tick is None:
                        error_count += 1
                        continue
                    
                    # Verificar si es un tick nuevo
                    if tick.time > last_tick_time:
                        last_tick_time = tick.time
                        
                        # Crear DataFrame del tick
                        tick_data = pd.DataFrame([{
                            'time': datetime.fromtimestamp(tick.time),
                            'bid': tick.bid,
                            'ask': tick.ask,
                            'last': tick.last,
                            'volume': tick.volume,
                            'spread': tick.ask - tick.bid
                        }])
                        tick_data.set_index('time', inplace=True)
                        
                        # Agregar al buffer
                        self.realtime_buffer.append(tick_data)
                        
                        # Callback si se proporciona
                        if callback:
                            callback(tick_data)
                        
                        # Reset contador de errores
                        error_count = 0
                    
                    # Pequeña pausa para no sobrecargar
                    time.sleep(0.1)
                    
                except KeyboardInterrupt:
                    logger.info("Stream detenido por usuario")
                    break
                except Exception as e:
                    error_count += 1
                    logger.error(f"Error en stream ({error_count}/{max_errors}): {e}")
                    time.sleep(1)
                    
        except Exception as e:
            logger.error(f"Error fatal en stream: {e}")
    
    def _validate_data_integrity(self, df: pd.DataFrame, timeframe: int) -> Dict:
        """Validar integridad de los datos"""
        issues = []
        gaps_count = 0
        
        # Verificar datos faltantes
        null_counts = df.isnull().sum()
        if null_counts.any():
            issues.append(f"Datos faltantes: {null_counts[null_counts > 0].to_dict()}")
        
        # Verificar gaps temporales
        expected_diff = self._get_expected_time_diff(timeframe)
        time_diffs = df.index.to_series().diff()
        
        gaps = time_diffs[time_diffs > expected_diff * 1.5]
        if len(gaps) > 0:
            gaps_count = len(gaps)
            issues.append(f"Gaps temporales detectados: {gaps_count}")
            
            # Log de gaps significativos
            significant_gaps = gaps[gaps > expected_diff * 5]
            if len(significant_gaps) > 0:
                logger.warning(f"Gaps significativos encontrados:")
                for idx, gap in significant_gaps.items():
                    logger.warning(f"  - {idx}: {gap}")
        
        # Verificar integridad de precios
        price_issues = []
        
        # High debe ser >= Low
        invalid_hl = df[df['high'] < df['low']]
        if len(invalid_hl) > 0:
            price_issues.append(f"High < Low en {len(invalid_hl)} barras")
        
        # Open y Close deben estar entre High y Low
        invalid_open = df[(df['open'] > df['high']) | (df['open'] < df['low'])]
        if len(invalid_open) > 0:
            price_issues.append(f"Open fuera de rango en {len(invalid_open)} barras")
        
        invalid_close = df[(df['close'] > df['high']) | (df['close'] < df['low'])]
        if len(invalid_close) > 0:
            price_issues.append(f"Close fuera de rango en {len(invalid_close)} barras")
        
        if price_issues:
            issues.extend(price_issues)
        
        # Verificar volumen
        zero_volume = len(df[df['tick_volume'] == 0])
        if zero_volume > 0:
            issues.append(f"Volumen cero en {zero_volume} barras")
        
        # Verificar valores extremos
        for col in ['open', 'high', 'low', 'close']:
            mean = df[col].mean()
            std = df[col].std()
            outliers = df[(df[col] > mean + 4*std) | (df[col] < mean - 4*std)]
            if len(outliers) > 0:
                issues.append(f"Valores extremos en {col}: {len(outliers)} barras")
        
        is_valid = len(issues) == 0
        
        if not is_valid:
            logger.warning("Problemas de integridad encontrados:")
            for issue in issues:
                logger.warning(f"  - {issue}")
            
            # Alerta de riesgo si hay muchos problemas
            if len(issues) > 5 or gaps_count > 50:
                log_risk_alert(
                    "INTEGRIDAD DE DATOS COMPROMETIDA",
                    f"Múltiples problemas detectados en los datos de {df.index[0]} a {df.index[-1]}",
                    {
                        'total_issues': len(issues),
                        'gaps_count': gaps_count,
                        'data_points': len(df)
                    }
                )
        
        return {
            'is_valid': is_valid,
            'issues': issues,
            'gaps_count': gaps_count
        }
    
    def _repair_data(self, df: pd.DataFrame, timeframe: int) -> pd.DataFrame:
        """Intentar reparar problemas en los datos"""
        logger.info("Reparando datos...")
        df_repaired = df.copy()
        
        # 1. Interpolar valores faltantes
        numeric_columns = ['open', 'high', 'low', 'close', 'tick_volume']
        for col in numeric_columns:
            if col in df_repaired.columns:
                df_repaired[col] = df_repaired[col].interpolate(method='linear', limit=5)
        
        # 2. Corregir High/Low inválidos
        mask = df_repaired['high'] < df_repaired['low']
        if mask.any():
            logger.info(f"  Corrigiendo {mask.sum()} barras con High < Low")
            df_repaired.loc[mask, 'high'] = df_repaired.loc[mask, 'low']
        
        # 3. Corregir Open/Close fuera de rango
        # Open
        mask_open_high = df_repaired['open'] > df_repaired['high']
        df_repaired.loc[mask_open_high, 'open'] = df_repaired.loc[mask_open_high, 'high']
        
        mask_open_low = df_repaired['open'] < df_repaired['low']
        df_repaired.loc[mask_open_low, 'open'] = df_repaired.loc[mask_open_low, 'low']
        
        # Close
        mask_close_high = df_repaired['close'] > df_repaired['high']
        df_repaired.loc[mask_close_high, 'close'] = df_repaired.loc[mask_close_high, 'high']
        
        mask_close_low = df_repaired['close'] < df_repaired['low']
        df_repaired.loc[mask_close_low, 'close'] = df_repaired.loc[mask_close_low, 'low']
        
        # 4. Rellenar volumen cero con promedio local
        zero_volume_mask = df_repaired['tick_volume'] == 0
        if zero_volume_mask.any():
            logger.info(f"  Rellenando {zero_volume_mask.sum()} barras con volumen cero")
            avg_volume = df_repaired['tick_volume'].rolling(20, min_periods=1).mean()
            df_repaired.loc[zero_volume_mask, 'tick_volume'] = avg_volume[zero_volume_mask]
        
        # 5. Rellenar gaps temporales si son pequeños
        expected_freq = self._get_pandas_freq(timeframe)
        if expected_freq:
            df_reindexed = df_repaired.resample(expected_freq).asfreq()
            
            # Interpolar gaps pequeños (máximo 5 períodos)
            for col in numeric_columns:
                if col in df_reindexed.columns:
                    df_reindexed[col] = df_reindexed[col].interpolate(
                        method='linear', 
                        limit=5,
                        limit_area='inside'
                    )
            
            # Eliminar filas que siguen siendo NaN
            df_repaired = df_reindexed.dropna()
        
        logger.info(f"  Reparación completada: {len(df)} -> {len(df_repaired)} barras")
        
        return df_repaired
    
    def _get_expected_time_diff(self, timeframe: int) -> timedelta:
        """Obtener diferencia de tiempo esperada según timeframe"""
        timeframe_minutes = {
            mt5.TIMEFRAME_M1: 1,
            mt5.TIMEFRAME_M5: 5,
            mt5.TIMEFRAME_M15: 15,
            mt5.TIMEFRAME_M30: 30,
            mt5.TIMEFRAME_H1: 60,
            mt5.TIMEFRAME_H4: 240,
            mt5.TIMEFRAME_D1: 1440,
            mt5.TIMEFRAME_W1: 10080,
            mt5.TIMEFRAME_MN1: 43200
        }
        
        minutes = timeframe_minutes.get(timeframe, 60)
        return timedelta(minutes=minutes)
    
    def _get_pandas_freq(self, timeframe: int) -> Optional[str]:
        """Obtener frecuencia de pandas según timeframe"""
        freq_map = {
            mt5.TIMEFRAME_M1: '1T',
            mt5.TIMEFRAME_M5: '5T',
            mt5.TIMEFRAME_M15: '15T',
            mt5.TIMEFRAME_M30: '30T',
            mt5.TIMEFRAME_H1: '1H',
            mt5.TIMEFRAME_H4: '4H',
            mt5.TIMEFRAME_D1: '1D',
            mt5.TIMEFRAME_W1: '1W',
            mt5.TIMEFRAME_MN1: '1M'
        }
        
        return freq_map.get(timeframe)
    
    def _timeframe_to_string(self, timeframe: int) -> str:
        """Convertir timeframe a string"""
        timeframes = {
            mt5.TIMEFRAME_M1: "M1",
            mt5.TIMEFRAME_M5: "M5",
            mt5.TIMEFRAME_M15: "M15",
            mt5.TIMEFRAME_M30: "M30",
            mt5.TIMEFRAME_H1: "H1",
            mt5.TIMEFRAME_H4: "H4",
            mt5.TIMEFRAME_D1: "D1",
            mt5.TIMEFRAME_W1: "W1",
            mt5.TIMEFRAME_MN1: "MN1"
        }
        return timeframes.get(timeframe, str(timeframe))
    
    def _update_realtime_buffer(self, df: pd.DataFrame):
        """Actualizar buffer de tiempo real"""
        # Agregar últimas barras al buffer
        for _, row in df.tail(10).iterrows():
            self.realtime_buffer.append(row.to_dict())
    
    def _save_to_disk(self, df: pd.DataFrame, symbol: str, timeframe: int):
        """Guardar datos en disco para respaldo"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{symbol}_{self._timeframe_to_string(timeframe)}_{timestamp}.pkl"
            filepath = self.storage_dir / filename
            
            df.to_pickle(filepath)
            logger.info(f"Datos guardados en: {filepath}")
            
            # También guardar en CSV para inspección manual
            csv_path = filepath.with_suffix('.csv')
            df.to_csv(csv_path)
            
        except Exception as e:
            logger.error(f"Error guardando datos: {e}")
    
    def load_from_disk(self, symbol: str, timeframe: int, 
                      date: Optional[str] = None) -> Optional[pd.DataFrame]:
        """Cargar datos desde disco"""
        try:
            pattern = f"{symbol}_{self._timeframe_to_string(timeframe)}_*.pkl"
            files = list(self.storage_dir.glob(pattern))
            
            if not files:
                logger.warning(f"No se encontraron archivos para {pattern}")
                return None
            
            # Ordenar por fecha (más reciente primero)
            files.sort(reverse=True)
            
            # Cargar el archivo más reciente o el de la fecha especificada
            if date:
                for file in files:
                    if date in file.name:
                        df = pd.read_pickle(file)
                        logger.info(f"Datos cargados desde: {file}")
                        return df
            else:
                df = pd.read_pickle(files[0])
                logger.info(f"Datos cargados desde: {files[0]}")
                return df
                
        except Exception as e:
            logger.error(f"Error cargando datos: {e}")
            return None
    
    def get_statistics(self) -> Dict:
        """Obtener estadísticas del recolector"""
        stats = self.collection_stats.copy()
        
        # Agregar estadísticas del cache
        if self.cache:
            stats['cache_stats'] = self.cache.get_stats()
        
        # Agregar información del buffer
        stats['realtime_buffer_size'] = len(self.realtime_buffer)
        
        # Calcular tasa de éxito
        total = stats['total_requests']
        if total > 0:
            stats['success_rate'] = stats['successful_requests'] / total
        else:
            stats['success_rate'] = 0
        
        # Log de estadísticas
        logger.info("="*60)
        logger.info("ESTADÍSTICAS DEL RECOLECTOR DE DATOS")
        logger.info("="*60)
        logger.info(f"Total solicitudes: {stats['total_requests']}")
        logger.info(f"Exitosas: {stats['successful_requests']}")
        logger.info(f"Fallidas: {stats['failed_requests']}")
        logger.info(f"Tasa de éxito: {stats['success_rate']:.1%}")
        logger.info(f"Puntos de datos recolectados: {stats['data_points_collected']:,}")
        logger.info(f"Gaps detectados: {stats['data_gaps_detected']}")
        
        if self.cache:
            cache_stats = stats['cache_stats']
            logger.info(f"\nEstadísticas del Cache:")
            logger.info(f"  Tamaño: {cache_stats['size']}/{cache_stats['max_size']}")
            logger.info(f"  Hit rate: {cache_stats['hit_rate']:.1%}")
            logger.info(f"  TTL: {cache_stats['ttl_seconds']}s")
        
        logger.info("="*60)
        
        return stats
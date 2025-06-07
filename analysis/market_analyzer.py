# analysis/market_analyzer.py
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

from utils.log_config import get_logger

# Imports opcionales
try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False

try:
    from scipy import stats
    from scipy.signal import find_peaks
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

logger = get_logger('analysis')


class MarketRegime(Enum):
    """Regímenes de mercado identificados"""
    BULL_MARKET = "BULL_MARKET"
    BEAR_MARKET = "BEAR_MARKET"
    SIDEWAYS = "SIDEWAYS"
    HIGH_VOLATILITY = "HIGH_VOLATILITY"
    LOW_VOLATILITY = "LOW_VOLATILITY"
    BREAKOUT = "BREAKOUT"
    REVERSAL = "REVERSAL"
    UNKNOWN = "UNKNOWN"


class TrendDirection(Enum):
    """Dirección de tendencia"""
    STRONG_UP = "STRONG_UP"
    WEAK_UP = "WEAK_UP"
    NEUTRAL = "NEUTRAL"
    WEAK_DOWN = "WEAK_DOWN"
    STRONG_DOWN = "STRONG_DOWN"


class VolatilityState(Enum):
    """Estados de volatilidad"""
    VERY_LOW = "VERY_LOW"
    LOW = "LOW"
    NORMAL = "NORMAL"
    HIGH = "HIGH"
    EXTREME = "EXTREME"


@dataclass
class SupportResistanceLevel:
    """Nivel de soporte o resistencia"""
    price: float
    level_type: str  # 'support' or 'resistance'
    strength: float  # 0-1, fuerza del nivel
    touches: int     # Número de veces que se ha tocado
    last_touch: datetime
    confidence: float  # Confianza en el nivel
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'price': self.price,
            'level_type': self.level_type,
            'strength': self.strength,
            'touches': self.touches,
            'last_touch': self.last_touch.isoformat(),
            'confidence': self.confidence
        }


@dataclass
class MarketConditions:
    """Condiciones actuales del mercado"""
    # Identificación temporal
    timestamp: datetime
    symbol: str
    timeframe: str
    
    # Régimen y tendencia
    market_regime: MarketRegime
    trend_direction: TrendDirection
    trend_strength: float  # 0-1
    
    # Volatilidad
    volatility_state: VolatilityState
    current_volatility: float
    volatility_percentile: float  # Percentil histórico
    
    # Momentum
    momentum: float
    momentum_direction: str  # 'increasing', 'decreasing', 'stable'
    
    # Niveles técnicos
    support_levels: List[SupportResistanceLevel] = field(default_factory=list)
    resistance_levels: List[SupportResistanceLevel] = field(default_factory=list)
    key_levels: Dict[str, float] = field(default_factory=dict)
    
    # Indicadores técnicos
    rsi: Optional[float] = None
    macd_signal: Optional[str] = None  # 'bullish', 'bearish', 'neutral'
    bb_position: Optional[float] = None  # Posición en Bollinger Bands (0-1)
    
    # Estructura de mercado
    market_structure: str = "unknown"  # 'trending', 'ranging', 'transitional'
    liquidity_condition: str = "normal"  # 'high', 'normal', 'low'
    
    # Correlaciones
    correlation_with_indices: Dict[str, float] = field(default_factory=dict)
    
    # Métricas adicionales
    gap_analysis: Dict[str, Any] = field(default_factory=dict)
    volume_analysis: Dict[str, Any] = field(default_factory=dict)
    breadth_indicators: Dict[str, float] = field(default_factory=dict)
    
    # Confianza general
    analysis_confidence: float = 0.0  # Confianza general del análisis
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario"""
        return {
            'timestamp': self.timestamp.isoformat(),
            'symbol': self.symbol,
            'timeframe': self.timeframe,
            'market_regime': self.market_regime.value,
            'trend_direction': self.trend_direction.value,
            'trend_strength': self.trend_strength,
            'volatility_state': self.volatility_state.value,
            'current_volatility': self.current_volatility,
            'volatility_percentile': self.volatility_percentile,
            'momentum': self.momentum,
            'momentum_direction': self.momentum_direction,
            'support_levels': [level.to_dict() for level in self.support_levels],
            'resistance_levels': [level.to_dict() for level in self.resistance_levels],
            'key_levels': self.key_levels,
            'rsi': self.rsi,
            'macd_signal': self.macd_signal,
            'bb_position': self.bb_position,
            'market_structure': self.market_structure,
            'liquidity_condition': self.liquidity_condition,
            'correlation_with_indices': self.correlation_with_indices,
            'gap_analysis': self.gap_analysis,
            'volume_analysis': self.volume_analysis,
            'breadth_indicators': self.breadth_indicators,
            'analysis_confidence': self.analysis_confidence
        }


class MarketAnalyzer:
    """
    Analizador principal de condiciones de mercado
    
    Funcionalidades:
    - Detección de regímenes de mercado
    - Análisis de tendencias y momentum
    - Identificación de niveles de soporte/resistencia
    - Análisis de volatilidad y estructura
    - Cálculo de indicadores técnicos avanzados
    """
    
    def __init__(self, symbol: str = "EURUSD"):
        """
        Inicializar analizador de mercado
        
        Args:
            symbol: Símbolo principal a analizar
        """
        self.symbol = symbol
        
        # Configuración del análisis
        self.analysis_periods = {
            'short': 20,
            'medium': 50,
            'long': 200
        }
        
        # Umbrales de volatilidad (percentiles históricos)
        self.volatility_thresholds = {
            'very_low': 10,
            'low': 25,
            'high': 75,
            'extreme': 90
        }
        
        # Configuración de soporte/resistencia
        self.sr_config = {
            'lookback_period': 100,
            'min_touches': 2,
            'price_tolerance': 0.0005,  # 5 pips para forex
            'min_strength': 0.3
        }
        
        # Historial de análisis
        self.analysis_history: List[MarketConditions] = []
        
        # Cache de cálculos
        self._volatility_cache: Dict[str, float] = {}
        self._sr_cache: Dict[str, List[SupportResistanceLevel]] = {}
        
        logger.info(f"MarketAnalyzer inicializado para {symbol}")
    
    def analyze(self, data: pd.DataFrame, 
                additional_data: Dict[str, pd.DataFrame] = None) -> MarketConditions:
        """
        Análisis completo de condiciones de mercado
        
        Args:
            data: DataFrame con datos OHLCV principales
            additional_data: Datos adicionales (otros símbolos, timeframes)
            
        Returns:
            MarketConditions con análisis completo
        """
        logger.info("Iniciando análisis de mercado...")
        
        if data.empty or len(data) < 50:
            logger.warning("Datos insuficientes para análisis")
            return self._create_default_conditions()
        
        try:
            # Crear objeto de condiciones
            conditions = MarketConditions(
                timestamp=datetime.now(),
                symbol=self.symbol,
                timeframe=self._detect_timeframe(data)
            )
            
            # 1. Análisis de tendencia
            self._analyze_trend(data, conditions)
            
            # 2. Análisis de volatilidad
            self._analyze_volatility(data, conditions)
            
            # 3. Análisis de momentum
            self._analyze_momentum(data, conditions)
            
            # 4. Detectar régimen de mercado
            self._detect_market_regime(data, conditions)
            
            # 5. Identificar niveles de soporte/resistencia
            self._identify_support_resistance(data, conditions)
            
            # 6. Calcular indicadores técnicos
            self._calculate_technical_indicators(data, conditions)
            
            # 7. Analizar estructura de mercado
            self._analyze_market_structure(data, conditions)
            
            # 8. Análisis de volumen
            self._analyze_volume(data, conditions)
            
            # 9. Análisis de gaps
            self._analyze_gaps(data, conditions)
            
            # 10. Correlaciones (si hay datos adicionales)
            if additional_data:
                self._analyze_correlations(data, additional_data, conditions)
            
            # 11. Calcular confianza del análisis
            conditions.analysis_confidence = self._calculate_analysis_confidence(conditions)
            
            # Guardar en historial
            self.analysis_history.append(conditions)
            
            # Mantener historial limitado
            if len(self.analysis_history) > 100:
                self.analysis_history = self.analysis_history[-50:]
            
            # Log resumen
            self._log_analysis_summary(conditions)
            
            return conditions
            
        except Exception as e:
            logger.error(f"Error en análisis de mercado: {e}")
            return self._create_default_conditions()
    
    def _analyze_trend(self, data: pd.DataFrame, conditions: MarketConditions):
        """Analizar tendencia y dirección"""
        try:
            # Calcular medias móviles
            sma_short = data['close'].rolling(self.analysis_periods['short']).mean()
            sma_medium = data['close'].rolling(self.analysis_periods['medium']).mean()
            sma_long = data['close'].rolling(self.analysis_periods['long']).mean()
            
            current_price = data['close'].iloc[-1]
            short_ma = sma_short.iloc[-1]
            medium_ma = sma_medium.iloc[-1]
            long_ma = sma_long.iloc[-1]
            
            # Determinar dirección de tendencia
            if current_price > short_ma > medium_ma > long_ma:
                trend_direction = TrendDirection.STRONG_UP
                trend_strength = 0.8
            elif current_price > short_ma > medium_ma:
                trend_direction = TrendDirection.WEAK_UP
                trend_strength = 0.6
            elif current_price < short_ma < medium_ma < long_ma:
                trend_direction = TrendDirection.STRONG_DOWN
                trend_strength = 0.8
            elif current_price < short_ma < medium_ma:
                trend_direction = TrendDirection.WEAK_DOWN
                trend_strength = 0.6
            else:
                trend_direction = TrendDirection.NEUTRAL
                trend_strength = 0.3
            
            # Ajustar fuerza basada en pendiente
            ma_slope = (short_ma - sma_short.iloc[-10]) / sma_short.iloc[-10]
            slope_factor = min(abs(ma_slope) * 1000, 0.5)  # Normalizar
            trend_strength = min(trend_strength + slope_factor, 1.0)
            
            conditions.trend_direction = trend_direction
            conditions.trend_strength = trend_strength
            
            logger.debug(f"Tendencia: {trend_direction.value}, Fuerza: {trend_strength:.2f}")
            
        except Exception as e:
            logger.warning(f"Error analizando tendencia: {e}")
            conditions.trend_direction = TrendDirection.NEUTRAL
            conditions.trend_strength = 0.5
    
    def _analyze_volatility(self, data: pd.DataFrame, conditions: MarketConditions):
        """Analizar volatilidad actual y estado"""
        try:
            # Calcular volatilidad realizada
            returns = data['close'].pct_change().dropna()
            current_vol = returns.rolling(20).std().iloc[-1] * np.sqrt(252)  # Anualizada
            
            # Volatilidad histórica para percentiles
            historical_vols = returns.rolling(20).std() * np.sqrt(252)
            vol_percentile = stats.percentileofscore(historical_vols.dropna(), current_vol)
            
            # Determinar estado de volatilidad
            if vol_percentile <= self.volatility_thresholds['very_low']:
                vol_state = VolatilityState.VERY_LOW
            elif vol_percentile <= self.volatility_thresholds['low']:
                vol_state = VolatilityState.LOW
            elif vol_percentile >= self.volatility_thresholds['extreme']:
                vol_state = VolatilityState.EXTREME
            elif vol_percentile >= self.volatility_thresholds['high']:
                vol_state = VolatilityState.HIGH
            else:
                vol_state = VolatilityState.NORMAL
            
            conditions.current_volatility = current_vol
            conditions.volatility_percentile = vol_percentile
            conditions.volatility_state = vol_state
            
            # Cache para uso posterior
            self._volatility_cache['current'] = current_vol
            self._volatility_cache['percentile'] = vol_percentile
            
            logger.debug(f"Volatilidad: {current_vol:.4f} ({vol_percentile:.1f}%ile) - {vol_state.value}")
            
        except Exception as e:
            logger.warning(f"Error analizando volatilidad: {e}")
            conditions.current_volatility = 0.02
            conditions.volatility_percentile = 50.0
            conditions.volatility_state = VolatilityState.NORMAL
    
    def _analyze_momentum(self, data: pd.DataFrame, conditions: MarketConditions):
        """Analizar momentum y cambios"""
        try:
            # Calcular momentum de diferentes períodos
            momentum_5 = data['close'].pct_change(5).iloc[-1]
            momentum_10 = data['close'].pct_change(10).iloc[-1]
            momentum_20 = data['close'].pct_change(20).iloc[-1]
            
            # Momentum ponderado
            momentum = (momentum_5 * 0.5 + momentum_10 * 0.3 + momentum_20 * 0.2)
            
            # Dirección del momentum
            recent_momentum = [
                data['close'].pct_change(5).iloc[-3],
                data['close'].pct_change(5).iloc[-2],
                data['close'].pct_change(5).iloc[-1]
            ]
            
            if len(recent_momentum) >= 2:
                if recent_momentum[-1] > recent_momentum[-2]:
                    momentum_direction = "increasing"
                elif recent_momentum[-1] < recent_momentum[-2]:
                    momentum_direction = "decreasing"
                else:
                    momentum_direction = "stable"
            else:
                momentum_direction = "stable"
            
            conditions.momentum = momentum
            conditions.momentum_direction = momentum_direction
            
            logger.debug(f"Momentum: {momentum:.4f} ({momentum_direction})")
            
        except Exception as e:
            logger.warning(f"Error analizando momentum: {e}")
            conditions.momentum = 0.0
            conditions.momentum_direction = "stable"
    
    def _detect_market_regime(self, data: pd.DataFrame, conditions: MarketConditions):
        """Detectar régimen de mercado actual"""
        try:
            # Combinar indicadores para determinar régimen
            trend_strength = conditions.trend_strength
            trend_direction = conditions.trend_direction
            volatility_state = conditions.volatility_state
            momentum = abs(conditions.momentum)
            
            # Lógica de detección de régimen
            if volatility_state in [VolatilityState.EXTREME]:
                regime = MarketRegime.HIGH_VOLATILITY
            elif volatility_state in [VolatilityState.VERY_LOW, VolatilityState.LOW]:
                regime = MarketRegime.LOW_VOLATILITY
            elif trend_direction in [TrendDirection.STRONG_UP, TrendDirection.WEAK_UP] and trend_strength > 0.6:
                regime = MarketRegime.BULL_MARKET
            elif trend_direction in [TrendDirection.STRONG_DOWN, TrendDirection.WEAK_DOWN] and trend_strength > 0.6:
                regime = MarketRegime.BEAR_MARKET
            elif trend_direction == TrendDirection.NEUTRAL and momentum < 0.01:
                regime = MarketRegime.SIDEWAYS
            else:
                # Detectar breakouts y reversals
                if momentum > 0.02 and trend_strength > 0.7:
                    regime = MarketRegime.BREAKOUT
                elif self._detect_reversal_pattern(data):
                    regime = MarketRegime.REVERSAL
                else:
                    regime = MarketRegime.UNKNOWN
            
            conditions.market_regime = regime
            
            logger.debug(f"Régimen de mercado: {regime.value}")
            
        except Exception as e:
            logger.warning(f"Error detectando régimen de mercado: {e}")
            conditions.market_regime = MarketRegime.UNKNOWN
    
    def _identify_support_resistance(self, data: pd.DataFrame, conditions: MarketConditions):
        """Identificar niveles de soporte y resistencia"""
        try:
            if not SCIPY_AVAILABLE:
                logger.debug("Scipy no disponible para análisis S/R")
                return
            
            lookback = min(self.sr_config['lookback_period'], len(data))
            recent_data = data.tail(lookback)
            
            # Identificar picos y valles
            highs = recent_data['high'].values
            lows = recent_data['low'].values
            
            # Encontrar picos (resistencias)
            resistance_indices, _ = find_peaks(highs, distance=5)
            resistance_prices = highs[resistance_indices]
            
            # Encontrar valles (soportes)
            support_indices, _ = find_peaks(-lows, distance=5)
            support_prices = lows[support_indices]
            
            # Procesar resistencias
            resistance_levels = []
            for price in resistance_prices:
                level = self._create_sr_level(price, 'resistance', recent_data)
                if level and level.confidence >= self.sr_config['min_strength']:
                    resistance_levels.append(level)
            
            # Procesar soportes
            support_levels = []
            for price in support_prices:
                level = self._create_sr_level(price, 'support', recent_data)
                if level and level.confidence >= self.sr_config['min_strength']:
                    support_levels.append(level)
            
            # Ordenar por proximidad al precio actual
            current_price = data['close'].iloc[-1]
            resistance_levels.sort(key=lambda x: abs(x.price - current_price))
            support_levels.sort(key=lambda x: abs(x.price - current_price))
            
            # Tomar los más relevantes
            conditions.resistance_levels = resistance_levels[:5]
            conditions.support_levels = support_levels[:5]
            
            # Niveles clave
            if resistance_levels:
                conditions.key_levels['nearest_resistance'] = resistance_levels[0].price
            if support_levels:
                conditions.key_levels['nearest_support'] = support_levels[0].price
            
            logger.debug(f"Identificados {len(resistance_levels)} resistencias y {len(support_levels)} soportes")
            
        except Exception as e:
            logger.warning(f"Error identificando S/R: {e}")
    
    def _create_sr_level(self, price: float, level_type: str, 
                        data: pd.DataFrame) -> Optional[SupportResistanceLevel]:
        """Crear nivel de soporte/resistencia"""
        try:
            tolerance = price * self.sr_config['price_tolerance']
            
            # Contar toques
            if level_type == 'resistance':
                touches = data[(data['high'] >= price - tolerance) & 
                              (data['high'] <= price + tolerance)]
            else:
                touches = data[(data['low'] >= price - tolerance) & 
                              (data['low'] <= price + tolerance)]
            
            touch_count = len(touches)
            
            if touch_count < self.sr_config['min_touches']:
                return None
            
            # Calcular fuerza y confianza
            strength = min(touch_count / 5.0, 1.0)  # Normalizar
            
            # Última vez que se tocó
            last_touch = touches.index[-1] if len(touches) > 0 else datetime.now()
            
            # Confianza basada en múltiples factores
            time_factor = max(0.5, 1.0 - (datetime.now() - last_touch).days / 30)
            confidence = strength * time_factor
            
            return SupportResistanceLevel(
                price=price,
                level_type=level_type,
                strength=strength,
                touches=touch_count,
                last_touch=last_touch,
                confidence=confidence
            )
            
        except Exception as e:
            logger.warning(f"Error creando nivel S/R: {e}")
            return None
    
    def _calculate_technical_indicators(self, data: pd.DataFrame, conditions: MarketConditions):
        """Calcular indicadores técnicos principales"""
        try:
            # RSI
            if TALIB_AVAILABLE:
                rsi = talib.RSI(data['close'].values, timeperiod=14)
                conditions.rsi = rsi[-1] if len(rsi) > 0 and not np.isnan(rsi[-1]) else None
            else:
                # RSI simplificado
                delta = data['close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                rsi = 100 - (100 / (1 + rs))
                conditions.rsi = rsi.iloc[-1] if not np.isnan(rsi.iloc[-1]) else None
            
            # MACD
            if TALIB_AVAILABLE:
                macd, macd_signal, macd_hist = talib.MACD(data['close'].values)
                if len(macd) > 0 and not np.isnan(macd[-1]):
                    if macd[-1] > macd_signal[-1]:
                        conditions.macd_signal = "bullish"
                    elif macd[-1] < macd_signal[-1]:
                        conditions.macd_signal = "bearish"
                    else:
                        conditions.macd_signal = "neutral"
            
            # Bollinger Bands
            if TALIB_AVAILABLE:
                bb_upper, bb_middle, bb_lower = talib.BBANDS(data['close'].values)
                if len(bb_upper) > 0 and not np.isnan(bb_upper[-1]):
                    current_price = data['close'].iloc[-1]
                    bb_position = (current_price - bb_lower[-1]) / (bb_upper[-1] - bb_lower[-1])
                    conditions.bb_position = max(0, min(1, bb_position))
            
            logger.debug(f"Indicadores: RSI={conditions.rsi}, MACD={conditions.macd_signal}, BB={conditions.bb_position}")
            
        except Exception as e:
            logger.warning(f"Error calculando indicadores técnicos: {e}")
    
    def _analyze_market_structure(self, data: pd.DataFrame, conditions: MarketConditions):
        """Analizar estructura de mercado"""
        try:
            # Determinar si el mercado está en tendencia o rango
            price_range = data['high'].rolling(20).max() - data['low'].rolling(20).min()
            price_movement = abs(data['close'].iloc[-1] - data['close'].iloc[-21])
            
            range_ratio = price_movement / price_range.iloc[-1] if price_range.iloc[-1] > 0 else 0
            
            if range_ratio > 0.7:
                conditions.market_structure = "trending"
            elif range_ratio < 0.3:
                conditions.market_structure = "ranging"
            else:
                conditions.market_structure = "transitional"
            
            logger.debug(f"Estructura de mercado: {conditions.market_structure}")
            
        except Exception as e:
            logger.warning(f"Error analizando estructura: {e}")
            conditions.market_structure = "unknown"
    
    def _analyze_volume(self, data: pd.DataFrame, conditions: MarketConditions):
        """Analizar patrones de volumen"""
        try:
            if 'tick_volume' not in data.columns:
                return
            
            # Volumen promedio
            avg_volume = data['tick_volume'].rolling(20).mean().iloc[-1]
            current_volume = data['tick_volume'].iloc[-1]
            
            # Ratio de volumen
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
            
            # Tendencia de volumen
            volume_trend = data['tick_volume'].rolling(5).mean().iloc[-1] / data['tick_volume'].rolling(10).mean().iloc[-1] - 1
            
            # Determinar condición de liquidez
            if volume_ratio > 1.5:
                conditions.liquidity_condition = "high"
            elif volume_ratio < 0.5:
                conditions.liquidity_condition = "low"
            else:
                conditions.liquidity_condition = "normal"
            
            conditions.volume_analysis = {
                'volume_ratio': volume_ratio,
                'volume_trend': volume_trend,
                'avg_volume': avg_volume,
                'current_volume': current_volume
            }
            
            logger.debug(f"Volumen: ratio={volume_ratio:.2f}, liquidez={conditions.liquidity_condition}")
            
        except Exception as e:
            logger.warning(f"Error analizando volumen: {e}")
    
    def _analyze_gaps(self, data: pd.DataFrame, conditions: MarketConditions):
        """Analizar gaps de precio"""
        try:
            # Detectar gaps
            gaps = []
            
            for i in range(1, len(data)):
                prev_close = data['close'].iloc[i-1]
                current_open = data['open'].iloc[i]
                current_high = data['high'].iloc[i]
                current_low = data['low'].iloc[i]
                
                # Gap alcista
                if current_low > prev_close:
                    gap_size = current_low - prev_close
                    gaps.append({
                        'type': 'gap_up',
                        'size': gap_size,
                        'size_pct': gap_size / prev_close,
                        'timestamp': data.index[i]
                    })
                
                # Gap bajista
                elif current_high < prev_close:
                    gap_size = prev_close - current_high
                    gaps.append({
                        'type': 'gap_down',
                        'size': gap_size,
                        'size_pct': gap_size / prev_close,
                        'timestamp': data.index[i]
                    })
            
            # Gaps recientes (últimos 10 días)
            recent_gaps = [g for g in gaps if (datetime.now() - g['timestamp']).days <= 10]
            
            conditions.gap_analysis = {
                'recent_gaps': len(recent_gaps),
                'largest_recent_gap': max([g['size_pct'] for g in recent_gaps]) if recent_gaps else 0,
                'gap_direction': recent_gaps[-1]['type'] if recent_gaps else None
            }
            
            logger.debug(f"Gaps: {len(recent_gaps)} recientes")
            
        except Exception as e:
            logger.warning(f"Error analizando gaps: {e}")
    
    def _analyze_correlations(self, data: pd.DataFrame, 
                            additional_data: Dict[str, pd.DataFrame], 
                            conditions: MarketConditions):
        """Analizar correlaciones con otros activos"""
        try:
            correlations = {}
            
            for symbol, symbol_data in additional_data.items():
                if len(symbol_data) >= 20:
                    # Alinear datos por fecha
                    aligned_data = pd.concat([
                        data['close'].rename('main'),
                        symbol_data['close'].rename(symbol)
                    ], axis=1).dropna()
                    
                    if len(aligned_data) >= 20:
                        corr = aligned_data['main'].corr(aligned_data[symbol])
                        if not np.isnan(corr):
                            correlations[symbol] = corr
            
            conditions.correlation_with_indices = correlations
            
            logger.debug(f"Correlaciones calculadas: {len(correlations)} símbolos")
            
        except Exception as e:
            logger.warning(f"Error analizando correlaciones: {e}")
    
    def _detect_reversal_pattern(self, data: pd.DataFrame) -> bool:
        """Detectar patrones de reversión"""
        try:
            if len(data) < 10:
                return False
            
            # Patrón simple: divergencia precio vs RSI
            if 'rsi_14' in data.columns:
                price_trend = data['close'].iloc[-5:].is_monotonic_increasing or data['close'].iloc[-5:].is_monotonic_decreasing
                rsi_trend = data['rsi_14'].iloc[-5:].is_monotonic_increasing or data['rsi_14'].iloc[-5:].is_monotonic_decreasing
                
                # Divergencia = tendencias opuestas
                return price_trend != rsi_trend
            
            return False
            
        except Exception as e:
            logger.warning(f"Error detectando reversión: {e}")
            return False
    
    def _calculate_analysis_confidence(self, conditions: MarketConditions) -> float:
        """Calcular confianza general del análisis"""
        try:
            confidence_factors = []
            
            # Factor de tendencia
            if conditions.trend_strength > 0.7:
                confidence_factors.append(0.9)
            elif conditions.trend_strength > 0.5:
                confidence_factors.append(0.7)
            else:
                confidence_factors.append(0.5)
            
            # Factor de volatilidad
            if conditions.volatility_state in [VolatilityState.NORMAL, VolatilityState.LOW]:
                confidence_factors.append(0.8)
            else:
                confidence_factors.append(0.6)
            
            # Factor de indicadores técnicos
            indicators_available = sum([
                1 for x in [conditions.rsi, conditions.macd_signal, conditions.bb_position]
                if x is not None
            ])
            confidence_factors.append(0.5 + (indicators_available / 6))  # Max 0.5 + 0.5 = 1.0
            
            # Factor de estructura
            if conditions.market_structure in ["trending", "ranging"]:
                confidence_factors.append(0.8)
            else:
                confidence_factors.append(0.6)
            
            # Promedio ponderado
            return np.mean(confidence_factors)
            
        except Exception as e:
            logger.warning(f"Error calculando confianza: {e}")
            return 0.5
    
    def _detect_timeframe(self, data: pd.DataFrame) -> str:
        """Detectar timeframe de los datos"""
        if len(data) < 2:
            return "unknown"
        
        try:
            time_diff = data.index[1] - data.index[0]
            minutes = time_diff.total_seconds() / 60
            
            if minutes <= 1:
                return "M1"
            elif minutes <= 5:
                return "M5"
            elif minutes <= 15:
                return "M15"
            elif minutes <= 30:
                return "M30"
            elif minutes <= 60:
                return "H1"
            elif minutes <= 240:
                return "H4"
            elif minutes <= 1440:
                return "D1"
            else:
                return "W1"
                
        except Exception:
            return "unknown"
    
    def _create_default_conditions(self) -> MarketConditions:
        """Crear condiciones por defecto en caso de error"""
        return MarketConditions(
            timestamp=datetime.now(),
            symbol=self.symbol,
            timeframe="unknown",
            market_regime=MarketRegime.UNKNOWN,
            trend_direction=TrendDirection.NEUTRAL,
            trend_strength=0.5,
            volatility_state=VolatilityState.NORMAL,
            current_volatility=0.02,
            volatility_percentile=50.0,
            momentum=0.0,
            momentum_direction="stable",
            market_structure="unknown",
            liquidity_condition="normal",
            analysis_confidence=0.3
        )
    
    def _log_analysis_summary(self, conditions: MarketConditions):
        """Log resumen del análisis"""
        logger.info("="*50)
        logger.info("ANÁLISIS DE MERCADO COMPLETADO")
        logger.info("="*50)
        logger.info(f"Símbolo: {conditions.symbol}")
        logger.info(f"Régimen: {conditions.market_regime.value}")
        logger.info(f"Tendencia: {conditions.trend_direction.value} (fuerza: {conditions.trend_strength:.2f})")
        logger.info(f"Volatilidad: {conditions.volatility_state.value} ({conditions.current_volatility:.4f})")
        logger.info(f"Momentum: {conditions.momentum:.4f} ({conditions.momentum_direction})")
        logger.info(f"Estructura: {conditions.market_structure}")
        logger.info(f"RSI: {conditions.rsi:.1f}" if conditions.rsi else "RSI: N/A")
        logger.info(f"Soportes: {len(conditions.support_levels)}, Resistencias: {len(conditions.resistance_levels)}")
        logger.info(f"Confianza: {conditions.analysis_confidence:.2%}")
        logger.info("="*50)
    
    # ==================== MÉTODOS PÚBLICOS ADICIONALES ====================
    
    def get_current_conditions(self) -> Optional[MarketConditions]:
        """Obtener últimas condiciones analizadas"""
        return self.analysis_history[-1] if self.analysis_history else None
    
    def get_trend_summary(self) -> Dict[str, Any]:
        """Obtener resumen de tendencia"""
        current = self.get_current_conditions()
        if not current:
            return {}
        
        return {
            'direction': current.trend_direction.value,
            'strength': current.trend_strength,
            'momentum': current.momentum,
            'regime': current.market_regime.value
        }
    
    def get_key_levels(self) -> Dict[str, List[float]]:
        """Obtener niveles clave de soporte y resistencia"""
        current = self.get_current_conditions()
        if not current:
            return {'support': [], 'resistance': []}
        
        return {
            'support': [level.price for level in current.support_levels],
            'resistance': [level.price for level in current.resistance_levels]
        }
    
    def is_favorable_for_trading(self) -> bool:
        """Verificar si las condiciones son favorables para trading"""
        current = self.get_current_conditions()
        if not current:
            return False
        
        # Criterios para condiciones favorables
        favorable_regimes = [
            MarketRegime.BULL_MARKET,
            MarketRegime.BEAR_MARKET,
            MarketRegime.BREAKOUT
        ]
        
        unfavorable_states = [
            VolatilityState.EXTREME
        ]
        
        return (
            current.market_regime in favorable_regimes and
            current.volatility_state not in unfavorable_states and
            current.analysis_confidence > 0.6
        )
    
    def get_analysis_history(self, hours: int = 24) -> List[MarketConditions]:
        """Obtener historial de análisis reciente"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [
            conditions for conditions in self.analysis_history
            if conditions.timestamp >= cutoff_time
        ]
    
    def export_analysis_report(self) -> Dict[str, Any]:
        """Exportar reporte completo de análisis"""
        current = self.get_current_conditions()
        if not current:
            return {}
        
        return {
            'timestamp': datetime.now().isoformat(),
            'current_conditions': current.to_dict(),
            'analysis_history_size': len(self.analysis_history),
            'key_metrics': {
                'trend_strength': current.trend_strength,
                'volatility_percentile': current.volatility_percentile,
                'momentum': current.momentum,
                'analysis_confidence': current.analysis_confidence
            },
            'trading_recommendation': 'favorable' if self.is_favorable_for_trading() else 'unfavorable'
        }


# Funciones de utilidad
def create_market_analyzer(symbol: str = "EURUSD") -> MarketAnalyzer:
    """Crear analizador de mercado con configuración por defecto"""
    return MarketAnalyzer(symbol)


def analyze_market_quickly(data: pd.DataFrame, symbol: str = "EURUSD") -> Dict[str, Any]:
    """Análisis rápido de mercado"""
    analyzer = MarketAnalyzer(symbol)
    conditions = analyzer.analyze(data)
    
    return {
        'regime': conditions.market_regime.value,
        'trend': conditions.trend_direction.value,
        'volatility': conditions.volatility_state.value,
        'momentum': conditions.momentum,
        'confidence': conditions.analysis_confidence,
        'favorable_for_trading': analyzer.is_favorable_for_trading()
    }
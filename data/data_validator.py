'''
8. data/data_validator.py
Ruta: TradingBot_Cuantitative_MT5/data/data_validator.py
Resumen:

Valida exhaustivamente los datos de entrada para asegurar integridad y calidad
Aplica reglas configurables de validación (tipos, rangos, outliers, consistencia temporal)
Genera reportes detallados de calidad y dispara alertas cuando detecta anomalías críticas
Integra completamente con el sistema de logs para registrar errores, advertencias y estadísticas
'''
# data/data_validator.py
import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
import json
from enum import Enum
import warnings

from utils.log_config import get_logger, log_risk_alert

logger = get_logger('data')


class ValidationSeverity(Enum):
    """Severidad de las validaciones"""
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class ValidationRule(Enum):
    """Tipos de reglas de validación"""
    DATA_TYPE = "DATA_TYPE"
    RANGE_CHECK = "RANGE_CHECK"
    NULL_CHECK = "NULL_CHECK"
    DUPLICATE_CHECK = "DUPLICATE_CHECK"
    OUTLIER_CHECK = "OUTLIER_CHECK"
    CONSISTENCY_CHECK = "CONSISTENCY_CHECK"
    TEMPORAL_CHECK = "TEMPORAL_CHECK"
    BUSINESS_LOGIC = "BUSINESS_LOGIC"


@dataclass
class ValidationIssue:
    """Representa un problema de validación encontrado"""
    rule: ValidationRule
    severity: ValidationSeverity
    column: Optional[str]
    message: str
    affected_rows: int
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario"""
        return {
            'rule': self.rule.value,
            'severity': self.severity.value,
            'column': self.column,
            'message': self.message,
            'affected_rows': self.affected_rows,
            'details': self.details,
            'timestamp': self.timestamp.isoformat()
        }


@dataclass
class ValidationConfig:
    """Configuración para el validador de datos"""
    
    # Configuración de tipos de datos
    expected_dtypes: Dict[str, str] = field(default_factory=lambda: {
        'open': 'float64',
        'high': 'float64',
        'low': 'float64',
        'close': 'float64',
        'tick_volume': 'int64'
    })
    
    # Rangos esperados para columnas
    column_ranges: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        'tick_volume': (0, 1e6),  # Volumen no puede ser negativo
        'spread': (0, 100)        # Spread máximo en pips
    })
    
    # Tolerancias para outliers (en desviaciones estándar)
    outlier_thresholds: Dict[str, float] = field(default_factory=lambda: {
        'returns': 4.0,
        'volume_ratio': 3.0,
        'range_pct': 3.5
    })
    
    # Porcentajes máximos permitidos de valores nulos
    max_null_percentage: Dict[str, float] = field(default_factory=lambda: {
        'open': 0.0,
        'high': 0.0,
        'low': 0.0,
        'close': 0.0,
        'tick_volume': 0.0,
        'spread': 0.01,  # 1% permitido para spread
        'returns': 0.001  # 0.1% para retornos
    })
    
    # Configuración temporal
    max_time_gap_minutes: int = 60  # Máximo gap permitido en minutos
    min_data_points: int = 100      # Mínimo número de puntos requeridos
    
    # Configuración de consistencia de precios
    price_consistency_tolerance: float = 0.001  # 0.1%
    
    # Configuración de reportes
    generate_detailed_report: bool = True
    save_failed_data: bool = True
    alert_on_critical: bool = True
    
    # Reglas de negocio específicas para Forex
    forex_business_rules: Dict[str, Any] = field(default_factory=lambda: {
        'min_spread_pips': 0.1,      # Spread mínimo esperado
        'max_spread_pips': 50.0,     # Spread máximo antes de alerta
        'max_daily_range_pct': 0.10, # Máximo rango diario (10%)
        'min_volume': 1,             # Volumen mínimo por barra
        'weekend_data_check': True   # Verificar datos de fin de semana
    })


@dataclass
class ValidationReport:
    """Reporte de validación completo"""
    timestamp: datetime
    data_shape: Tuple[int, int]
    validation_config: ValidationConfig
    issues: List[ValidationIssue]
    statistics: Dict[str, Any]
    overall_quality_score: float
    passed: bool
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir reporte a diccionario"""
        return {
            'timestamp': self.timestamp.isoformat(),
            'data_shape': self.data_shape,
            'issues': [issue.to_dict() for issue in self.issues],
            'statistics': self.statistics,
            'overall_quality_score': self.overall_quality_score,
            'passed': self.passed
        }
    
    def save(self, filepath: Path):
        """Guardar reporte en archivo"""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2, default=str)


class DataValidator:
    """Validador principal de datos de mercado"""
    
    def __init__(self, config: Optional[ValidationConfig] = None):
        self.config = config or ValidationConfig()
        self.validation_history: List[ValidationReport] = []
        self.statistics = {
            'total_validations': 0,
            'total_issues_found': 0,
            'critical_issues': 0,
            'validations_passed': 0,
            'validations_failed': 0
        }
        
        logger.info("DataValidator inicializado")
        logger.info(f"Reglas configuradas: {len(ValidationRule)}")
        logger.info(f"Tipos de datos esperados: {len(self.config.expected_dtypes)}")
    
    def validate(self, df: pd.DataFrame, context: str = "Unknown") -> ValidationReport:
        """
        Validar DataFrame completo
        
        Args:
            df: DataFrame a validar
            context: Contexto de la validación (ej: "training_data", "live_data")
            
        Returns:
            ValidationReport con todos los resultados
        """
        logger.info("="*60)
        logger.info("INICIANDO VALIDACIÓN DE DATOS")
        logger.info("="*60)
        logger.info(f"Contexto: {context}")
        logger.info(f"Datos: {df.shape[0]} filas, {df.shape[1]} columnas")
        logger.info(f"Rango temporal: {df.index[0]} a {df.index[-1]}")
        
        start_time = datetime.now()
        issues: List[ValidationIssue] = []
        
        # 1. Validaciones básicas de estructura
        logger.info("\n1. Validando estructura de datos...")
        issues.extend(self._validate_data_structure(df))
        
        # 2. Validaciones de tipos de datos
        logger.info("\n2. Validando tipos de datos...")
        issues.extend(self._validate_data_types(df))
        
        # 3. Validaciones de valores nulos
        logger.info("\n3. Validando valores nulos...")
        issues.extend(self._validate_null_values(df))
        
        # 4. Validaciones de rangos
        logger.info("\n4. Validando rangos de valores...")
        issues.extend(self._validate_value_ranges(df))
        
        # 5. Validaciones de consistencia de precios
        logger.info("\n5. Validando consistencia de precios...")
        issues.extend(self._validate_price_consistency(df))
        
        # 6. Validaciones temporales
        logger.info("\n6. Validando consistencia temporal...")
        issues.extend(self._validate_temporal_consistency(df))
        
        # 7. Detección de outliers
        logger.info("\n7. Detectando outliers...")
        issues.extend(self._detect_outliers(df))
        
        # 8. Validaciones de duplicados
        logger.info("\n8. Validando duplicados...")
        issues.extend(self._validate_duplicates(df))
        
        # 9. Reglas de negocio específicas
        logger.info("\n9. Aplicando reglas de negocio...")
        issues.extend(self._validate_business_rules(df))
        
        # Calcular estadísticas
        validation_stats = self._calculate_validation_statistics(df, issues)
        
        # Calcular score de calidad
        quality_score = self._calculate_quality_score(issues, len(df))
        
        # Determinar si pasó la validación
        critical_issues = [i for i in issues if i.severity == ValidationSeverity.CRITICAL]
        error_issues = [i for i in issues if i.severity == ValidationSeverity.ERROR]
        passed = len(critical_issues) == 0 and len(error_issues) == 0
        
        # Crear reporte
        report = ValidationReport(
            timestamp=start_time,
            data_shape=df.shape,
            validation_config=self.config,
            issues=issues,
            statistics=validation_stats,
            overall_quality_score=quality_score,
            passed=passed
        )
        
        # Actualizar estadísticas
        self.statistics['total_validations'] += 1
        self.statistics['total_issues_found'] += len(issues)
        self.statistics['critical_issues'] += len(critical_issues)
        
        if passed:
            self.statistics['validations_passed'] += 1
        else:
            self.statistics['validations_failed'] += 1
        
        # Guardar en historial
        self.validation_history.append(report)
        
        # Log de resultados
        self._log_validation_results(report, datetime.now() - start_time)
        
        # Alertas críticas
        if critical_issues and self.config.alert_on_critical:
            self._send_critical_alerts(critical_issues, context)
        
        return report
    
    def _validate_data_structure(self, df: pd.DataFrame) -> List[ValidationIssue]:
        """Validar estructura básica del DataFrame"""
        issues = []
        
        # Verificar que no esté vacío
        if len(df) == 0:
            issues.append(ValidationIssue(
                rule=ValidationRule.DATA_TYPE,
                severity=ValidationSeverity.CRITICAL,
                column=None,
                message="DataFrame está vacío",
                affected_rows=0
            ))
            return issues
        
        # Verificar mínimo número de puntos
        if len(df) < self.config.min_data_points:
            issues.append(ValidationIssue(
                rule=ValidationRule.DATA_TYPE,
                severity=ValidationSeverity.ERROR,
                column=None,
                message=f"Datos insuficientes: {len(df)} < {self.config.min_data_points}",
                affected_rows=len(df)
            ))
        
        # Verificar que tiene índice temporal
        if not isinstance(df.index, pd.DatetimeIndex):
            issues.append(ValidationIssue(
                rule=ValidationRule.DATA_TYPE,
                severity=ValidationSeverity.CRITICAL,
                column="index",
                message="Índice no es de tipo datetime",
                affected_rows=len(df)
            ))
        
        # Verificar columnas requeridas
        required_columns = ['open', 'high', 'low', 'close', 'tick_volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            issues.append(ValidationIssue(
                rule=ValidationRule.DATA_TYPE,
                severity=ValidationSeverity.CRITICAL,
                column=None,
                message=f"Columnas faltantes: {missing_columns}",
                affected_rows=len(df),
                details={'missing_columns': missing_columns}
            ))
        
        return issues
    
    def _validate_data_types(self, df: pd.DataFrame) -> List[ValidationIssue]:
        """Validar tipos de datos"""
        issues = []
        
        for column, expected_dtype in self.config.expected_dtypes.items():
            if column not in df.columns:
                continue
            
            actual_dtype = str(df[column].dtype)
            
            # Verificar tipo exacto o compatible
            compatible_types = {
                'float64': ['float32', 'float64', 'int32', 'int64'],
                'int64': ['int32', 'int64'],
                'object': ['object', 'string']
            }
            
            expected_compatible = compatible_types.get(expected_dtype, [expected_dtype])
            
            if actual_dtype not in expected_compatible:
                # Intentar conversión automática
                try:
                    if expected_dtype.startswith('float'):
                        df[column] = pd.to_numeric(df[column], errors='coerce')
                    elif expected_dtype.startswith('int'):
                        df[column] = pd.to_numeric(df[column], errors='coerce').astype('Int64')
                    
                    logger.info(f"  Conversión exitosa: {column} -> {expected_dtype}")
                    
                except Exception as e:
                    issues.append(ValidationIssue(
                        rule=ValidationRule.DATA_TYPE,
                        severity=ValidationSeverity.ERROR,
                        column=column,
                        message=f"Tipo de dato incorrecto: {actual_dtype} (esperado: {expected_dtype})",
                        affected_rows=len(df),
                        details={'actual_dtype': actual_dtype, 'expected_dtype': expected_dtype, 'error': str(e)}
                    ))
        
        return issues
    
    def _validate_null_values(self, df: pd.DataFrame) -> List[ValidationIssue]:
        """Validar valores nulos"""
        issues = []
        
        for column in df.columns:
            null_count = df[column].isnull().sum()
            null_percentage = null_count / len(df)
            
            max_allowed = self.config.max_null_percentage.get(column, 0.05)  # 5% por defecto
            
            if null_percentage > max_allowed:
                severity = ValidationSeverity.CRITICAL if null_percentage > 0.1 else ValidationSeverity.ERROR
                
                issues.append(ValidationIssue(
                    rule=ValidationRule.NULL_CHECK,
                    severity=severity,
                    column=column,
                    message=f"Demasiados valores nulos: {null_percentage:.2%} > {max_allowed:.2%}",
                    affected_rows=null_count,
                    details={'null_count': null_count, 'null_percentage': null_percentage}
                ))
        
        return issues
    
    def _validate_value_ranges(self, df: pd.DataFrame) -> List[ValidationIssue]:
        """Validar rangos de valores"""
        issues = []
        
        # Rangos configurados
        for column, (min_val, max_val) in self.config.column_ranges.items():
            if column not in df.columns:
                continue
            
            out_of_range = df[(df[column] < min_val) | (df[column] > max_val)]
            
            if len(out_of_range) > 0:
                issues.append(ValidationIssue(
                    rule=ValidationRule.RANGE_CHECK,
                    severity=ValidationSeverity.WARNING,
                    column=column,
                    message=f"Valores fuera de rango [{min_val}, {max_val}]",
                    affected_rows=len(out_of_range),
                    details={'min_val': min_val, 'max_val': max_val, 'out_of_range_count': len(out_of_range)}
                ))
        
        # Validaciones específicas de precios
        if all(col in df.columns for col in ['open', 'high', 'low', 'close']):
            # Verificar que valores sean positivos
            for price_col in ['open', 'high', 'low', 'close']:
                negative_prices = df[df[price_col] <= 0]
                if len(negative_prices) > 0:
                    issues.append(ValidationIssue(
                        rule=ValidationRule.RANGE_CHECK,
                        severity=ValidationSeverity.CRITICAL,
                        column=price_col,
                        message="Precios negativos o cero detectados",
                        affected_rows=len(negative_prices)
                    ))
        
        # Verificar volumen
        if 'tick_volume' in df.columns:
            zero_volume = df[df['tick_volume'] <= 0]
            if len(zero_volume) > 0:
                issues.append(ValidationIssue(
                    rule=ValidationRule.RANGE_CHECK,
                    severity=ValidationSeverity.WARNING,
                    column='tick_volume',
                    message="Volumen cero o negativo detectado",
                    affected_rows=len(zero_volume)
                ))
        
        return issues
    
    def _validate_price_consistency(self, df: pd.DataFrame) -> List[ValidationIssue]:
        """Validar consistencia de precios OHLC"""
        issues = []
        
        if not all(col in df.columns for col in ['open', 'high', 'low', 'close']):
            return issues
        
        # High debe ser >= Low
        invalid_hl = df[df['high'] < df['low']]
        if len(invalid_hl) > 0:
            issues.append(ValidationIssue(
                rule=ValidationRule.CONSISTENCY_CHECK,
                severity=ValidationSeverity.CRITICAL,
                column='high/low',
                message="High < Low detectado",
                affected_rows=len(invalid_hl)
            ))
        
        # Open y Close deben estar entre High y Low
        invalid_open = df[(df['open'] > df['high']) | (df['open'] < df['low'])]
        if len(invalid_open) > 0:
            issues.append(ValidationIssue(
                rule=ValidationRule.CONSISTENCY_CHECK,
                severity=ValidationSeverity.CRITICAL,
                column='open',
                message="Open fuera del rango High-Low",
                affected_rows=len(invalid_open)
            ))
        
        invalid_close = df[(df['close'] > df['high']) | (df['close'] < df['low'])]
        if len(invalid_close) > 0:
            issues.append(ValidationIssue(
                rule=ValidationRule.CONSISTENCY_CHECK,
                severity=ValidationSeverity.CRITICAL,
                column='close',
                message="Close fuera del rango High-Low",
                affected_rows=len(invalid_close)
            ))
        
        # Verificar cambios de precio extremos
        if 'close' in df.columns and len(df) > 1:
            price_changes = df['close'].pct_change().abs()
            extreme_changes = price_changes[price_changes > 0.1]  # 10% cambio
            
            if len(extreme_changes) > 0:
                issues.append(ValidationIssue(
                    rule=ValidationRule.CONSISTENCY_CHECK,
                    severity=ValidationSeverity.WARNING,
                    column='close',
                    message=f"Cambios de precio extremos detectados (>10%)",
                    affected_rows=len(extreme_changes),
                    details={'max_change': extreme_changes.max()}
                ))
        
        return issues
    
    def _validate_temporal_consistency(self, df: pd.DataFrame) -> List[ValidationIssue]:
        """Validar consistencia temporal"""
        issues = []
        
        if not isinstance(df.index, pd.DatetimeIndex):
            return issues
        
        # Verificar orden temporal
        if not df.index.is_monotonic_increasing:
            issues.append(ValidationIssue(
                rule=ValidationRule.TEMPORAL_CHECK,
                severity=ValidationSeverity.ERROR,
                column='index',
                message="Índice temporal no está ordenado ascendentemente",
                affected_rows=len(df)
            ))
        
        # Verificar gaps temporales
        time_diffs = df.index.to_series().diff()
        
        # Calcular gap esperado (asumiendo timeframe regular)
        median_diff = time_diffs.median()
        max_expected_gap = timedelta(minutes=self.config.max_time_gap_minutes)
        
        large_gaps = time_diffs[time_diffs > max_expected_gap]
        
        if len(large_gaps) > 0:
            issues.append(ValidationIssue(
                rule=ValidationRule.TEMPORAL_CHECK,
                severity=ValidationSeverity.WARNING,
                column='index',
                message=f"Gaps temporales grandes detectados: {len(large_gaps)}",
                affected_rows=len(large_gaps),
                details={'largest_gap': str(large_gaps.max()), 'gap_count': len(large_gaps)}
            ))
        
        # Verificar datos de fin de semana en Forex
        if self.config.forex_business_rules['weekend_data_check']:
            weekend_data = df[df.index.weekday >= 5]  # Sábado y Domingo
            
            if len(weekend_data) > 0:
                issues.append(ValidationIssue(
                    rule=ValidationRule.BUSINESS_LOGIC,
                    severity=ValidationSeverity.WARNING,
                    column='index',
                    message="Datos de fin de semana detectados en Forex",
                    affected_rows=len(weekend_data)
                ))
        
        return issues
    
    def _detect_outliers(self, df: pd.DataFrame) -> List[ValidationIssue]:
        """Detectar outliers estadísticos"""
        issues = []
        
        # Calcular retornos si no existen
        if 'returns' not in df.columns and 'close' in df.columns:
            df['returns'] = df['close'].pct_change()
        
        # Detectar outliers usando Z-score
        for column, threshold in self.config.outlier_thresholds.items():
            if column not in df.columns:
                continue
            
            # Calcular Z-scores
            column_data = df[column].dropna()
            if len(column_data) < 10:  # Datos insuficientes
                continue
            
            z_scores = np.abs(stats.zscore(column_data))
            outliers = column_data[z_scores > threshold]
            
            if len(outliers) > 0:
                outlier_percentage = len(outliers) / len(column_data)
                severity = ValidationSeverity.WARNING
                
                if outlier_percentage > 0.05:  # Más del 5%
                    severity = ValidationSeverity.ERROR
                
                issues.append(ValidationIssue(
                    rule=ValidationRule.OUTLIER_CHECK,
                    severity=severity,
                    column=column,
                    message=f"Outliers detectados: {len(outliers)} ({outlier_percentage:.2%})",
                    affected_rows=len(outliers),
                    details={
                        'outlier_count': len(outliers),
                        'outlier_percentage': outlier_percentage,
                        'threshold': threshold,
                        'max_z_score': z_scores.max()
                    }
                ))
        
        # Outliers específicos para precios usando IQR
        price_columns = ['open', 'high', 'low', 'close']
        for col in price_columns:
            if col not in df.columns:
                continue
            
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 3 * IQR
            upper_bound = Q3 + 3 * IQR
            
            price_outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
            
            if len(price_outliers) > 0:
                issues.append(ValidationIssue(
                    rule=ValidationRule.OUTLIER_CHECK,
                    severity=ValidationSeverity.WARNING,
                    column=col,
                    message=f"Outliers de precio detectados (método IQR): {len(price_outliers)}",
                    affected_rows=len(price_outliers),
                    details={'bounds': [lower_bound, upper_bound]}
                ))
        
        return issues
    
    def _validate_duplicates(self, df: pd.DataFrame) -> List[ValidationIssue]:
        """Validar duplicados temporales"""
        issues = []
        
        # Verificar duplicados en el índice temporal
        duplicate_times = df.index.duplicated()
        if duplicate_times.any():
            issues.append(ValidationIssue(
                rule=ValidationRule.DUPLICATE_CHECK,
                severity=ValidationSeverity.ERROR,
                column='index',
                message=f"Timestamps duplicados detectados: {duplicate_times.sum()}",
                affected_rows=duplicate_times.sum()
            ))
        
        # Verificar filas completamente duplicadas
        duplicate_rows = df.duplicated()
        if duplicate_rows.any():
            issues.append(ValidationIssue(
                rule=ValidationRule.DUPLICATE_CHECK,
                severity=ValidationSeverity.WARNING,
                column=None,
                message=f"Filas completamente duplicadas: {duplicate_rows.sum()}",
                affected_rows=duplicate_rows.sum()
            ))
        
        return issues
    
    def _validate_business_rules(self, df: pd.DataFrame) -> List[ValidationIssue]:
        """Aplicar reglas de negocio específicas de Forex"""
        issues = []
        
        # Calcular spread si no existe
        if 'spread' not in df.columns and 'ask' in df.columns and 'bid' in df.columns:
            df['spread'] = df['ask'] - df['bid']
        elif 'spread' not in df.columns:
            # Estimar spread basado en volatilidad
            if 'high' in df.columns and 'low' in df.columns:
                df['spread'] = (df['high'] - df['low']) * 0.1  # Estimación conservadora
        
        # Validar spread
        if 'spread' in df.columns:
            min_spread = self.config.forex_business_rules['min_spread_pips'] * 0.0001
            max_spread = self.config.forex_business_rules['max_spread_pips'] * 0.0001
            
            # Spreads demasiado pequeños
            small_spreads = df[df['spread'] < min_spread]
            if len(small_spreads) > 0:
                issues.append(ValidationIssue(
                    rule=ValidationRule.BUSINESS_LOGIC,
                    severity=ValidationSeverity.WARNING,
                    column='spread',
                    message=f"Spreads muy pequeños detectados: {len(small_spreads)}",
                    affected_rows=len(small_spreads)
                ))
            
            # Spreads demasiado grandes
            large_spreads = df[df['spread'] > max_spread]
            if len(large_spreads) > 0:
                issues.append(ValidationIssue(
                    rule=ValidationRule.BUSINESS_LOGIC,
                    severity=ValidationSeverity.ERROR,
                    column='spread',
                    message=f"Spreads excesivos detectados: {len(large_spreads)}",
                    affected_rows=len(large_spreads)
                ))
        
        # Validar rango diario
        if all(col in df.columns for col in ['high', 'low', 'close']):
            daily_range = (df['high'] - df['low']) / df['close']
            max_range = self.config.forex_business_rules['max_daily_range_pct']
            
            excessive_ranges = df[daily_range > max_range]
            if len(excessive_ranges) > 0:
                issues.append(ValidationIssue(
                    rule=ValidationRule.BUSINESS_LOGIC,
                    severity=ValidationSeverity.WARNING,
                    column='high/low',
                    message=f"Rangos diarios excesivos: {len(excessive_ranges)}",
                    affected_rows=len(excessive_ranges)
                ))
        
        return issues
    
    def _calculate_validation_statistics(self, df: pd.DataFrame, 
                                       issues: List[ValidationIssue]) -> Dict[str, Any]:
        """Calcular estadísticas detalladas de validación"""
        stats = {
            'data_shape': df.shape,
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2,
            'total_issues': len(issues),
            'issues_by_severity': {},
            'issues_by_rule': {},
            'affected_rows_total': 0,
            'data_completeness': {}
        }
        
        # Contar por severidad
        for severity in ValidationSeverity:
            count = len([i for i in issues if i.severity == severity])
            stats['issues_by_severity'][severity.value] = count
        
        # Contar por regla
        for rule in ValidationRule:
            count = len([i for i in issues if i.rule == rule])
            stats['issues_by_rule'][rule.value] = count
        
        # Total de filas afectadas
        stats['affected_rows_total'] = sum(issue.affected_rows for issue in issues)
        
        # Completitud de datos
        for column in df.columns:
            completeness = (len(df) - df[column].isnull().sum()) / len(df)
            stats['data_completeness'][column] = completeness
        
        # Estadísticas básicas
        stats['date_range'] = {
            'start': df.index[0].isoformat() if len(df) > 0 else None,
            'end': df.index[-1].isoformat() if len(df) > 0 else None,
            'days': (df.index[-1] - df.index[0]).days if len(df) > 0 else 0
        }
        
        return stats
    
    def _calculate_quality_score(self, issues: List[ValidationIssue], 
                               total_rows: int) -> float:
        """Calcular score de calidad (0-100)"""
        if total_rows == 0:
            return 0.0
        
        # Peso por severidad
        severity_weights = {
            ValidationSeverity.INFO: 0,
            ValidationSeverity.WARNING: 1,
            ValidationSeverity.ERROR: 5,
            ValidationSeverity.CRITICAL: 20
        }
        
        # Calcular penalización
        total_penalty = 0
        for issue in issues:
            weight = severity_weights[issue.severity]
            # Penalización proporcional a filas afectadas
            penalty = weight * (issue.affected_rows / total_rows)
            total_penalty += penalty
        
        # Score base 100, restar penalizaciones
        quality_score = max(0, 100 - total_penalty * 10)
        
        return round(quality_score, 2)
    
    def _log_validation_results(self, report: ValidationReport, duration: timedelta):
        """Log detallado de resultados"""
        logger.info("\n" + "="*60)
        logger.info("RESULTADOS DE VALIDACIÓN")
        logger.info("="*60)
        logger.info(f"Duración: {duration.total_seconds():.2f} segundos")
        logger.info(f"Datos procesados: {report.data_shape[0]:,} filas")
        logger.info(f"Score de calidad: {report.overall_quality_score:.1f}/100")
        logger.info(f"Estado: {'✅ PASÓ' if report.passed else '❌ FALLÓ'}")
        
        if report.issues:
            logger.info(f"\nProblemas encontrados: {len(report.issues)}")
            
            # Agrupar por severidad
            by_severity = {}
            for issue in report.issues:
                severity = issue.severity.value
                if severity not in by_severity:
                    by_severity[severity] = []
                by_severity[severity].append(issue)
            
            for severity, issues in by_severity.items():
                logger.info(f"\n{severity} ({len(issues)}):")
                for issue in issues[:5]:  # Mostrar máximo 5 por categoría
                    logger.info(f"  - {issue.column or 'General'}: {issue.message}")
                if len(issues) > 5:
                    logger.info(f"  ... y {len(issues) - 5} más")
        else:
            logger.info("✅ No se encontraron problemas")
        
        logger.info("="*60)
    
    def _send_critical_alerts(self, critical_issues: List[ValidationIssue], context: str):
        """Enviar alertas para problemas críticos"""
        for issue in critical_issues:
            log_risk_alert(
                "VALIDACIÓN DE DATOS CRÍTICA",
                f"Problema crítico en {context}: {issue.message}",
                {
                    'context': context,
                    'column': issue.column,
                    'affected_rows': issue.affected_rows,
                    'rule': issue.rule.value,
                    'details': issue.details
                }
            )
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """Obtener resumen de todas las validaciones"""
        if not self.validation_history:
            return {'message': 'No hay validaciones en el historial'}
        
        recent_validations = self.validation_history[-10:]  # Últimas 10
        
        summary = {
            'statistics': self.statistics,
            'recent_quality_scores': [v.overall_quality_score for v in recent_validations],
            'avg_quality_score': np.mean([v.overall_quality_score for v in recent_validations]),
            'trend': self._calculate_quality_trend()
        }
        
        return summary
    
    def _calculate_quality_trend(self) -> str:
        """Calcular tendencia de calidad"""
        if len(self.validation_history) < 2:
            return "INSUFICIENTE"
        
        recent_scores = [v.overall_quality_score for v in self.validation_history[-5:]]
        
        if len(recent_scores) >= 3:
            # Calcular pendiente
            x = np.arange(len(recent_scores))
            slope = np.polyfit(x, recent_scores, 1)[0]
            
            if slope > 1:
                return "MEJORANDO"
            elif slope < -1:
                return "EMPEORANDO"
            else:
                return "ESTABLE"
        
        return "ESTABLE"
    
    def export_issues_report(self, filepath: Path, include_details: bool = True):
        """Exportar reporte detallado de problemas"""
        if not self.validation_history:
            logger.warning("No hay historial de validaciones para exportar")
            return
        
        # Compilar todos los problemas
        all_issues = []
        for report in self.validation_history:
            for issue in report.issues:
                issue_dict = issue.to_dict()
                issue_dict['validation_timestamp'] = report.timestamp.isoformat()
                issue_dict['data_shape'] = report.data_shape
                all_issues.append(issue_dict)
        
        # Crear DataFrame para análisis
        issues_df = pd.DataFrame(all_issues)
        
        # Guardar en diferentes formatos
        if filepath.suffix == '.csv':
            issues_df.to_csv(filepath, index=False)
        elif filepath.suffix == '.json':
            with open(filepath, 'w') as f:
                json.dump(all_issues, f, indent=2, default=str)
        elif filepath.suffix == '.xlsx':
            with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
                issues_df.to_excel(writer, sheet_name='Issues', index=False)
                
                # Hoja de resumen
                summary_df = pd.DataFrame([self.get_validation_summary()])
                summary_df.to_excel(writer, sheet_name='Summary', index=False)
        
        logger.info(f"Reporte de problemas exportado a: {filepath}")


# Funciones de utilidad
def create_default_validator() -> DataValidator:
    """Crear validador con configuración por defecto"""
    return DataValidator()

def create_strict_validator() -> DataValidator:
    """Crear validador con configuración estricta"""
    config = ValidationConfig()
    config.max_null_percentage = {col: 0.0 for col in config.max_null_percentage.keys()}
    config.outlier_thresholds = {col: 3.0 for col in config.outlier_thresholds.keys()}
    config.alert_on_critical = True
    return DataValidator(config)

def create_lenient_validator() -> DataValidator:
    """Crear validador con configuración permisiva"""
    config = ValidationConfig()
    config.max_null_percentage = {col: 0.1 for col in config.max_null_percentage.keys()}
    config.outlier_thresholds = {col: 5.0 for col in config.outlier_thresholds.keys()}
    config.alert_on_critical = False
    return DataValidator(config)
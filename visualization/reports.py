import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
import xlsxwriter
from jinja2 import Template
import pdfkit
import base64
from io import BytesIO

from analysis.performance_analyzer import PerformanceAnalyzer, PerformanceMetrics
from analysis.market_analyzer import MarketAnalyzer
from visualization.charts import ChartGenerator
from utils.log_config import get_logger

@dataclass
class ReportConfig:
    """Configuración para generación de reportes"""
    report_type: str  # 'daily', 'weekly', 'monthly', 'custom'
    format: str  # 'pdf', 'html', 'excel', 'all'
    include_sections: List[str]  # Secciones a incluir
    date_range: Tuple[datetime, datetime]
    recipients: List[str]  # Emails para distribución
    branding: Dict[str, Any]  # Logo, colores, etc.
    language: str = 'en'
    
class ReportGenerator:
    """Generador principal de reportes"""
    
    def __init__(self, trading_bot):
        self.trading_bot = trading_bot
        self.logger = get_logger(__name__)
        
        # Analizadores
        self.performance_analyzer = PerformanceAnalyzer()
        self.market_analyzer = MarketAnalyzer()
        self.chart_generator = ChartGenerator()
        
        # Templates
        self.template_loader = TemplateLoader()
        
        # Secciones disponibles
        self.report_sections = {
            'executive_summary': self._generate_executive_summary,
            'performance_analysis': self._generate_performance_analysis,
            'risk_analysis': self._generate_risk_analysis,
            'trade_details': self._generate_trade_details,
            'market_analysis': self._generate_market_analysis,
            'ml_performance': self._generate_ml_performance,
            'recommendations': self._generate_recommendations
        }
        
        # Estilos para PDF
        self.pdf_styles = self._initialize_pdf_styles()
        
    def generate_report(self, config: ReportConfig) -> Dict[str, str]:
        """Genera reporte según configuración"""
        
        self.logger.info(f"Generando reporte {config.report_type}")
        
        # Recopilar datos
        report_data = self._collect_report_data(config)
        
        # Generar secciones
        sections = {}
        for section_name in config.include_sections:
            if section_name in self.report_sections:
                sections[section_name] = self.report_sections[section_name](
                    report_data, config
                )
        
        # Generar reportes en formatos solicitados
        generated_reports = {}
        
        if config.format in ['pdf', 'all']:
            pdf_path = self._generate_pdf_report(sections, report_data, config)
            generated_reports['pdf'] = pdf_path
        
        if config.format in ['html', 'all']:
            html_path = self._generate_html_report(sections, report_data, config)
            generated_reports['html'] = html_path
        
        if config.format in ['excel', 'all']:
            excel_path = self._generate_excel_report(report_data, config)
            generated_reports['excel'] = excel_path
        
        # Distribuir si hay destinatarios
        if config.recipients:
            self._distribute_reports(generated_reports, config)
        
        return generated_reports
    
    def _collect_report_data(self, config: ReportConfig) -> Dict[str, Any]:
        """Recopila todos los datos necesarios para el reporte"""
        
        start_date, end_date = config.date_range
        
        # Performance metrics
        performance_metrics = self.performance_analyzer.analyze_performance(
            self.trading_bot.get_trades(start_date, end_date),
            self.trading_bot.get_equity_curve(start_date, end_date)
        )
        
        # Risk metrics
        risk_metrics = self.trading_bot.risk_manager.get_risk_metrics(
            start_date, end_date
        )
        
        # Trade data
        trades = self.trading_bot.get_trades(start_date, end_date)
        
        # Market conditions
        market_data = self.market_analyzer.get_market_summary(end_date)
        
        # ML model performance
        ml_performance = self.trading_bot.model_hub.get_performance_summary(
            start_date, end_date
        )
        
        # Account info
        account_info = self.trading_bot.get_account_info()
        
        return {
            'performance_metrics': performance_metrics,
            'risk_metrics': risk_metrics,
            'trades': trades,
            'market_data': market_data,
            'ml_performance': ml_performance,
            'account_info': account_info,
            'date_range': config.date_range
        }
    
    def _generate_executive_summary(self, data: Dict[str, Any], 
                                  config: ReportConfig) -> Dict[str, Any]:
        """Genera resumen ejecutivo"""
        
        perf = data['performance_metrics']
        risk = data['risk_metrics']
        
        summary = {
            'period': f"{config.date_range[0].strftime('%Y-%m-%d')} to "
                     f"{config.date_range[1].strftime('%Y-%m-%d')}",
            'total_return': perf.total_return,
            'sharpe_ratio': perf.sharpe_ratio,
            'max_drawdown': perf.max_drawdown,
            'total_trades': len(data['trades']),
            'win_rate': perf.win_rate,
            'profit_factor': perf.profit_factor,
            'current_exposure': risk.get('current_exposure', 0),
            'account_balance': data['account_info']['balance'],
            'key_highlights': self._generate_key_highlights(data),
            'recommendations': self._generate_summary_recommendations(data)
        }
        
        return summary
    
    def _generate_performance_analysis(self, data: Dict[str, Any],
                                     config: ReportConfig) -> Dict[str, Any]:
        """Genera análisis detallado de performance"""
        
        perf = data['performance_metrics']
        trades = data['trades']
        
        # Análisis por timeframe
        timeframe_analysis = self._analyze_by_timeframe(trades)
        
        # Análisis por estrategia
        strategy_analysis = self._analyze_by_strategy(trades)
        
        # Análisis por símbolo
        symbol_analysis = self._analyze_by_symbol(trades)
        
        # Gráficos de performance
        charts = {
            'equity_curve': self._create_equity_curve_chart(data),
            'monthly_returns': self._create_monthly_returns_chart(data),
            'drawdown': self._create_drawdown_chart(data),
            'return_distribution': self._create_return_distribution_chart(data)
        }
        
        return {
            'metrics': {
                'returns': {
                    'total': perf.total_return,
                    'annualized': perf.annualized_return,
                    'monthly_avg': np.mean(perf.monthly_returns),
                    'monthly_std': np.std(perf.monthly_returns),
                    'best_month': np.max(perf.monthly_returns),
                    'worst_month': np.min(perf.monthly_returns)
                },
                'risk_adjusted': {
                    'sharpe_ratio': perf.sharpe_ratio,
                    'sortino_ratio': perf.sortino_ratio,
                    'calmar_ratio': perf.calmar_ratio,
                    'omega_ratio': perf.omega_ratio
                },
                'drawdown': {
                    'max_drawdown': perf.max_drawdown,
                    'avg_drawdown': self._calculate_avg_drawdown(perf.underwater_curve),
                    'max_duration': perf.max_drawdown_duration,
                    'recovery_time': perf.recovery_time
                }
            },
            'analysis': {
                'by_timeframe': timeframe_analysis,
                'by_strategy': strategy_analysis,
                'by_symbol': symbol_analysis
            },
            'charts': charts
        }
    
    def _generate_pdf_report(self, sections: Dict[str, Any],
                           data: Dict[str, Any],
                           config: ReportConfig) -> str:
        """Genera reporte en formato PDF"""
        
        filename = self._generate_filename('pdf', config)
        doc = SimpleDocTemplate(filename, pagesize=A4)
        
        # Elementos del documento
        elements = []
        
        # Página de título
        elements.extend(self._create_title_page(config))
        elements.append(PageBreak())
        
        # Tabla de contenidos
        elements.extend(self._create_table_of_contents(sections))
        elements.append(PageBreak())
        
        # Executive Summary
        if 'executive_summary' in sections:
            elements.extend(self._format_executive_summary_pdf(
                sections['executive_summary']
            ))
            elements.append(PageBreak())
        
        # Performance Analysis
        if 'performance_analysis' in sections:
            elements.extend(self._format_performance_analysis_pdf(
                sections['performance_analysis']
            ))
            elements.append(PageBreak())
        
        # Risk Analysis
        if 'risk_analysis' in sections:
            elements.extend(self._format_risk_analysis_pdf(
                sections['risk_analysis']
            ))
            elements.append(PageBreak())
        
        # Trade Details
        if 'trade_details' in sections:
            elements.extend(self._format_trade_details_pdf(
                sections['trade_details']
            ))
            elements.append(PageBreak())
        
        # Generar PDF
        doc.build(elements)
        
        return filename
    
    def _generate_html_report(self, sections: Dict[str, Any],
                            data: Dict[str, Any],
                            config: ReportConfig) -> str:
        """Genera reporte en formato HTML"""
        
        # Cargar template
        template = self.template_loader.get_template('report_template.html')
        
        # Preparar contexto
        context = {
            'title': f"{config.report_type.title()} Trading Report",
            'date_generated': datetime.now().strftime('%Y-%m-%d %H:%M'),
            'period': f"{config.date_range[0].strftime('%Y-%m-%d')} - "
                     f"{config.date_range[1].strftime('%Y-%m-%d')}",
            'sections': sections,
            'branding': config.branding,
            'charts': self._prepare_charts_for_html(sections)
        }
        
        # Renderizar HTML
        html_content = template.render(**context)
        
        # Guardar archivo
        filename = self._generate_filename('html', config)
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        # Opcionalmente convertir a PDF usando wkhtmltopdf
        if config.format == 'all':
            pdf_filename = filename.replace('.html', '_from_html.pdf')
            pdfkit.from_file(filename, pdf_filename)
        
        return filename
    
    def _generate_excel_report(self, data: Dict[str, Any],
                             config: ReportConfig) -> str:
        """Genera reporte en formato Excel"""
        
        filename = self._generate_filename('xlsx', config)
        
        # Crear workbook
        with pd.ExcelWriter(filename, engine='xlsxwriter') as writer:
            workbook = writer.book
            
            # Formatos
            header_format = workbook.add_format({
                'bold': True,
                'bg_color': '#366092',
                'font_color': 'white',
                'border': 1
            })
            
            number_format = workbook.add_format({'num_format': '#,##0.00'})
            percent_format = workbook.add_format({'num_format': '0.00%'})
            
            # Executive Summary
            summary_df = self._create_summary_dataframe(data)
            summary_df.to_excel(writer, sheet_name='Executive Summary', index=False)
            
            # Performance Metrics
            metrics_df = self._create_metrics_dataframe(data['performance_metrics'])
            metrics_df.to_excel(writer, sheet_name='Performance Metrics', index=False)
            
            # Trade Details
            trades_df = pd.DataFrame(data['trades'])
            if not trades_df.empty:
                trades_df.to_excel(writer, sheet_name='Trade Details', index=False)
                
                # Aplicar formatos
                worksheet = writer.sheets['Trade Details']
                for idx, col in enumerate(trades_df.columns):
                    worksheet.write(0, idx, col, header_format)
            
            # Risk Analysis
            risk_df = self._create_risk_dataframe(data['risk_metrics'])
            risk_df.to_excel(writer, sheet_name='Risk Analysis', index=False)
            
            # Monthly Returns
            if 'performance_metrics' in data:
                monthly_returns = data['performance_metrics'].monthly_returns
                monthly_df = pd.DataFrame({
                    'Month': monthly_returns.index,
                    'Return': monthly_returns.values
                })
                monthly_df.to_excel(writer, sheet_name='Monthly Returns', index=False)
            
            # Charts como imágenes
            self._add_charts_to_excel(writer, data)
        
        return filename

class ReportScheduler:
    """Programador de reportes automáticos"""
    
    def __init__(self, report_generator: ReportGenerator):
        self.report_generator = report_generator
        self.scheduled_reports = []
        
    def schedule_report(self, config: ReportConfig, schedule: str):
        """Programa generación automática de reportes"""
        
        scheduled_report = {
            'config': config,
            'schedule': schedule,  # 'daily', 'weekly', 'monthly'
            'next_run': self._calculate_next_run(schedule),
            'enabled': True
        }
        
        self.scheduled_reports.append(scheduled_report)
        
    def run_scheduled_reports(self):
        """Ejecuta reportes programados"""
        
        current_time = datetime.now()
        
        for report in self.scheduled_reports:
            if report['enabled'] and current_time >= report['next_run']:
                try:
                    # Actualizar período del reporte
                    config = report['config']
                    config.date_range = self._get_report_period(report['schedule'])
                    
                    # Generar reporte
                    self.report_generator.generate_report(config)
                    
                    # Actualizar próxima ejecución
                    report['next_run'] = self._calculate_next_run(report['schedule'])
                    
                except Exception as e:
                    self.logger.error(f"Error generando reporte programado: {e}")

class PerformanceReportBuilder:
    """Constructor especializado de reportes de performance"""
    
    def __init__(self):
        self.sections = []
        
    def add_returns_analysis(self, returns_data: pd.Series) -> 'PerformanceReportBuilder':
        """Agrega análisis de retornos"""
        
        analysis = {
            'title': 'Returns Analysis',
            'data': {
                'total_return': (returns_data + 1).prod() - 1,
                'annualized_return': self._annualize_return(returns_data),
                'volatility': returns_data.std() * np.sqrt(252),
                'skewness': returns_data.skew(),
                'kurtosis': returns_data.kurtosis(),
                'var_95': returns_data.quantile(0.05),
                'cvar_95': returns_data[returns_data <= returns_data.quantile(0.05)].mean()
            },
            'charts': {
                'returns_histogram': self._create_returns_histogram(returns_data),
                'qq_plot': self._create_qq_plot(returns_data),
                'rolling_volatility': self._create_rolling_vol_chart(returns_data)
            }
        }
        
        self.sections.append(analysis)
        return self
    
    def add_attribution_analysis(self, trades: pd.DataFrame) -> 'PerformanceReportBuilder':
        """Agrega análisis de atribución"""
        
        # Attribution por estrategia
        strategy_attribution = trades.groupby('strategy').agg({
            'pnl': ['sum', 'mean', 'count'],
            'return': 'mean'
        })
        
        # Attribution por símbolo
        symbol_attribution = trades.groupby('symbol').agg({
            'pnl': ['sum', 'mean', 'count'],
            'return': 'mean'
        })
        
        # Attribution temporal
        trades['hour'] = pd.to_datetime(trades['entry_time']).dt.hour
        hourly_attribution = trades.groupby('hour')['pnl'].agg(['sum', 'mean', 'count'])
        
        analysis = {
            'title': 'Performance Attribution',
            'data': {
                'by_strategy': strategy_attribution,
                'by_symbol': symbol_attribution,
                'by_hour': hourly_attribution
            },
            'charts': {
                'strategy_contribution': self._create_contribution_chart(strategy_attribution),
                'symbol_contribution': self._create_contribution_chart(symbol_attribution),
                'hourly_pattern': self._create_hourly_pattern_chart(hourly_attribution)
            }
        }
        
        self.sections.append(analysis)
        return self
    
    def build(self) -> Dict[str, Any]:
        """Construye el reporte final"""
        return {
            'sections': self.sections,
            'metadata': {
                'generated_at': datetime.now(),
                'n_sections': len(self.sections)
            }
        }

class ReportTemplates:
    """Templates predefinidos para diferentes tipos de reportes"""
    
    def __init__(self):
        self.templates = {
            'daily_summary': self._get_daily_template(),
            'weekly_performance': self._get_weekly_template(),
            'monthly_detailed': self._get_monthly_template(),
            'risk_report': self._get_risk_template(),
            'compliance_report': self._get_compliance_template()
        }
    
    def _get_daily_template(self) -> ReportConfig:
        """Template para reporte diario"""
        
        return ReportConfig(
            report_type='daily',
            format='html',
            include_sections=[
                'executive_summary',
                'performance_analysis',
                'trade_details',
                'risk_analysis'
            ],
            date_range=(
                datetime.now().replace(hour=0, minute=0, second=0),
                datetime.now()
            ),
            recipients=[],
            branding={
                'logo': 'assets/logo.png',
                'primary_color': '#1976d2',
                'secondary_color': '#ff9800'
            }
        )
    
    def _get_monthly_template(self) -> ReportConfig:
        """Template para reporte mensual detallado"""
        
        # Primer día del mes actual
        first_day = datetime.now().replace(day=1, hour=0, minute=0, second=0)
        
        return ReportConfig(
            report_type='monthly',
            format='all',
            include_sections=[
                'executive_summary',
                'performance_analysis',
                'risk_analysis',
                'trade_details',
                'market_analysis',
                'ml_performance',
                'recommendations'
            ],
            date_range=(first_day, datetime.now()),
            recipients=[],
            branding={
                'logo': 'assets/logo.png',
                'primary_color': '#1976d2',
                'secondary_color': '#ff9800'
            }
        )
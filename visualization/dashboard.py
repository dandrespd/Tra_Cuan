import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import asyncio
from concurrent.futures import ThreadPoolExecutor

from core.trading_bot import TradingBot
from analysis.performance_analyzer import PerformanceAnalyzer
from visualization.charts import ChartGenerator
from utils.log_config import get_logger

class TradingDashboard:
    """Dashboard principal del sistema de trading"""
    
    def __init__(self, trading_bot: TradingBot):
        self.trading_bot = trading_bot
        self.logger = get_logger(__name__)
        
        # Generadores de gráficos
        self.chart_generator = ChartGenerator()
        self.performance_analyzer = PerformanceAnalyzer()
        
        # Estado del dashboard
        self.update_interval = 5  # segundos
        self.theme = 'dark'
        
        # Cache de datos
        self.data_cache = DashboardDataCache()
        
        # Configuración de páginas
        self.pages = {
            'overview': self._render_overview_page,
            'positions': self._render_positions_page,
            'performance': self._render_performance_page,
            'market_analysis': self._render_market_analysis_page,
            'risk_management': self._render_risk_page,
            'ml_models': self._render_ml_models_page,
            'settings': self._render_settings_page,
            'logs': self._render_logs_page
        }
        
    def run(self):
        """Ejecuta el dashboard"""
        st.set_page_config(
            page_title="Trading Bot Dashboard",
            page_icon="📈",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Aplicar tema
        self._apply_theme()
        
        # Sidebar para navegación
        self._render_sidebar()
        
        # Contenido principal
        selected_page = st.session_state.get('selected_page', 'overview')
        
        # Renderizar página seleccionada
        if selected_page in self.pages:
            self.pages[selected_page]()
        
        # Auto-refresh
        if st.session_state.get('auto_refresh', True):
            st.experimental_rerun()
    
    def _render_sidebar(self):
        """Renderiza la barra lateral con navegación y controles"""
        
        with st.sidebar:
            # Logo y título
            st.image("assets/logo.png", width=200)
            st.title("Trading Bot Control")
            
            # Estado del sistema
            self._render_system_status()
            
            st.divider()
            
            # Navegación
            st.subheader("Navigation")
            for page_name, page_func in self.pages.items():
                if st.button(
                    page_name.replace('_', ' ').title(),
                    key=f"nav_{page_name}",
                    use_container_width=True
                ):
                    st.session_state.selected_page = page_name
            
            st.divider()
            
            # Controles rápidos
            self._render_quick_controls()
            
            st.divider()
            
            # Configuración
            st.subheader("Settings")
            
            # Auto-refresh
            st.session_state.auto_refresh = st.checkbox(
                "Auto-refresh",
                value=st.session_state.get('auto_refresh', True)
            )
            
            # Intervalo de actualización
            self.update_interval = st.slider(
                "Update interval (seconds)",
                min_value=1,
                max_value=60,
                value=self.update_interval
            )
            
            # Tema
            theme_options = ['dark', 'light', 'custom']
            selected_theme = st.selectbox(
                "Theme",
                options=theme_options,
                index=theme_options.index(self.theme)
            )
            if selected_theme != self.theme:
                self.theme = selected_theme
                st.experimental_rerun()
    
    def _render_system_status(self):
        """Muestra el estado actual del sistema"""
        
        status = self.trading_bot.get_system_status()
        
        # Indicador de estado
        if status['trading_active']:
            st.success("🟢 System Active")
        else:
            st.error("🔴 System Inactive")
        
        # Métricas clave
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(
                "Positions",
                status['open_positions'],
                delta=status.get('positions_change', 0)
            )
        
        with col2:
            st.metric(
                "Daily P&L",
                f"${status['daily_pnl']:,.2f}",
                delta=f"{status['daily_pnl_pct']:.2%}"
            )
        
        # Conexiones
        connections = status.get('connections', {})
        if connections.get('mt5', False):
            st.success("✓ MT5 Connected")
        else:
            st.error("✗ MT5 Disconnected")
    
    def _render_overview_page(self):
        """Página principal con resumen del sistema"""
        
        st.title("Trading Bot Overview")
        
        # Métricas principales en la parte superior
        self._render_key_metrics()
        
        # Layout de 2 columnas
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Gráfico de equity curve
            st.subheader("Equity Curve")
            equity_data = self.data_cache.get_equity_curve()
            if equity_data is not None:
                fig = self._create_equity_chart(equity_data)
                st.plotly_chart(fig, use_container_width=True)
            
            # Posiciones activas
            st.subheader("Active Positions")
            positions = self.trading_bot.get_open_positions()
            if positions:
                self._render_positions_table(positions)
            else:
                st.info("No active positions")
        
        with col2:
            # Distribución de P&L
            st.subheader("P&L Distribution")
            pnl_data = self.data_cache.get_pnl_distribution()
            if pnl_data is not None:
                fig = self._create_pnl_distribution_chart(pnl_data)
                st.plotly_chart(fig, use_container_width=True)
            
            # Actividad reciente
            st.subheader("Recent Activity")
            self._render_recent_activity()
    
    def _render_key_metrics(self):
        """Muestra métricas clave en cards"""
        
        metrics = self.performance_analyzer.get_current_metrics()
        
        # Crear 4 columnas para métricas
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Total Return",
                f"{metrics['total_return']:.2%}",
                delta=f"{metrics['monthly_return']:.2%}",
                help="Total return since inception"
            )
        
        with col2:
            st.metric(
                "Sharpe Ratio",
                f"{metrics['sharpe_ratio']:.2f}",
                delta=f"{metrics['sharpe_change']:.2f}",
                help="Risk-adjusted return metric"
            )
        
        with col3:
            st.metric(
                "Win Rate",
                f"{metrics['win_rate']:.1%}",
                delta=f"{metrics['win_rate_change']:.1%}",
                help="Percentage of profitable trades"
            )
        
        with col4:
            st.metric(
                "Max Drawdown",
                f"{metrics['max_drawdown']:.1%}",
                delta=None,
                help="Maximum peak-to-trough decline"
            )
    
    def _render_positions_page(self):
        """Página detallada de posiciones"""
        
        st.title("Position Management")
        
        # Tabs para diferentes vistas
        tab1, tab2, tab3 = st.tabs(["Active Positions", "Pending Orders", "History"])
        
        with tab1:
            self._render_active_positions_detailed()
        
        with tab2:
            self._render_pending_orders()
        
        with tab3:
            self._render_position_history()
    
    def _render_active_positions_detailed(self):
        """Vista detallada de posiciones activas"""
        
        positions = self.trading_bot.get_open_positions()
        
        if not positions:
            st.info("No active positions")
            return
        
        # Resumen de exposición
        self._render_exposure_summary(positions)
        
        # Tabla interactiva de posiciones
        for position in positions:
            with st.expander(
                f"{position['symbol']} - {position['type']} "
                f"({position['volume']} lots)",
                expanded=True
            ):
                # Layout de 3 columnas
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.write("**Entry Details**")
                    st.write(f"Entry Price: {position['entry_price']}")
                    st.write(f"Entry Time: {position['entry_time']}")
                    st.write(f"Strategy: {position['strategy']}")
                
                with col2:
                    st.write("**Current Status**")
                    current_pnl = position['current_pnl']
                    pnl_color = "green" if current_pnl > 0 else "red"
                    st.markdown(
                        f"P&L: <span style='color:{pnl_color}'>"
                        f"${current_pnl:,.2f} ({position['pnl_pct']:.2%})</span>",
                        unsafe_allow_html=True
                    )
                    st.write(f"Current Price: {position['current_price']}")
                
                with col3:
                    st.write("**Risk Management**")
                    st.write(f"Stop Loss: {position['stop_loss']}")
                    st.write(f"Take Profit: {position['take_profit']}")
                    st.write(f"Risk Amount: ${position['risk_amount']:,.2f}")
                
                # Botones de acción
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    if st.button(f"Close", key=f"close_{position['ticket']}"):
                        self._close_position(position['ticket'])
                
                with col2:
                    if st.button(f"Modify SL/TP", key=f"modify_{position['ticket']}"):
                        self._show_modify_dialog(position)
                
                with col3:
                    if st.button(f"Add to Position", key=f"add_{position['ticket']}"):
                        self._show_add_dialog(position)
                
                with col4:
                    if st.button(f"Chart", key=f"chart_{position['ticket']}"):
                        self._show_position_chart(position)
    
    def _render_performance_page(self):
        """Página de análisis de performance"""
        
        st.title("Performance Analysis")
        
        # Selector de período
        period = st.selectbox(
            "Select Period",
            options=['1D', '1W', '1M', '3M', '6M', '1Y', 'YTD', 'All'],
            index=2
        )
        
        # Obtener datos de performance
        perf_data = self.performance_analyzer.get_performance_data(period)
        
        # Tabs para diferentes análisis
        tab1, tab2, tab3, tab4 = st.tabs([
            "Returns", "Risk Metrics", "Trade Analysis", "Attribution"
        ])
        
        with tab1:
            self._render_returns_analysis(perf_data)
        
        with tab2:
            self._render_risk_analysis(perf_data)
        
        with tab3:
            self._render_trade_analysis(perf_data)
        
        with tab4:
            self._render_attribution_analysis(perf_data)
    
    def _render_returns_analysis(self, perf_data: Dict[str, Any]):
        """Análisis de retornos"""
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Gráfico de retornos acumulados
            st.subheader("Cumulative Returns")
            fig = go.Figure()
            
            # Portfolio returns
            fig.add_trace(go.Scatter(
                x=perf_data['dates'],
                y=perf_data['cumulative_returns'],
                mode='lines',
                name='Portfolio',
                line=dict(color='blue', width=2)
            ))
            
            # Benchmark si está disponible
            if 'benchmark_returns' in perf_data:
                fig.add_trace(go.Scatter(
                    x=perf_data['dates'],
                    y=perf_data['benchmark_returns'],
                    mode='lines',
                    name='Benchmark',
                    line=dict(color='gray', width=1, dash='dash')
                ))
            
            fig.update_layout(
                template=self._get_plotly_theme(),
                xaxis_title="Date",
                yaxis_title="Cumulative Return",
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Tabla de retornos por período
            st.subheader("Period Returns")
            returns_table = pd.DataFrame({
                'Period': ['Daily', 'Weekly', 'Monthly', 'Quarterly', 'Yearly'],
                'Return': [
                    f"{perf_data['daily_return']:.2%}",
                    f"{perf_data['weekly_return']:.2%}",
                    f"{perf_data['monthly_return']:.2%}",
                    f"{perf_data['quarterly_return']:.2%}",
                    f"{perf_data['yearly_return']:.2%}"
                ],
                'Volatility': [
                    f"{perf_data['daily_vol']:.2%}",
                    f"{perf_data['weekly_vol']:.2%}",
                    f"{perf_data['monthly_vol']:.2%}",
                    f"{perf_data['quarterly_vol']:.2%}",
                    f"{perf_data['yearly_vol']:.2%}"
                ]
            })
            st.dataframe(returns_table, use_container_width=True)
        
        # Heatmap de retornos mensuales
        st.subheader("Monthly Returns Heatmap")
        monthly_returns_pivot = perf_data['monthly_returns_pivot']
        
        fig = go.Figure(data=go.Heatmap(
            z=monthly_returns_pivot.values,
            x=monthly_returns_pivot.columns,
            y=monthly_returns_pivot.index,
            colorscale='RdYlGn',
            zmid=0,
            text=monthly_returns_pivot.values,
            texttemplate='%{text:.1%}',
            textfont={"size": 10}
        ))
        
        fig.update_layout(
            template=self._get_plotly_theme(),
            xaxis_title="Month",
            yaxis_title="Year"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _render_ml_models_page(self):
        """Página de gestión de modelos ML"""
        
        st.title("Machine Learning Models")
        
        # Obtener información de modelos
        model_hub = self.trading_bot.model_hub
        active_models = model_hub.get_active_models()
        
        # Resumen de modelos
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Active Models", len(active_models))
        
        with col2:
            avg_accuracy = np.mean([m.get_accuracy() for m in active_models])
            st.metric("Avg Accuracy", f"{avg_accuracy:.1%}")
        
        with col3:
            total_predictions = sum([m.prediction_count for m in active_models])
            st.metric("Total Predictions", f"{total_predictions:,}")
        
        # Tabs para diferentes vistas
        tab1, tab2, tab3, tab4 = st.tabs([
            "Model Performance", "Feature Importance", 
            "Predictions Monitor", "Model Management"
        ])
        
        with tab1:
            self._render_model_performance()
        
        with tab2:
            self._render_feature_importance()
        
        with tab3:
            self._render_predictions_monitor()
        
        with tab4:
            self._render_model_management()
    
    def _render_model_performance(self):
        """Muestra performance de modelos"""
        
        model_hub = self.trading_bot.model_hub
        models = model_hub.get_all_models()
        
        # Tabla comparativa
        model_data = []
        for model in models:
            metrics = model.get_performance_metrics()
            model_data.append({
                'Model': model.name,
                'Type': model.model_type,
                'Accuracy': f"{metrics['accuracy']:.2%}",
                'Precision': f"{metrics['precision']:.2%}",
                'Recall': f"{metrics['recall']:.2%}",
                'F1 Score': f"{metrics['f1_score']:.2%}",
                'Active': '✓' if model.is_active else '✗',
                'Last Updated': model.last_update
            })
        
        df = pd.DataFrame(model_data)
        st.dataframe(
            df,
            use_container_width=True,
            column_config={
                "Active": st.column_config.TextColumn(
                    "Active",
                    help="Is the model currently active?"
                )
            }
        )
        
        # Gráfico de evolución de accuracy
        st.subheader("Model Accuracy Evolution")
        
        fig = go.Figure()
        
        for model in models[:5]:  # Top 5 modelos
            history = model.get_performance_history()
            fig.add_trace(go.Scatter(
                x=history['dates'],
                y=history['accuracy'],
                mode='lines',
                name=model.name
            ))
        
        fig.update_layout(
            template=self._get_plotly_theme(),
            xaxis_title="Date",
            yaxis_title="Accuracy",
            yaxis_tickformat='.0%'
        )
        
        st.plotly_chart(fig, use_container_width=True)

class DashboardDataCache:
    """Cache para datos del dashboard"""
    
    def __init__(self, ttl_seconds: int = 60):
        self.cache = {}
        self.ttl = ttl_seconds
        self.executor = ThreadPoolExecutor(max_workers=5)
        
    def get_equity_curve(self) -> Optional[pd.DataFrame]:
        """Obtiene equity curve con cache"""
        return self._get_cached_data('equity_curve', self._fetch_equity_curve)
    
    def _get_cached_data(self, key: str, fetch_func):
        """Obtiene datos del cache o los actualiza si es necesario"""
        if key in self.cache:
            data, timestamp = self.cache[key]
            if datetime.now() - timestamp < timedelta(seconds=self.ttl):
                return data
        
        # Actualizar datos
        data = fetch_func()
        self.cache[key] = (data, datetime.now())
        return data

class RealTimeUpdater:
    """Actualizador en tiempo real para el dashboard"""
    
    def __init__(self, dashboard: TradingDashboard):
        self.dashboard = dashboard
        self.update_callbacks = {
            'positions': self._update_positions,
            'performance': self._update_performance,
            'market': self._update_market_data,
            'logs': self._update_logs
        }
        
    async def start_updates(self):
        """Inicia actualizaciones en tiempo real"""
        while True:
            try:
                # Actualizar todas las secciones
                for section, callback in self.update_callbacks.items():
                    await callback()
                
                # Esperar intervalo
                await asyncio.sleep(self.dashboard.update_interval)
                
            except Exception as e:
                self.dashboard.logger.error(f"Error en actualización: {e}")

class ChartCustomizer:
    """Personalizador de gráficos"""
    
    def __init__(self):
        self.themes = {
            'dark': {
                'template': 'plotly_dark',
                'colors': ['#00ff41', '#ff0080', '#00d4ff', '#ffff00'],
                'background': '#0e1117',
                'grid_color': '#1e1e1e'
            },
            'light': {
                'template': 'plotly_white',
                'colors': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'],
                'background': '#ffffff',
                'grid_color': '#e0e0e0'
            }
        }
    
    def apply_theme(self, fig: go.Figure, theme: str = 'dark'):
        """Aplica tema a un gráfico"""
        theme_config = self.themes.get(theme, self.themes['dark'])
        
        fig.update_layout(
            template=theme_config['template'],
            plot_bgcolor=theme_config['background'],
            paper_bgcolor=theme_config['background'],
            colorway=theme_config['colors']
        )
        
        return fig
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import talib

from utils.log_config import get_logger
from analysis.market_analyzer import MarketConditions
from strategies.base_strategy import TradingSignal

class ChartGenerator:
    """Generador principal de gráficos para trading"""
    
    def __init__(self, theme: str = 'dark'):
        self.logger = get_logger(__name__)
        self.theme = theme
        
        # Configuración de colores por tema
        self.color_schemes = {
            'dark': {
                'background': '#0e1117',
                'grid': '#1e1e1e',
                'text': '#fafafa',
                'up_candle': '#26a69a',
                'down_candle': '#ef5350',
                'volume': '#1976d2',
                'sma': '#ffa726',
                'ema': '#ab47bc',
                'bb_fill': 'rgba(33, 150, 243, 0.1)',
                'support': '#4caf50',
                'resistance': '#f44336'
            },
            'light': {
                'background': '#ffffff',
                'grid': '#e0e0e0',
                'text': '#000000',
                'up_candle': '#26a69a',
                'down_candle': '#ef5350',
                'volume': '#1976d2',
                'sma': '#ff9800',
                'ema': '#9c27b0',
                'bb_fill': 'rgba(33, 150, 243, 0.1)',
                'support': '#4caf50',
                'resistance': '#f44336'
            }
        }
        
        # Layouts predefinidos
        self.layouts = {
            'minimal': self._get_minimal_layout,
            'detailed': self._get_detailed_layout,
            'multi_timeframe': self._get_multi_timeframe_layout
        }
        
    def create_candlestick_chart(self, data: pd.DataFrame,
                                indicators: Optional[List[str]] = None,
                                signals: Optional[List[TradingSignal]] = None,
                                annotations: Optional[List[Dict]] = None,
                                layout_type: str = 'detailed') -> go.Figure:
        """Crea gráfico de velas con indicadores y señales"""
        
        # Validar datos
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in data.columns for col in required_columns):
            raise ValueError(f"Data must contain columns: {required_columns}")
        
        # Crear subplots según layout
        if layout_type == 'minimal':
            fig = make_subplots(rows=1, cols=1)
            row_heights = None
        else:
            # Layout con volumen y indicadores adicionales
            n_subplots = 2  # Precio y volumen
            if indicators:
                # Agregar subplot para cada indicador que lo requiera
                oscillators = ['rsi', 'stochastic', 'macd']
                n_oscillators = len([i for i in indicators if i in oscillators])
                n_subplots += n_oscillators
            
            row_heights = [0.6] + [0.2] + [0.2] * (n_subplots - 2)
            
            fig = make_subplots(
                rows=n_subplots,
                cols=1,
                shared_xaxes=True,
                vertical_spacing=0.03,
                row_heights=row_heights,
                subplot_titles=self._get_subplot_titles(indicators)
            )
        
        # Agregar candlestick
        candlestick = go.Candlestick(
            x=data.index,
            open=data['open'],
            high=data['high'],
            low=data['low'],
            close=data['close'],
            name='OHLC',
            increasing_line_color=self.color_schemes[self.theme]['up_candle'],
            decreasing_line_color=self.color_schemes[self.theme]['down_candle'],
            increasing_fillcolor=self.color_schemes[self.theme]['up_candle'],
            decreasing_fillcolor=self.color_schemes[self.theme]['down_candle']
        )
        fig.add_trace(candlestick, row=1, col=1)
        
        # Agregar indicadores
        if indicators:
            self._add_indicators(fig, data, indicators)
        
        # Agregar volumen
        if layout_type != 'minimal':
            self._add_volume(fig, data, row=2)
        
        # Agregar señales de trading
        if signals:
            self._add_trading_signals(fig, signals, data)
        
        # Agregar anotaciones
        if annotations:
            self._add_annotations(fig, annotations)
        
        # Aplicar layout
        layout_func = self.layouts.get(layout_type, self._get_detailed_layout)
        fig.update_layout(layout_func())
        
        # Agregar herramientas interactivas
        self._add_interactive_tools(fig)
        
        return fig
    
    def _add_indicators(self, fig: go.Figure, data: pd.DataFrame, 
                       indicators: List[str]):
        """Agrega indicadores técnicos al gráfico"""
        
        colors = self.color_schemes[self.theme]
        current_row = 1
        
        for indicator in indicators:
            if indicator == 'sma_20' and 'sma_20' in data.columns:
                fig.add_trace(go.Scatter(
                    x=data.index,
                    y=data['sma_20'],
                    mode='lines',
                    name='SMA 20',
                    line=dict(color=colors['sma'], width=1),
                    showlegend=True
                ), row=1, col=1)
            
            elif indicator == 'ema_20' and 'ema_20' in data.columns:
                fig.add_trace(go.Scatter(
                    x=data.index,
                    y=data['ema_20'],
                    mode='lines',
                    name='EMA 20',
                    line=dict(color=colors['ema'], width=1),
                    showlegend=True
                ), row=1, col=1)
            
            elif indicator == 'bollinger_bands':
                if all(col in data.columns for col in ['bb_upper', 'bb_middle', 'bb_lower']):
                    # Banda superior
                    fig.add_trace(go.Scatter(
                        x=data.index,
                        y=data['bb_upper'],
                        mode='lines',
                        name='BB Upper',
                        line=dict(color='rgba(33, 150, 243, 0.5)', width=1),
                        showlegend=False
                    ), row=1, col=1)
                    
                    # Banda media
                    fig.add_trace(go.Scatter(
                        x=data.index,
                        y=data['bb_middle'],
                        mode='lines',
                        name='BB Middle',
                        line=dict(color='rgba(33, 150, 243, 0.8)', width=1),
                        showlegend=True
                    ), row=1, col=1)
                    
                    # Banda inferior
                    fig.add_trace(go.Scatter(
                        x=data.index,
                        y=data['bb_lower'],
                        mode='lines',
                        name='BB Lower',
                        line=dict(color='rgba(33, 150, 243, 0.5)', width=1),
                        fill='tonexty',
                        fillcolor=colors['bb_fill'],
                        showlegend=False
                    ), row=1, col=1)
            
            elif indicator == 'rsi' and 'rsi' in data.columns:
                current_row = 3  # RSI en subplot separado
                
                # RSI
                fig.add_trace(go.Scatter(
                    x=data.index,
                    y=data['rsi'],
                    mode='lines',
                    name='RSI',
                    line=dict(color='#ff9800', width=2)
                ), row=current_row, col=1)
                
                # Niveles de sobrecompra/sobreventa
                fig.add_hline(y=70, line_dash="dash", line_color="red", 
                            annotation_text="Overbought", row=current_row, col=1)
                fig.add_hline(y=30, line_dash="dash", line_color="green", 
                            annotation_text="Oversold", row=current_row, col=1)
            
            elif indicator == 'macd':
                if all(col in data.columns for col in ['macd', 'macd_signal', 'macd_hist']):
                    current_row = 4 if 'rsi' in indicators else 3
                    
                    # MACD line
                    fig.add_trace(go.Scatter(
                        x=data.index,
                        y=data['macd'],
                        mode='lines',
                        name='MACD',
                        line=dict(color='#2196f3', width=2)
                    ), row=current_row, col=1)
                    
                    # Signal line
                    fig.add_trace(go.Scatter(
                        x=data.index,
                        y=data['macd_signal'],
                        mode='lines',
                        name='Signal',
                        line=dict(color='#ff5722', width=2)
                    ), row=current_row, col=1)
                    
                    # Histograma
                    colors = ['green' if val >= 0 else 'red' for val in data['macd_hist']]
                    fig.add_trace(go.Bar(
                        x=data.index,
                        y=data['macd_hist'],
                        name='MACD Hist',
                        marker_color=colors,
                        showlegend=False
                    ), row=current_row, col=1)
    
    def _add_trading_signals(self, fig: go.Figure, signals: List[TradingSignal],
                           data: pd.DataFrame):
        """Agrega señales de trading al gráfico"""
        
        buy_signals = []
        sell_signals = []
        
        for signal in signals:
            # Encontrar el índice más cercano en los datos
            signal_time = signal.timestamp
            closest_idx = data.index.get_indexer([signal_time], method='nearest')[0]
            
            if closest_idx >= 0 and closest_idx < len(data):
                price = data.iloc[closest_idx]['close']
                
                if signal.signal_type.value == 'BUY':
                    buy_signals.append({
                        'x': data.index[closest_idx],
                        'y': price * 0.995,  # Ligeramente debajo del precio
                        'text': f"BUY<br>Conf: {signal.confidence:.2f}",
                        'strategy': signal.strategy_name
                    })
                elif signal.signal_type.value == 'SELL':
                    sell_signals.append({
                        'x': data.index[closest_idx],
                        'y': price * 1.005,  # Ligeramente arriba del precio
                        'text': f"SELL<br>Conf: {signal.confidence:.2f}",
                        'strategy': signal.strategy_name
                    })
        
        # Agregar marcadores de compra
        if buy_signals:
            fig.add_trace(go.Scatter(
                x=[s['x'] for s in buy_signals],
                y=[s['y'] for s in buy_signals],
                mode='markers+text',
                name='Buy Signals',
                marker=dict(
                    symbol='triangle-up',
                    size=12,
                    color='green'
                ),
                text=[s['text'] for s in buy_signals],
                textposition='bottom center',
                showlegend=True
            ), row=1, col=1)
        
        # Agregar marcadores de venta
        if sell_signals:
            fig.add_trace(go.Scatter(
                x=[s['x'] for s in sell_signals],
                y=[s['y'] for s in sell_signals],
                mode='markers+text',
                name='Sell Signals',
                marker=dict(
                    symbol='triangle-down',
                    size=12,
                    color='red'
                ),
                text=[s['text'] for s in sell_signals],
                textposition='top center',
                showlegend=True
            ), row=1, col=1)
    
    def create_performance_chart(self, equity_curve: pd.Series,
                               drawdown: pd.Series,
                               benchmark: Optional[pd.Series] = None) -> go.Figure:
        """Crea gráfico de performance con equity curve y drawdown"""
        
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.7, 0.3],
            subplot_titles=('Equity Curve', 'Drawdown')
        )
        
        # Equity curve
        fig.add_trace(go.Scatter(
            x=equity_curve.index,
            y=equity_curve.values,
            mode='lines',
            name='Portfolio',
            line=dict(color='#2196f3', width=2),
            fill='tozeroy',
            fillcolor='rgba(33, 150, 243, 0.1)'
        ), row=1, col=1)
        
        # Benchmark si está disponible
        if benchmark is not None:
            fig.add_trace(go.Scatter(
                x=benchmark.index,
                y=benchmark.values,
                mode='lines',
                name='Benchmark',
                line=dict(color='gray', width=1, dash='dash')
            ), row=1, col=1)
        
        # Drawdown
        fig.add_trace(go.Scatter(
            x=drawdown.index,
            y=drawdown.values * 100,  # Convertir a porcentaje
            mode='lines',
            name='Drawdown',
            line=dict(color='red', width=1),
            fill='tozeroy',
            fillcolor='rgba(255, 0, 0, 0.1)',
            showlegend=False
        ), row=2, col=1)
        
        # Actualizar layout
        fig.update_layout(
            template='plotly_dark' if self.theme == 'dark' else 'plotly_white',
            height=600,
            showlegend=True,
            hovermode='x unified'
        )
        
        # Formato de ejes
        fig.update_yaxes(title_text="Value", row=1, col=1)
        fig.update_yaxes(title_text="Drawdown %", row=2, col=1)
        fig.update_xaxes(title_text="Date", row=2, col=1)
        
        return fig
    
    def create_multi_timeframe_chart(self, symbol: str,
                                   timeframes: List[str],
                                   data_dict: Dict[str, pd.DataFrame]) -> go.Figure:
        """Crea gráfico multi-timeframe para análisis comprehensivo"""
        
        n_timeframes = len(timeframes)
        
        fig = make_subplots(
            rows=n_timeframes,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.02,
            subplot_titles=[f"{symbol} - {tf}" for tf in timeframes]
        )
        
        for i, timeframe in enumerate(timeframes, 1):
            data = data_dict[timeframe]
            
            # Candlestick para cada timeframe
            fig.add_trace(go.Candlestick(
                x=data.index,
                open=data['open'],
                high=data['high'],
                low=data['low'],
                close=data['close'],
                name=f'OHLC {timeframe}',
                showlegend=False,
                increasing_line_color=self.color_schemes[self.theme]['up_candle'],
                decreasing_line_color=self.color_schemes[self.theme]['down_candle']
            ), row=i, col=1)
            
            # Agregar EMA 20 a cada timeframe
            if 'ema_20' in data.columns:
                fig.add_trace(go.Scatter(
                    x=data.index,
                    y=data['ema_20'],
                    mode='lines',
                    name=f'EMA 20 {timeframe}',
                    line=dict(color=self.color_schemes[self.theme]['ema'], width=1),
                    showlegend=False
                ), row=i, col=1)
        
        # Actualizar layout
        fig.update_layout(
            template='plotly_dark' if self.theme == 'dark' else 'plotly_white',
            height=300 * n_timeframes,
            showlegend=False,
            hovermode='x unified'
        )
        
        # Ocultar rangeslider para todos excepto el último
        for i in range(1, n_timeframes):
            fig.update_xaxes(rangeslider_visible=False, row=i, col=1)
        
        return fig
    
    def create_correlation_matrix(self, correlation_data: pd.DataFrame,
                                title: str = "Asset Correlation Matrix") -> go.Figure:
        """Crea matriz de correlación interactiva"""
        
        # Crear texto para hover
        text = []
        for i in range(len(correlation_data)):
            row_text = []
            for j in range(len(correlation_data.columns)):
                row_text.append(
                    f"{correlation_data.index[i]} vs {correlation_data.columns[j]}<br>"
                    f"Correlation: {correlation_data.iloc[i, j]:.3f}"
                )
            text.append(row_text)
        
        fig = go.Figure(data=go.Heatmap(
            z=correlation_data.values,
            x=correlation_data.columns,
            y=correlation_data.index,
            colorscale='RdBu',
            zmid=0,
            text=correlation_data.values,
            texttemplate='%{text:.2f}',
            textfont={"size": 10},
            hovertext=text,
            hovertemplate='%{hovertext}<extra></extra>',
            colorbar=dict(title="Correlation")
        ))
        
        fig.update_layout(
            title=title,
            template='plotly_dark' if self.theme == 'dark' else 'plotly_white',
            height=600,
            width=800
        )
        
        return fig
    
    def create_market_profile_chart(self, data: pd.DataFrame,
                                  value_area_pct: float = 0.7) -> go.Figure:
        """Crea gráfico de perfil de mercado (Market Profile)"""
        
        # Calcular perfil de volumen por precio
        price_bins = pd.cut(data['close'], bins=50)
        volume_profile = data.groupby(price_bins)['volume'].sum()
        
        # Calcular área de valor (Value Area)
        total_volume = volume_profile.sum()
        target_volume = total_volume * value_area_pct
        
        # Ordenar por volumen y encontrar área de valor
        sorted_profile = volume_profile.sort_values(ascending=False)
        cumsum = 0
        value_area_prices = []
        
        for price_range, vol in sorted_profile.items():
            cumsum += vol
            value_area_prices.append(price_range.mid)
            if cumsum >= target_volume:
                break
        
        # Crear gráfico
        fig = make_subplots(
            rows=1, cols=2,
            column_widths=[0.8, 0.2],
            horizontal_spacing=0.01,
            shared_yaxes=True
        )
        
        # Gráfico de velas
        fig.add_trace(go.Candlestick(
            x=data.index,
            open=data['open'],
            high=data['high'],
            low=data['low'],
            close=data['close'],
            name='OHLC',
            showlegend=False
        ), row=1, col=1)
        
        # Perfil de volumen
        y_values = [p.mid for p in volume_profile.index]
        
        fig.add_trace(go.Bar(
            x=volume_profile.values,
            y=y_values,
            orientation='h',
            name='Volume Profile',
            marker_color='rgba(33, 150, 243, 0.5)',
            showlegend=False
        ), row=1, col=2)
        
        # Marcar área de valor
        value_area_min = min(value_area_prices)
        value_area_max = max(value_area_prices)
        
        fig.add_hrect(
            y0=value_area_min, y1=value_area_max,
            fillcolor="rgba(255, 193, 7, 0.1)",
            annotation_text="Value Area",
            annotation_position="right",
            row=1, col=1
        )
        
        # POC (Point of Control)
        poc_price = volume_profile.idxmax().mid
        fig.add_hline(
            y=poc_price,
            line_dash="dash",
            line_color="red",
            annotation_text="POC",
            row=1, col=1
        )
        
        fig.update_layout(
            template='plotly_dark' if self.theme == 'dark' else 'plotly_white',
            height=600,
            showlegend=False,
            hovermode='y unified'
        )
        
        return fig
    
    def create_heatmap_calendar(self, returns: pd.Series,
                              title: str = "Returns Calendar") -> go.Figure:
        """Crea calendario heatmap de retornos"""
        
        # Preparar datos para calendario
        returns_df = returns.to_frame('returns')
        returns_df['year'] = returns_df.index.year
        returns_df['month'] = returns_df.index.month
        returns_df['day'] = returns_df.index.day
        
        # Pivot para formato calendario
        calendar_data = returns_df.pivot_table(
            values='returns',
            index='year',
            columns='month',
            aggfunc='sum'
        )
        
        # Nombres de meses
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        fig = go.Figure(data=go.Heatmap(
            z=calendar_data.values * 100,  # Convertir a porcentaje
            x=month_names,
            y=calendar_data.index,
            colorscale='RdYlGn',
            zmid=0,
            text=calendar_data.values * 100,
            texttemplate='%{text:.1f}%',
            textfont={"size": 10},
            colorbar=dict(title="Returns %")
        ))
        
        fig.update_layout(
            title=title,
            template='plotly_dark' if self.theme == 'dark' else 'plotly_white',
            height=400,
            xaxis_title="Month",
            yaxis_title="Year"
        )
        
        return fig

class InteractiveChartTools:
    """Herramientas interactivas para gráficos"""
    
    def __init__(self):
        self.drawing_tools = [
            'drawline',
            'drawopenpath',
            'drawclosedpath',
            'drawcircle',
            'drawrect',
            'eraseshape'
        ]
        
    def add_drawing_tools(self, fig: go.Figure) -> go.Figure:
        """Agrega herramientas de dibujo al gráfico"""
        
        fig.update_layout(
            dragmode='drawline',
            newshape=dict(line_color='cyan'),
            modebar_add=self.drawing_tools
        )
        
        return fig
    
    def add_range_selector(self, fig: go.Figure) -> go.Figure:
        """Agrega selector de rango temporal"""
        
        fig.update_xaxes(
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1D", step="day", stepmode="backward"),
                    dict(count=7, label="1W", step="day", stepmode="backward"),
                    dict(count=1, label="1M", step="month", stepmode="backward"),
                    dict(count=3, label="3M", step="month", stepmode="backward"),
                    dict(count=6, label="6M", step="month", stepmode="backward"),
                    dict(count=1, label="YTD", step="year", stepmode="todate"),
                    dict(count=1, label="1Y", step="year", stepmode="backward"),
                    dict(step="all")
                ])
            ),
            rangeslider=dict(visible=True),
            type="date"
        )
        
        return fig

class ChartExporter:
    """Exportador de gráficos en múltiples formatos"""
    
    def __init__(self):
        self.supported_formats = ['png', 'jpg', 'svg', 'pdf', 'html']
        
    def export_chart(self, fig: go.Figure, filename: str, 
                    format: str = 'png', **kwargs) -> bool:
        """Exporta gráfico en el formato especificado"""
        
        if format not in self.supported_formats:
            raise ValueError(f"Format {format} not supported")
        
        try:
            if format == 'html':
                fig.write_html(f"{filename}.html", **kwargs)
            else:
                fig.write_image(f"{filename}.{format}", **kwargs)
            return True
        except Exception as e:
            logger.error(f"Error exporting chart: {e}")
            return False
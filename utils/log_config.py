# utils/log_config.py
import os
import sys
import logging
import logging.handlers
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, List, Tuple, Any
import json
import platform
import shutil
from collections import defaultdict
from enum import Enum

# Manejo robusto de imports opcionales
try:
    import colorama
    from colorama import Fore, Back, Style
    colorama.init(autoreset=True)
    COLORAMA_AVAILABLE = True
except ImportError:
    COLORAMA_AVAILABLE = False

try:
    import rich
    from rich.console import Console
    from rich.table import Table
    from rich.progress import Progress, SpinnerColumn, TextColumn
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


class LogLevel(Enum):
    """Niveles de log personalizados"""
    TRADE_OPEN = 25
    TRADE_CLOSE = 26
    TRADE_WIN = 27
    TRADE_LOSS = 28
    PERFORMANCE = 29
    RISK_ALERT = 35


class TerminalCapabilities:
    """Detectar capacidades del terminal"""
    
    def __init__(self):
        self.os_type = platform.system()
        self.terminal = os.environ.get('TERM', '')
        self.colorterm = os.environ.get('COLORTERM', '')
        self.columns = shutil.get_terminal_size().columns
        self.encoding = sys.stdout.encoding or 'utf-8'
        
        # Detección de capacidades
        self.supports_color = self._detect_color_support()
        self.supports_unicode = self._detect_unicode_support()
        self.supports_emoji = self._detect_emoji_support()
        self.is_notebook = self._detect_notebook()
    
    def _detect_color_support(self) -> bool:
        """Detectar soporte de colores"""
        # Windows con colorama
        if self.os_type == 'Windows' and COLORAMA_AVAILABLE:
            return True
        
        # Windows Terminal o VS Code
        if os.environ.get('WT_SESSION') or os.environ.get('TERM_PROGRAM') == 'vscode':
            return True
        
        # Terminales Unix/Linux
        if 'color' in self.terminal or self.colorterm:
            return True
        
        # CI/CD environments
        if os.environ.get('CI') or os.environ.get('GITHUB_ACTIONS'):
            return False
        
        return self.os_type in ['Linux', 'Darwin']
    
    def _detect_unicode_support(self) -> bool:
        """Detectar soporte Unicode"""
        try:
            # Intentar codificar caracteres Unicode
            test_chars = '│├─└█▄▀●○◆◇'
            test_chars.encode(self.encoding)
            
            # Verificar si es una terminal real
            if hasattr(sys.stdout, 'isatty') and sys.stdout.isatty():
                return True
            
            return self.encoding.lower() in ['utf-8', 'utf8', 'utf-16']
        except:
            return False
    
    def _detect_emoji_support(self) -> bool:
        """Detectar soporte de emojis"""
        # Windows Terminal moderna
        if os.environ.get('WT_SESSION'):
            return True
        
        # macOS generalmente soporta emojis
        if self.os_type == 'Darwin':
            return True
        
        # Linux con terminal moderna
        if self.os_type == 'Linux' and self.supports_unicode:
            return 'xterm' in self.terminal or 'gnome' in self.terminal
        
        # Windows 10 build 1903+
        if self.os_type == 'Windows':
            try:
                import winreg
                key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, 
                                   r"SOFTWARE\Microsoft\Windows NT\CurrentVersion")
                build = int(winreg.QueryValueEx(key, "CurrentBuildNumber")[0])
                return build >= 18362
            except:
                pass
        
        return False
    
    def _detect_notebook(self) -> bool:
        """Detectar si estamos en Jupyter/IPython"""
        try:
            from IPython import get_ipython
            return get_ipython() is not None
        except:
            return False


class SymbolSet:
    """Conjunto de símbolos adaptativo según capacidades"""
    
    def __init__(self, capabilities: TerminalCapabilities):
        self.caps = capabilities
        self._init_symbols()
    
    def _init_symbols(self):
        """Inicializar símbolos según capacidades"""
        if self.caps.supports_emoji:
            self.symbols = {
                # Status
                'success': '✅',
                'error': '❌',
                'warning': '⚠️ ',
                'info': 'ℹ️ ',
                'debug': '🔍',
                'critical': '🔥',
                
                # Trading
                'buy': '📈',
                'sell': '📉',
                'win': '💰',
                'loss': '💸',
                'neutral': '➖',
                
                # Indicators
                'up': '⬆️ ',
                'down': '⬇️ ',
                'chart': '📊',
                'target': '🎯',
                'clock': '⏰',
                'robot': '🤖',
                
                # Decorative
                'star': '⭐',
                'sparkle': '✨',
                'rocket': '🚀',
                'shield': '🛡️ '
            }
        elif self.caps.supports_unicode:
            self.symbols = {
                # Status
                'success': '✓',
                'error': '✗',
                'warning': '⚠',
                'info': 'ⓘ',
                'debug': '◎',
                'critical': '▲',
                
                # Trading
                'buy': '▲',
                'sell': '▼',
                'win': '●',
                'loss': '○',
                'neutral': '─',
                
                # Indicators
                'up': '↑',
                'down': '↓',
                'chart': '▦',
                'target': '◉',
                'clock': '◷',
                'robot': '◆',
                
                # Decorative
                'star': '★',
                'sparkle': '◇',
                'rocket': '►',
                'shield': '◈'
            }
        else:
            self.symbols = {
                # Status
                'success': '[OK]',
                'error': '[ERR]',
                'warning': '[WRN]',
                'info': '[INF]',
                'debug': '[DBG]',
                'critical': '[CRT]',
                
                # Trading
                'buy': '[BUY]',
                'sell': '[SELL]',
                'win': '[WIN]',
                'loss': '[LOSS]',
                'neutral': '[-]',
                
                # Indicators
                'up': '[UP]',
                'down': '[DN]',
                'chart': '[=]',
                'target': '[X]',
                'clock': '[T]',
                'robot': '[BOT]',
                
                # Decorative
                'star': '*',
                'sparkle': '+',
                'rocket': '>',
                'shield': '#'
            }
        
        # Box drawing
        if self.caps.supports_unicode:
            self.box = {
                'tl': '╔', 'tr': '╗', 'bl': '╚', 'br': '╝',
                'h': '═', 'v': '║', 'cross': '╬',
                'vr': '╠', 'vl': '╣', 'ht': '╦', 'hb': '╩',
                'light_h': '─', 'light_v': '│'
            }
        else:
            self.box = {
                'tl': '+', 'tr': '+', 'bl': '+', 'br': '+',
                'h': '=', 'v': '|', 'cross': '+',
                'vr': '+', 'vl': '+', 'ht': '+', 'hb': '+',
                'light_h': '-', 'light_v': '|'
            }


class ColorScheme:
    """Esquema de colores adaptativo"""
    
    def __init__(self, capabilities: TerminalCapabilities):
        self.caps = capabilities
        self.use_color = capabilities.supports_color and COLORAMA_AVAILABLE
    
    def apply(self, text: str, *styles) -> str:
        """Aplicar estilos al texto"""
        if not self.use_color:
            return text
        
        styled_text = text
        for style in styles:
            if hasattr(Fore, style.upper()):
                styled_text = getattr(Fore, style.upper()) + styled_text
            elif hasattr(Style, style.upper()):
                styled_text = getattr(Style, style.upper()) + styled_text
        
        return styled_text + Style.RESET_ALL
    
    def gradient(self, text: str, colors: List[str]) -> str:
        """Aplicar gradiente de colores"""
        if not self.use_color or len(colors) == 0:
            return text
        
        result = ""
        color_count = len(colors)
        text_len = len(text)
        
        for i, char in enumerate(text):
            color_idx = int(i * color_count / text_len)
            color_idx = min(color_idx, color_count - 1)
            result += self.apply(char, colors[color_idx])
        
        return result


class TradingFormatter(logging.Formatter):
    """Formatter personalizado para trading"""
    
    def __init__(self, capabilities: TerminalCapabilities, 
                 symbols: SymbolSet, colors: ColorScheme,
                 show_time: bool = True, show_module: bool = False):
        self.caps = capabilities
        self.symbols = symbols
        self.colors = colors
        self.show_time = show_time
        self.show_module = show_module
        
        # Formato base
        fmt_parts = []
        if show_time:
            fmt_parts.append('%(asctime)s')
        fmt_parts.append('%(symbol)s')
        fmt_parts.append('%(levelname)-8s')
        if show_module:
            fmt_parts.append('[%(name)s]')
        fmt_parts.append('%(message)s')
        
        super().__init__(' '.join(fmt_parts), datefmt='%H:%M:%S')
    
    def format(self, record: logging.LogRecord) -> str:
        # Asignar símbolo según nivel
        level_symbols = {
            'DEBUG': self.symbols.symbols['debug'],
            'INFO': self.symbols.symbols['info'],
            'WARNING': self.symbols.symbols['warning'],
            'ERROR': self.symbols.symbols['error'],
            'CRITICAL': self.symbols.symbols['critical'],
            'TRADE_OPEN': self.symbols.symbols['chart'],
            'TRADE_CLOSE': self.symbols.symbols['target'],
            'TRADE_WIN': self.symbols.symbols['win'],
            'TRADE_LOSS': self.symbols.symbols['loss'],
            'PERFORMANCE': self.symbols.symbols['star'],
            'RISK_ALERT': self.symbols.symbols['shield']
        }
        
        record.symbol = level_symbols.get(record.levelname, '')
        
        # Colorear nivel
        level_colors = {
            'DEBUG': 'cyan',
            'INFO': 'green',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'red',
            'TRADE_WIN': 'green',
            'TRADE_LOSS': 'red',
            'PERFORMANCE': 'blue',
            'RISK_ALERT': 'magenta'
        }
        
        color = level_colors.get(record.levelname, 'white')
        
        # Aplicar formato base
        formatted = super().format(record)
        
        # Aplicar colores si están disponibles
        if self.colors.use_color:
            # Colorear timestamp
            if self.show_time:
                time_part = formatted.split(' ')[0]
                formatted = formatted.replace(time_part, 
                    self.colors.apply(time_part, 'cyan'), 1)
            
            # Colorear nivel
            levelname = record.levelname
            formatted = formatted.replace(levelname,
                self.colors.apply(levelname, color, 'bright'))
        
        return formatted


class TradingLogger:
    """Sistema de logging avanzado para trading"""
    
    def __init__(self, name: str = "TradingBot", base_dir: Path = None):
        self.name = name
        self.base_dir = base_dir or Path.cwd()
        self.session_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Detectar capacidades
        self.capabilities = TerminalCapabilities()
        self.symbols = SymbolSet(self.capabilities)
        self.colors = ColorScheme(self.capabilities)
        
        # Directorios
        self.log_dir = self.base_dir / 'logs' / self.session_id
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Registrar niveles personalizados
        self._register_custom_levels()
        
        # Configurar loggers
        self.loggers: Dict[str, logging.Logger] = {}
        self._setup_loggers()
        
        # Estadísticas
        self.stats = defaultdict(int)
        
        # Console rica si está disponible
        self.console = Console() if RICH_AVAILABLE else None
        
        # Log de inicio
        self.print_header()
    
    def _register_custom_levels(self):
        """Registrar niveles de log personalizados"""
        for level in LogLevel:
            logging.addLevelName(level.value, level.name)
    
    def _setup_loggers(self):
        """Configurar todos los loggers"""
        # Logger principal
        main_logger = logging.getLogger(self.name)
        main_logger.setLevel(logging.DEBUG)
        main_logger.handlers = []
        
        # Console handler
        console = logging.StreamHandler(sys.stdout)
        console.setLevel(logging.INFO)
        console.setFormatter(TradingFormatter(
            self.capabilities, self.symbols, self.colors,
            show_time=True, show_module=False
        ))
        main_logger.addHandler(console)
        
        # File handler - Todos los logs
        file_all = logging.handlers.RotatingFileHandler(
            self.log_dir / 'trading_bot.log',
            maxBytes=10*1024*1024,
            backupCount=5,
            encoding='utf-8'
        )
        file_all.setLevel(logging.DEBUG)
        file_all.setFormatter(logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s'
        ))
        main_logger.addHandler(file_all)
        
        self.loggers['main'] = main_logger
        
        # Loggers especializados
        specialized = {
            'trades': {'file': 'trades.log', 'console': True},
            'performance': {'file': 'performance.log', 'console': True},
            'risk': {'file': 'risk.log', 'console': True},
            'models': {'file': 'models.log', 'console': False},
            'data': {'file': 'data.log', 'console': False},
        }
        
        for name, config in specialized.items():
            logger = logging.getLogger(f"{self.name}.{name}")
            logger.setLevel(logging.DEBUG)
            logger.handlers = []
            logger.propagate = False
            
            # File handler
            file_handler = logging.FileHandler(
                self.log_dir / config['file'],
                encoding='utf-8'
            )
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(logging.Formatter(
                '%(asctime)s | %(levelname)-8s | %(message)s'
            ))
            logger.addHandler(file_handler)
            
            # Console handler si está habilitado
            if config['console']:
                console = logging.StreamHandler(sys.stdout)
                console.setLevel(logging.INFO)
                console.setFormatter(TradingFormatter(
                    self.capabilities, self.symbols, self.colors,
                    show_time=True, show_module=False
                ))
                logger.addHandler(console)
            
            self.loggers[name] = logger
    
    def print_header(self):
        """Imprimir header del sistema"""
        width = min(self.capabilities.columns, 80)
        
        # Crear líneas del header
        lines = []
        
        # Línea superior
        if self.capabilities.supports_unicode:
            lines.append(self.symbols.box['tl'] + self.symbols.box['h'] * (width-2) + self.symbols.box['tr'])
        else:
            lines.append('=' * width)
        
        # Título
        title = f"{self.symbols.symbols['robot']} TRADING BOT SYSTEM {self.symbols.symbols['robot']}"
        lines.append(self._center_text(title, width))
        
        # Subtítulo
        subtitle = f"Session: {self.session_id}"
        lines.append(self._center_text(subtitle, width))
        
        # Información del sistema
        info_lines = [
            f"Platform: {self.capabilities.os_type}",
            f"Terminal: {self.capabilities.terminal or 'Unknown'}",
            f"Color Support: {'Yes' if self.capabilities.supports_color else 'No'}",
            f"Unicode Support: {'Yes' if self.capabilities.supports_unicode else 'No'}",
            f"Emoji Support: {'Yes' if self.capabilities.supports_emoji else 'No'}"
        ]
        
        if self.capabilities.supports_unicode:
            lines.append(self.symbols.box['vr'] + self.symbols.box['light_h'] * (width-2) + self.symbols.box['vl'])
        else:
            lines.append('-' * width)
        
        for info in info_lines:
            lines.append(self._left_align_text(info, width, 2))
        
        # Línea inferior
        if self.capabilities.supports_unicode:
            lines.append(self.symbols.box['bl'] + self.symbols.box['h'] * (width-2) + self.symbols.box['br'])
        else:
            lines.append('=' * width)
        
        # Imprimir header
        header_text = '\n'.join(lines)
        
        if self.colors.use_color:
            # Aplicar gradiente de colores
            print(self.colors.gradient(header_text, ['cyan', 'blue', 'magenta']))
        else:
            print(header_text)
        
        print()  # Línea en blanco
    
    def _center_text(self, text: str, width: int) -> str:
        """Centrar texto con bordes"""
        padding = (width - len(text) - 2) // 2
        left_pad = padding
        right_pad = width - len(text) - 2 - left_pad
        
        if self.capabilities.supports_unicode:
            return f"{self.symbols.box['v']} {' ' * left_pad}{text}{' ' * right_pad} {self.symbols.box['v']}"
        else:
            return f"| {' ' * left_pad}{text}{' ' * right_pad} |"
    
    def _left_align_text(self, text: str, width: int, indent: int = 0) -> str:
        """Alinear texto a la izquierda con bordes"""
        content = ' ' * indent + text
        padding = width - len(content) - 2
        
        if self.capabilities.supports_unicode:
            return f"{self.symbols.box['v']} {content}{' ' * padding} {self.symbols.box['v']}"
        else:
            return f"| {content}{' ' * padding} |"
    
    def get_logger(self, name: str = 'main') -> logging.Logger:
        """Obtener logger específico"""
        return self.loggers.get(name, self.loggers['main'])
    
    def log_trade(self, trade_data: Dict[str, Any]):
        """Log especializado para trades"""
        logger = self.loggers['trades']
        
        action = trade_data.get('action', 'UNKNOWN')
        symbol = trade_data.get('symbol', 'N/A')
        price = trade_data.get('price', 0)
        size = trade_data.get('size', 0)
        profit = trade_data.get('profit', None)
        
        # Incrementar estadísticas
        self.stats['total_trades'] += 1
        
        if profit is None:
            # Trade abierto
            logger.log(LogLevel.TRADE_OPEN.value, 
                      f"{self.symbols.symbols['chart']} OPEN {action} | {symbol} | "
                      f"Size: {size:.2f} | Price: {price:.5f}")
        else:
            # Trade cerrado
            if profit > 0:
                self.stats['winning_trades'] += 1
                logger.log(LogLevel.TRADE_WIN.value,
                          f"{self.symbols.symbols['win']} WIN {action} | {symbol} | "
                          f"Size: {size:.2f} | Profit: ${profit:.2f} "
                          f"{self.symbols.symbols['up']}")
            else:
                self.stats['losing_trades'] += 1
                logger.log(LogLevel.TRADE_LOSS.value,
                          f"{self.symbols.symbols['loss']} LOSS {action} | {symbol} | "
                          f"Size: {size:.2f} | Loss: ${abs(profit):.2f} "
                          f"{self.symbols.symbols['down']}")
    
    def log_performance(self, metrics: Dict[str, Any]):
        """Log de rendimiento con formato especial"""
        logger = self.loggers['performance']
        
        # Crear tabla de rendimiento
        width = min(self.capabilities.columns, 60)
        lines = []
        
        # Header
        if self.capabilities.supports_unicode:
            lines.append(self.symbols.box['tl'] + self.symbols.box['h'] * (width-2) + self.symbols.box['tr'])
        else:
            lines.append('=' * width)
        
        title = f"{self.symbols.symbols['star']} PERFORMANCE METRICS {self.symbols.symbols['star']}"
        lines.append(self._center_text(title, width))
        
        if self.capabilities.supports_unicode:
            lines.append(self.symbols.box['vr'] + self.symbols.box['h'] * (width-2) + self.symbols.box['vl'])
        else:
            lines.append('-' * width)
        
        # Métricas
        for key, value in metrics.items():
            if isinstance(value, float):
                if 'rate' in key.lower() or 'ratio' in key.lower():
                    formatted_value = f"{value:.2%}"
                elif 'balance' in key.lower() or 'profit' in key.lower():
                    formatted_value = f"${value:,.2f}"
                else:
                    formatted_value = f"{value:.4f}"
            else:
                formatted_value = str(value)
            
            metric_line = f"{key}: {formatted_value}"
            lines.append(self._left_align_text(metric_line, width, 2))
        
        # Footer
        if self.capabilities.supports_unicode:
            lines.append(self.symbols.box['bl'] + self.symbols.box['h'] * (width-2) + self.symbols.box['br'])
        else:
            lines.append('=' * width)
        
        # Log cada línea
        for line in lines:
            logger.log(LogLevel.PERFORMANCE.value, line)
    
    def log_risk_alert(self, alert_type: str, message: str, details: Dict = None):
        """Log de alertas de riesgo"""
        logger = self.loggers['risk']
        
        alert_msg = f"{self.symbols.symbols['shield']} RISK ALERT: {alert_type} - {message}"
        
        if self.colors.use_color:
            alert_msg = self.colors.apply(alert_msg, 'red', 'bright')
        
        logger.log(LogLevel.RISK_ALERT.value, alert_msg)
        
        if details:
            for key, value in details.items():
                logger.log(LogLevel.RISK_ALERT.value, f"  {key}: {value}")
    
    def create_progress_bar(self, total: int, description: str = "Processing"):
        """Crear barra de progreso si Rich está disponible"""
        if RICH_AVAILABLE and self.console:
            return Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=self.console
            )
        return None
    
    def show_statistics(self):
        """Mostrar estadísticas de la sesión"""
        logger = self.loggers['main']
        
        # Calcular estadísticas
        total_trades = self.stats['total_trades']
        winning_trades = self.stats['winning_trades']
        losing_trades = self.stats['losing_trades']
        
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # Crear resumen
        summary = {
            'Total Trades': total_trades,
            'Winning Trades': winning_trades,
            'Losing Trades': losing_trades,
            'Win Rate': win_rate,
            'Session Duration': str(datetime.now() - datetime.strptime(self.session_id, '%Y%m%d_%H%M%S'))
        }
        
        self.log_performance(summary)
        
        # Guardar estadísticas en archivo
        stats_file = self.log_dir / 'session_stats.json'
        with open(stats_file, 'w') as f:
            json.dump({
                'session_id': self.session_id,
                'statistics': self.stats,
                'summary': summary
            }, f, indent=2, default=str)


# Instancia global
_logger_instance = None

def get_trading_logger(name: str = "TradingBot", base_dir: Path = None) -> TradingLogger:
    """Obtener instancia del logger"""
    global _logger_instance
    if _logger_instance is None:
        _logger_instance = TradingLogger(name, base_dir)
    return _logger_instance

# Funciones de conveniencia
def get_logger(name: str = 'main') -> logging.Logger:
    """Obtener logger específico"""
    return get_trading_logger().get_logger(name)

def log_trade(trade_data: Dict[str, Any]):
    """Log de trade"""
    get_trading_logger().log_trade(trade_data)

def log_performance(metrics: Dict[str, Any]):
    """Log de performance"""
    get_trading_logger().log_performance(metrics)

def log_risk_alert(alert_type: str, message: str, details: Dict = None):
    """Log de alerta de riesgo"""
    get_trading_logger().log_risk_alert(alert_type, message, details)
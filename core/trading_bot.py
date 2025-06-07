# core/trading_bot.py
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import pandas as pd
import numpy as np

from config.settings import settings
from config.trading_config import TradingConfig
from core.mt5_connector import MT5Connector
from core.trade_executor import TradeExecutor
from data.data_collector import DataCollector
from models.model_hub import ModelHub
from strategies.ml_strategy import MLStrategy
from risk.risk_manager import RiskManager
from analysis.market_analyzer import MarketAnalyzer
from visualization.dashboard import Dashboard
from utils.log_config import get_logger, log_trade, log_performance

logger = get_logger('main')


class BotState(Enum):
    """Estados del bot"""
    INITIALIZING = "INITIALIZING"
    READY = "READY"
    TRADING = "TRADING"
    PAUSED = "PAUSED"
    ERROR = "ERROR"
    STOPPED = "STOPPED"


@dataclass
class BotStatus:
    """Estado actual del bot"""
    state: BotState
    start_time: datetime
    last_update: datetime
    total_trades: int
    active_positions: int
    current_balance: float
    current_equity: float
    daily_pnl: float
    error_count: int
    last_error: Optional[str] = None


class TradingBot:
    """Bot de trading principal con arquitectura modular"""
    
    def __init__(self, config: TradingConfig):
        self.config = config
        self.state = BotState.INITIALIZING
        
        # Componentes del sistema
        self.connector = MT5Connector(config.symbol, config.magic_number)
        self.executor = TradeExecutor(self.connector)
        self.data_collector = DataCollector(self.connector)
        self.model_hub = ModelHub(settings.MODELS_DIR)
        self.risk_manager = RiskManager(config)
        self.market_analyzer = MarketAnalyzer()
        self.strategy = MLStrategy(config)
        
        # Dashboard (opcional)
        self.dashboard = None
        if settings.DASHBOARD_ENABLED:
            self.dashboard = Dashboard()
        
        # Estado interno
        self.status = BotStatus(
            state=BotState.INITIALIZING,
            start_time=datetime.now(),
            last_update=datetime.now(),
            total_trades=0,
            active_positions=0,
            current_balance=0,
            current_equity=0,
            daily_pnl=0,
            error_count=0
        )
        
        # Datos en memoria
        self.market_data: Optional[pd.DataFrame] = None
        self.current_signal: Optional[Dict] = None
        
        logger.info("TradingBot inicializado")
    
    def initialize(self) -> bool:
        """Inicializar todos los componentes"""
        try:
            logger.info("Inicializando componentes del bot...")
            
            # Conectar a MT5
            if not self.connector.connect():
                raise RuntimeError("No se pudo conectar a MT5")
            
            # Verificar trading permitido
            allowed, reason = self.connector.check_trading_allowed()
            if not allowed:
                raise RuntimeError(f"Trading no permitido: {reason}")
            
            # Obtener información de cuenta
            account_info = self.connector.get_account_info()
            self.status.current_balance = account_info['balance']
            self.status.current_equity = account_info['equity']
            
            logger.info(f"Cuenta: Balance=${account_info['balance']:.2f}, "
                       f"Equity=${account_info['equity']:.2f}")
            
            # Cargar o entrenar modelo
            if not self._initialize_model():
                raise RuntimeError("No se pudo inicializar el modelo")
            
            # Inicializar estrategia
            self.strategy.initialize(self.model_hub)
            
            # Iniciar dashboard si está habilitado
            if self.dashboard:
                self.dashboard.start()
            
            self.state = BotState.READY
            self.status.state = BotState.READY
            
            logger.info("✅ Bot inicializado correctamente")
            return True
            
        except Exception as e:
            logger.error(f"Error en inicialización: {e}")
            self.state = BotState.ERROR
            self.status.state = BotState.ERROR
            self.status.last_error = str(e)
            return False
    
    def _initialize_model(self) -> bool:
        """Inicializar o cargar modelo de ML"""
        try:
            # Intentar cargar modelo existente
            models = self.model_hub.list_models()
            
            if len(models) > 0:
                # Usar el mejor modelo disponible
                best_model = self.model_hub.get_best_model()
                if best_model:
                    logger.info(f"Cargando modelo existente: {best_model.name}")
                    self.model_hub.load_model(best_model.name)
                    return True
            
            # Entrenar nuevo modelo
            logger.info("Entrenando nuevo modelo...")
            
            # Obtener datos históricos
            data = self.data_collector.collect_training_data(
                self.config.symbol,
                self.config.timeframe,
                bars=5000
            )
            
            if data is None or len(data) < 1000:
                logger.error("Datos insuficientes para entrenar")
                return False
            
            # Entrenar modelo
            success = self.model_hub.train_new_model(
                data,
                model_type='ensemble',
                auto_select_best=True
            )
            
            return success
            
        except Exception as e:
            logger.error(f"Error inicializando modelo: {e}")
            return False
    
    def run(self):
        """Ejecutar bucle principal del bot"""
        if self.state != BotState.READY:
            logger.error("Bot no está listo para ejecutar")
            return
        
        logger.info("🚀 Iniciando bucle de trading...")
        self.state = BotState.TRADING
        self.status.state = BotState.TRADING
        
        try:
            while self.state == BotState.TRADING:
                # Actualizar ciclo
                self._trading_cycle()
                
                # Esperar hasta la siguiente iteración
                time.sleep(self.config.update_interval)
                
        except KeyboardInterrupt:
            logger.info("Bot detenido por usuario")
        except Exception as e:
            logger.error(f"Error en bucle principal: {e}")
            self.status.error_count += 1
            self.status.last_error = str(e)
        finally:
            self.stop()
    
    def _trading_cycle(self):
        """Ciclo principal de trading"""
        try:
            # 1. Actualizar datos de mercado
            self._update_market_data()
            
            # 2. Analizar condiciones de mercado
            market_conditions = self.market_analyzer.analyze(self.market_data)
            
            # 3. Verificar restricciones de riesgo
            if not self.risk_manager.can_trade():
                logger.debug("Trading restringido por gestión de riesgo")
                return
            
            # 4. Gestionar posiciones existentes
            self._manage_positions()
            
            # 5. Buscar nuevas señales
            if self._should_look_for_signals():
                signal = self.strategy.generate_signal(
                    self.market_data,
                    market_conditions
                )
                
                if signal and signal['confidence'] >= settings.CONFIDENCE_THRESHOLD:
                    self._process_signal(signal)
            
            # 6. Actualizar estado
            self._update_status()
            
            # 7. Actualizar dashboard
            if self.dashboard:
                self._update_dashboard()
                
        except Exception as e:
            logger.error(f"Error en ciclo de trading: {e}")
            self.status.error_count += 1
    
    def _update_market_data(self):
        """Actualizar datos de mercado"""
        new_data = self.data_collector.get_latest_data(
            self.config.symbol,
            self.config.timeframe,
            bars=self.config.lookback_period
        )
        
        if new_data is not None:
            self.market_data = new_data
            logger.debug(f"Datos actualizados: {len(new_data)} barras")
    
    def _manage_positions(self):
        """Gestionar posiciones abiertas"""
        positions = self.connector.get_positions(self.config.symbol)
        self.status.active_positions = len(positions)
        
        for position in positions:
            # Verificar trailing stop
            if self.config.use_trailing_stop:
                self.executor.update_trailing_stop(position)
            
            # Verificar tiempo máximo
            if self.config.max_position_time:
                time_open = (datetime.now() - position.time_open).total_seconds() / 3600
                if time_open > self.config.max_position_time:
                    logger.info(f"Cerrando posición por tiempo: {position.ticket}")
                    self.executor.close_position(position.ticket)
    
    def _should_look_for_signals(self) -> bool:
        """Determinar si buscar nuevas señales"""
        # No buscar si hay posiciones abiertas
        if self.status.active_positions >= self.config.max_positions:
            return False
        
        # Verificar cooldown después de trade
        # TODO: Implementar cooldown
        
        return True
    
    def _process_signal(self, signal: Dict):
        """Procesar señal de trading"""
        logger.info(f"📊 Nueva señal: {signal['action']} con confianza {signal['confidence']:.2%}")
        
        # Calcular tamaño de posición
        position_size = self.risk_manager.calculate_position_size(
            signal,
            self.status.current_balance
        )
        
        if position_size <= 0:
            logger.warning("Tamaño de posición inválido")
            return
        
        # Ejecutar trade
        result = self.executor.execute_market_order(
            symbol=self.config.symbol,
            order_type=signal['action'],
            volume=position_size,
            sl=signal.get('stop_loss'),
            tp=signal.get('take_profit'),
            comment=f"ML Signal {signal['confidence']:.0%}"
        )
        
        if result['success']:
            self.status.total_trades += 1
            log_trade({
                'action': signal['action'],
                'symbol': self.config.symbol,
                'size': position_size,
                'price': result['price'],
                'ticket': result['ticket']
            })
        else:
            logger.error(f"Error ejecutando orden: {result['error']}")
    
    def _update_status(self):
        """Actualizar estado del bot"""
        account_info = self.connector.get_account_info()
        
        if account_info:
            old_balance = self.status.current_balance
            self.status.current_balance = account_info['balance']
            self.status.current_equity = account_info['equity']
            self.status.daily_pnl = account_info['profit']
            self.status.last_update = datetime.now()
            
            # Log si hay cambio significativo
            balance_change = self.status.current_balance - old_balance
            if abs(balance_change) > 1:
                log_performance({
                    'balance': self.status.current_balance,
                    'equity': self.status.current_equity,
                    'daily_pnl': self.status.daily_pnl,
                    'positions': self.status.active_positions
                })
    
    def _update_dashboard(self):
        """Actualizar dashboard visual"""
        if not self.dashboard:
            return
        
        try:
            # Actualizar precio
            price_data = self.connector.get_current_price()
            if price_data:
                self.dashboard.update_price(
                    price_data['time'],
                    price_data['mid']
                )
            
            # Actualizar métricas
            self.dashboard.update_metrics({
                'Balance': f"${self.status.current_balance:.2f}",
                'Equity': f"${self.status.current_equity:.2f}",
                'Daily P&L': f"${self.status.daily_pnl:.2f}",
                'Positions': self.status.active_positions,
                'Total Trades': self.status.total_trades
            })
            
        except Exception as e:
            logger.error(f"Error actualizando dashboard: {e}")
    
    def pause(self):
        """Pausar el bot"""
        if self.state == BotState.TRADING:
            self.state = BotState.PAUSED
            self.status.state = BotState.PAUSED
            logger.info("Bot pausado")
    
    def resume(self):
        """Reanudar el bot"""
        if self.state == BotState.PAUSED:
            self.state = BotState.TRADING
            self.status.state = BotState.TRADING
            logger.info("Bot reanudado")
    
    def stop(self):
        """Detener el bot"""
        logger.info("Deteniendo bot...")
        
        self.state = BotState.STOPPED
        self.status.state = BotState.STOPPED
        
        # Cerrar todas las posiciones si está configurado
        if self.config.close_on_stop:
            positions = self.connector.get_positions(self.config.symbol)
            for position in positions:
                logger.info(f"Cerrando posición {position.ticket}")
                self.executor.close_position(position.ticket)
        
        # Detener dashboard
        if self.dashboard:
            self.dashboard.stop()
        
        # Desconectar de MT5
        self.connector.disconnect()
        
        # Mostrar resumen
        self._show_summary()
        
        logger.info("Bot detenido")
    
    def _show_summary(self):
        """Mostrar resumen de la sesión"""
        duration = datetime.now() - self.status.start_time
        
        summary = {
            'Session Duration': str(duration),
            'Total Trades': self.status.total_trades,
            'Final Balance': self.status.current_balance,
            'Total P&L': self.status.current_balance - settings.INITIAL_BALANCE,
            'Errors': self.status.error_count
        }
        
        log_performance(summary)
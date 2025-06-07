# 2. core/trade_executor.py - Ejecutor de Órdenes
# core/trade_executor.py
import MetaTrader5 as mt5
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import time

from core.mt5_connector import MT5Connector, OrderType, Position
from config.trading_config import TradingConfig
from utils.log_config import get_logger, log_trade, log_risk_alert

logger = get_logger('main')
trade_logger = get_logger('trades')
risk_logger = get_logger('risk')


class OrderResult:
    """Resultado de una orden"""
    def __init__(self, success: bool, ticket: Optional[int] = None, 
                 price: Optional[float] = None, error: Optional[str] = None,
                 retcode: Optional[int] = None):
        self.success = success
        self.ticket = ticket
        self.price = price
        self.error = error
        self.retcode = retcode
        self.timestamp = datetime.now()
    
    def to_dict(self) -> Dict:
        """Convertir a diccionario"""
        return {
            'success': self.success,
            'ticket': self.ticket,
            'price': self.price,
            'error': self.error,
            'retcode': self.retcode,
            'timestamp': self.timestamp.isoformat()
        }


class TradeExecutor:
    """Ejecutor de órdenes con logging completo"""
    
    def __init__(self, connector: MT5Connector, config: Optional[TradingConfig] = None):
        self.connector = connector
        self.config = config or TradingConfig()
        self.pending_orders: Dict[int, Dict] = {}
        self.execution_stats = {
            'total_orders': 0,
            'successful_orders': 0,
            'failed_orders': 0,
            'total_slippage': 0.0,
            'rejections': {}
        }
        
        logger.info("TradeExecutor inicializado")
    
    def execute_market_order(self, symbol: str, order_type: Union[str, OrderType],
                           volume: float, sl: Optional[float] = None,
                           tp: Optional[float] = None, comment: str = "",
                           max_retries: int = 3) -> OrderResult:
        """
        Ejecutar orden de mercado con reintentos
        
        Args:
            symbol: Símbolo a operar
            order_type: Tipo de orden (BUY/SELL)
            volume: Volumen de la operación
            sl: Stop loss (opcional)
            tp: Take profit (opcional)
            comment: Comentario de la orden
            max_retries: Número máximo de reintentos
            
        Returns:
            OrderResult con el resultado de la operación
        """
        # Log de inicio de orden
        trade_logger.info(f"{'='*60}")
        trade_logger.info(f"NUEVA ORDEN DE MERCADO")
        trade_logger.info(f"{'='*60}")
        trade_logger.info(f"Símbolo: {symbol}")
        trade_logger.info(f"Tipo: {order_type}")
        trade_logger.info(f"Volumen: {volume}")
        trade_logger.info(f"SL: {sl}")
        trade_logger.info(f"TP: {tp}")
        trade_logger.info(f"Comentario: {comment}")
        
        # Validaciones previas
        validation = self._validate_order(symbol, order_type, volume)
        if not validation['valid']:
            error_msg = f"Validación fallida: {validation['error']}"
            trade_logger.error(error_msg)
            return OrderResult(False, error=error_msg)
        
        # Normalizar volumen
        volume = self.connector.normalize_volume(volume)
        trade_logger.info(f"Volumen normalizado: {volume}")
        
        # Obtener precio actual
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            error_msg = "No se pudo obtener precio actual"
            trade_logger.error(error_msg)
            return OrderResult(False, error=error_msg)
        
        # Determinar precio según tipo de orden
        is_buy = order_type in [OrderType.BUY, 'BUY']
        price = tick.ask if is_buy else tick.bid
        
        # Calcular SL/TP si no se proporcionaron
        if sl is None and self.config.default_sl_pips > 0:
            sl = self._calculate_sl(price, is_buy, self.config.default_sl_pips)
            trade_logger.info(f"SL calculado automáticamente: {sl}")
        
        if tp is None and self.config.default_tp_pips > 0:
            tp = self._calculate_tp(price, is_buy, self.config.default_tp_pips)
            trade_logger.info(f"TP calculado automáticamente: {tp}")
        
        # Normalizar precios
        if sl: sl = self.connector.normalize_price(sl)
        if tp: tp = self.connector.normalize_price(tp)
        
        # Preparar request
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": volume,
            "type": mt5.ORDER_TYPE_BUY if is_buy else mt5.ORDER_TYPE_SELL,
            "price": price,
            "deviation": self.config.slippage,
            "magic": self.config.magic_number,
            "comment": comment,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        if sl: request["sl"] = sl
        if tp: request["tp"] = tp
        
        # Intentar ejecutar con reintentos
        attempts = 0
        last_error = None
        
        while attempts < max_retries:
            attempts += 1
            trade_logger.info(f"Intento {attempts}/{max_retries}")
            
            # Ejecutar orden
            result = mt5.order_send(request)
            
            # Log detallado del resultado
            self._log_order_result(result)
            
            if result.retcode == mt5.TRADE_RETCODE_DONE:
                # Éxito
                execution_price = result.price
                slippage = abs(execution_price - price) * 10000  # en pips
                
                trade_logger.info(f"✅ ORDEN EJECUTADA EXITOSAMENTE")
                trade_logger.info(f"Ticket: {result.order}")
                trade_logger.info(f"Precio solicitado: {price}")
                trade_logger.info(f"Precio ejecutado: {execution_price}")
                trade_logger.info(f"Slippage: {slippage:.1f} pips")
                
                # Actualizar estadísticas
                self.execution_stats['total_orders'] += 1
                self.execution_stats['successful_orders'] += 1
                self.execution_stats['total_slippage'] += slippage
                
                # Log para sistema de trading
                log_trade({
                    'action': 'BUY' if is_buy else 'SELL',
                    'symbol': symbol,
                    'size': volume,
                    'price': execution_price,
                    'sl': sl,
                    'tp': tp,
                    'ticket': result.order,
                    'slippage': slippage
                })
                
                return OrderResult(
                    success=True,
                    ticket=result.order,
                    price=execution_price,
                    retcode=result.retcode
                )
            
            else:
                # Error
                last_error = self._get_error_message(result.retcode)
                trade_logger.error(f"❌ Error en orden: {last_error}")
                
                # Verificar si el error es recuperable
                if not self._is_recoverable_error(result.retcode):
                    break
                
                # Esperar antes de reintentar
                if attempts < max_retries:
                    wait_time = attempts * 2  # Backoff exponencial
                    trade_logger.info(f"Esperando {wait_time} segundos antes de reintentar...")
                    time.sleep(wait_time)
                    
                    # Actualizar precio para siguiente intento
                    tick = mt5.symbol_info_tick(symbol)
                    if tick:
                        request["price"] = tick.ask if is_buy else tick.bid
        
        # Fallo después de todos los intentos
        self.execution_stats['total_orders'] += 1
        self.execution_stats['failed_orders'] += 1
        
        # Registrar tipo de rechazo
        error_key = f"retcode_{result.retcode}"
        self.execution_stats['rejections'][error_key] = \
            self.execution_stats['rejections'].get(error_key, 0) + 1
        
        trade_logger.error(f"{'='*60}")
        trade_logger.error(f"ORDEN FALLIDA DESPUÉS DE {attempts} INTENTOS")
        trade_logger.error(f"Último error: {last_error}")
        trade_logger.error(f"{'='*60}")
        
        return OrderResult(
            success=False,
            error=last_error,
            retcode=result.retcode
        )
    
    def close_position(self, ticket: int, comment: str = "Close by bot") -> OrderResult:
        """Cerrar posición específica"""
        trade_logger.info(f"{'='*60}")
        trade_logger.info(f"CERRANDO POSICIÓN")
        trade_logger.info(f"Ticket: {ticket}")
        
        # Obtener información de la posición
        position = mt5.positions_get(ticket=ticket)
        
        if not position:
            error_msg = f"Posición {ticket} no encontrada"
            trade_logger.error(error_msg)
            return OrderResult(False, error=error_msg)
        
        position = position[0]
        
        # Preparar orden de cierre
        symbol = position.symbol
        volume = position.volume
        is_buy = position.type == mt5.POSITION_TYPE_BUY
        
        # Precio de cierre (inverso a la posición)
        tick = mt5.symbol_info_tick(symbol)
        if not tick:
            error_msg = "No se pudo obtener precio para cierre"
            trade_logger.error(error_msg)
            return OrderResult(False, error=error_msg)
        
        price = tick.bid if is_buy else tick.ask
        
        # Calcular P&L estimado
        if is_buy:
            pnl_pips = (price - position.price_open) * 10000
        else:
            pnl_pips = (position.price_open - price) * 10000
        
        trade_logger.info(f"Símbolo: {symbol}")
        trade_logger.info(f"Tipo: {'BUY' if is_buy else 'SELL'}")
        trade_logger.info(f"Volumen: {volume}")
        trade_logger.info(f"Precio apertura: {position.price_open}")
        trade_logger.info(f"Precio cierre: {price}")
        trade_logger.info(f"P&L estimado: {pnl_pips:.1f} pips / ${position.profit:.2f}")
        
        # Preparar request de cierre
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": volume,
            "type": mt5.ORDER_TYPE_SELL if is_buy else mt5.ORDER_TYPE_BUY,
            "position": ticket,
            "price": price,
            "deviation": self.config.slippage,
            "magic": self.config.magic_number,
            "comment": comment,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        # Ejecutar cierre
        result = mt5.order_send(request)
        self._log_order_result(result)
        
        if result.retcode == mt5.TRADE_RETCODE_DONE:
            trade_logger.info(f"✅ POSICIÓN CERRADA EXITOSAMENTE")
            trade_logger.info(f"Precio de cierre: {result.price}")
            
            # Log trade cerrado
            log_trade({
                'action': 'CLOSE',
                'symbol': symbol,
                'size': volume,
                'price': result.price,
                'profit': position.profit,
                'ticket': ticket,
                'pnl_pips': pnl_pips
            })
            
            # Verificar si fue pérdida significativa
            if position.profit < -self.config.risk_per_trade * 1000:  # Ejemplo: pérdida mayor al riesgo
                log_risk_alert(
                    "PÉRDIDA SIGNIFICATIVA",
                    f"Posición {ticket} cerrada con pérdida de ${abs(position.profit):.2f}",
                    {
                        'ticket': ticket,
                        'symbol': symbol,
                        'loss': position.profit,
                        'pips': pnl_pips
                    }
                )
            
            return OrderResult(
                success=True,
                ticket=result.order,
                price=result.price,
                retcode=result.retcode
            )
        else:
            error_msg = self._get_error_message(result.retcode)
            trade_logger.error(f"❌ Error cerrando posición: {error_msg}")
            return OrderResult(False, error=error_msg, retcode=result.retcode)
    
    def modify_position(self, ticket: int, sl: Optional[float] = None,
                       tp: Optional[float] = None) -> OrderResult:
        """Modificar SL/TP de una posición"""
        trade_logger.info(f"Modificando posición {ticket} - SL: {sl}, TP: {tp}")
        
        # Obtener posición
        position = mt5.positions_get(ticket=ticket)
        if not position:
            error_msg = f"Posición {ticket} no encontrada"
            trade_logger.error(error_msg)
            return OrderResult(False, error=error_msg)
        
        position = position[0]
        
        # Normalizar precios
        if sl: sl = self.connector.normalize_price(sl)
        if tp: tp = self.connector.normalize_price(tp)
        
        # Preparar request
        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "position": ticket,
            "symbol": position.symbol,
            "magic": self.config.magic_number,
        }
        
        if sl is not None: request["sl"] = sl
        if tp is not None: request["tp"] = tp
        
        # Ejecutar modificación
        result = mt5.order_send(request)
        
        if result.retcode == mt5.TRADE_RETCODE_DONE:
            trade_logger.info(f"✅ Posición modificada exitosamente")
            return OrderResult(success=True, ticket=ticket)
        else:
            error_msg = self._get_error_message(result.retcode)
            trade_logger.error(f"❌ Error modificando posición: {error_msg}")
            return OrderResult(False, error=error_msg, retcode=result.retcode)
    
    def update_trailing_stop(self, position: Position, 
                           trail_points: Optional[int] = None) -> bool:
        """Actualizar trailing stop de una posición"""
        if not self.config.use_trailing_stop:
            return False
        
        trail_points = trail_points or int(self.config.trailing_stop_pips * 10)
        step_points = int(self.config.trailing_step_pips * 10)
        
        # Calcular nuevo stop loss
        current_price = position.price_current
        current_sl = position.sl
        
        if position.is_buy:
            # Para BUY: mover SL hacia arriba
            new_sl = current_price - (trail_points * self.connector.symbol_info.point)
            
            # Solo actualizar si mejora
            if current_sl == 0 or new_sl > current_sl + (step_points * self.connector.symbol_info.point):
                result = self.modify_position(position.ticket, sl=new_sl)
                if result.success:
                    trade_logger.info(f"📈 Trailing stop actualizado para {position.ticket}: "
                                    f"{current_sl:.5f} -> {new_sl:.5f}")
                return result.success
        else:
            # Para SELL: mover SL hacia abajo
            new_sl = current_price + (trail_points * self.connector.symbol_info.point)
            
            # Solo actualizar si mejora
            if current_sl == 0 or new_sl < current_sl - (step_points * self.connector.symbol_info.point):
                result = self.modify_position(position.ticket, sl=new_sl)
                if result.success:
                    trade_logger.info(f"📉 Trailing stop actualizado para {position.ticket}: "
                                    f"{current_sl:.5f} -> {new_sl:.5f}")
                return result.success
        
        return False
    
    def _validate_order(self, symbol: str, order_type: Union[str, OrderType],
                       volume: float) -> Dict[str, Union[bool, str]]:
        """Validar orden antes de ejecutar"""
        # Verificar conexión
        if not self.connector.connected:
            return {'valid': False, 'error': 'No conectado a MT5'}
        
        # Verificar símbolo
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            return {'valid': False, 'error': f'Símbolo {symbol} no encontrado'}
        
        if not symbol_info.visible:
            return {'valid': False, 'error': f'Símbolo {symbol} no visible'}
        
        # Verificar volumen
        if volume <= 0:
            return {'valid': False, 'error': 'Volumen debe ser mayor a 0'}
        
        if volume < symbol_info.volume_min:
            return {'valid': False, 'error': f'Volumen menor al mínimo ({symbol_info.volume_min})'}
        
        if volume > symbol_info.volume_max:
            return {'valid': False, 'error': f'Volumen mayor al máximo ({symbol_info.volume_max})'}
        
        # Verificar spread
        if self.config.use_spread_filter:
            tick = mt5.symbol_info_tick(symbol)
            if tick:
                spread_pips = (tick.ask - tick.bid) * 10000
                if spread_pips > self.config.max_spread:
                    return {'valid': False, 'error': f'Spread muy alto: {spread_pips:.1f} pips'}
        
        # Verificar margen
        account = mt5.account_info()
        if account:
            if account.margin_free <= 0:
                return {'valid': False, 'error': 'Margen libre insuficiente'}
        
        return {'valid': True, 'error': None}
    
    def _calculate_sl(self, price: float, is_buy: bool, pips: float) -> float:
        """Calcular stop loss"""
        point_value = self.connector.symbol_info.point * 10  # Para pips
        
        if is_buy:
            return price - (pips * point_value)
        else:
            return price + (pips * point_value)
    
    def _calculate_tp(self, price: float, is_buy: bool, pips: float) -> float:
        """Calcular take profit"""
        point_value = self.connector.symbol_info.point * 10  # Para pips
        
        if is_buy:
            return price + (pips * point_value)
        else:
            return price - (pips * point_value)
    
    def _log_order_result(self, result):
        """Log detallado del resultado de orden"""
        trade_logger.info(f"--- Resultado de Orden ---")
        trade_logger.info(f"Retcode: {result.retcode} ({self._get_error_message(result.retcode)})")
        trade_logger.info(f"Deal: {result.deal}")
        trade_logger.info(f"Order: {result.order}")
        trade_logger.info(f"Volume: {result.volume}")
        trade_logger.info(f"Price: {result.price}")
        trade_logger.info(f"Bid: {result.bid}")
        trade_logger.info(f"Ask: {result.ask}")
        trade_logger.info(f"Comment: {result.comment}")
        trade_logger.info(f"Request ID: {result.request_id}")
        trade_logger.info(f"-------------------------")
    
    def _get_error_message(self, retcode: int) -> str:
        """Obtener mensaje de error legible"""
        error_codes = {
            mt5.TRADE_RETCODE_REQUOTE: "Requote - precios cambiaron",
            mt5.TRADE_RETCODE_REJECT: "Orden rechazada",
            mt5.TRADE_RETCODE_CANCEL: "Orden cancelada",
            mt5.TRADE_RETCODE_PLACED: "Orden colocada",
            mt5.TRADE_RETCODE_DONE: "Orden ejecutada",
            mt5.TRADE_RETCODE_DONE_PARTIAL: "Orden parcialmente ejecutada",
            mt5.TRADE_RETCODE_ERROR: "Error general",
            mt5.TRADE_RETCODE_TIMEOUT: "Timeout",
            mt5.TRADE_RETCODE_INVALID: "Orden inválida",
            mt5.TRADE_RETCODE_INVALID_VOLUME: "Volumen inválido",
            mt5.TRADE_RETCODE_INVALID_PRICE: "Precio inválido",
            mt5.TRADE_RETCODE_INVALID_STOPS: "Stops inválidos",
            mt5.TRADE_RETCODE_TRADE_DISABLED: "Trading deshabilitado",
            mt5.TRADE_RETCODE_MARKET_CLOSED: "Mercado cerrado",
            mt5.TRADE_RETCODE_NO_MONEY: "Fondos insuficientes",
            mt5.TRADE_RETCODE_PRICE_CHANGED: "Precio cambió",
            mt5.TRADE_RETCODE_PRICE_OFF: "Precio fuera de mercado",
            mt5.TRADE_RETCODE_INVALID_EXPIRATION: "Expiración inválida",
            mt5.TRADE_RETCODE_ORDER_CHANGED: "Orden cambió",
            mt5.TRADE_RETCODE_TOO_MANY_REQUESTS: "Demasiadas solicitudes",
            mt5.TRADE_RETCODE_NO_CHANGES: "Sin cambios",
            mt5.TRADE_RETCODE_SERVER_DISABLES_AT: "AutoTrading deshabilitado por servidor",
            mt5.TRADE_RETCODE_CLIENT_DISABLES_AT: "AutoTrading deshabilitado por cliente",
            mt5.TRADE_RETCODE_LOCKED: "Orden bloqueada",
            mt5.TRADE_RETCODE_FROZEN: "Orden congelada",
            mt5.TRADE_RETCODE_INVALID_FILL: "Tipo de fill inválido",
            mt5.TRADE_RETCODE_CONNECTION: "Sin conexión",
            mt5.TRADE_RETCODE_ONLY_REAL: "Solo permitido en cuenta real",
            mt5.TRADE_RETCODE_LIMIT_ORDERS: "Límite de órdenes alcanzado",
            mt5.TRADE_RETCODE_LIMIT_VOLUME: "Límite de volumen alcanzado",
            mt5.TRADE_RETCODE_INVALID_ORDER: "Orden inválida",
            mt5.TRADE_RETCODE_POSITION_CLOSED: "Posición ya cerrada"
        }
        
        return error_codes.get(retcode, f"Error desconocido (código: {retcode})")
    
    def _is_recoverable_error(self, retcode: int) -> bool:
        """Determinar si un error es recuperable"""
        recoverable_errors = [
            mt5.TRADE_RETCODE_REQUOTE,
            mt5.TRADE_RETCODE_PRICE_CHANGED,
            mt5.TRADE_RETCODE_PRICE_OFF,
            mt5.TRADE_RETCODE_CONNECTION,
            mt5.TRADE_RETCODE_TIMEOUT,
            mt5.TRADE_RETCODE_TOO_MANY_REQUESTS
        ]
        
        return retcode in recoverable_errors
    
    def get_execution_statistics(self) -> Dict:
        """Obtener estadísticas de ejecución"""
        stats = self.execution_stats.copy()
        
        # Calcular métricas adicionales
        if stats['total_orders'] > 0:
            stats['success_rate'] = stats['successful_orders'] / stats['total_orders']
            stats['avg_slippage'] = stats['total_slippage'] / stats['successful_orders'] \
                if stats['successful_orders'] > 0 else 0
        else:
            stats['success_rate'] = 0
            stats['avg_slippage'] = 0
        
        # Log estadísticas
        trade_logger.info("="*60)
        trade_logger.info("ESTADÍSTICAS DE EJECUCIÓN")
        trade_logger.info("="*60)
        trade_logger.info(f"Total órdenes: {stats['total_orders']}")
        trade_logger.info(f"Exitosas: {stats['successful_orders']}")
        trade_logger.info(f"Fallidas: {stats['failed_orders']}")
        trade_logger.info(f"Tasa de éxito: {stats['success_rate']:.1%}")
        trade_logger.info(f"Slippage promedio: {stats['avg_slippage']:.2f} pips")
        
        if stats['rejections']:
            trade_logger.info("\nRazones de rechazo:")
            for reason, count in stats['rejections'].items():
                trade_logger.info(f"  {reason}: {count}")
        
        trade_logger.info("="*60)
        
        return stats
import sys
import os
import asyncio
import signal
import argparse
from datetime import datetime
import logging
from typing import Optional
from pathlib import Path

# Agregar el directorio del proyecto al path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.settings import Settings
from config.trading_config import TradingConfig
from core.trading_bot import TradingBot
from utils.log_config import setup_logging, get_logger
from visualization.dashboard import TradingDashboard
from utils.notifications import NotificationManager

class TradingBotApplication:
    """Aplicación principal del Trading Bot"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Inicializa la aplicación del trading bot
        
        Args:
            config_path: Ruta al archivo de configuración (opcional)
        """
        self.config_path = config_path or "config/settings.yaml"
        self.logger = None
        self.trading_bot = None
        self.dashboard = None
        self.notification_manager = None
        self.is_running = False
        
        # Configurar signal handlers
        self._setup_signal_handlers()
        
    def _setup_signal_handlers(self):
        """Configura manejadores de señales para cierre ordenado"""
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        # Windows specific
        if sys.platform == "win32":
            signal.signal(signal.SIGBREAK, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Maneja señales del sistema para cierre ordenado"""
        self.logger.info(f"Señal recibida: {signum}. Iniciando cierre ordenado...")
        self.shutdown()
    
    def initialize(self) -> bool:
        """Inicializa todos los componentes del sistema"""
        try:
            # 1. Configurar logging
            log_config = setup_logging()
            self.logger = get_logger(__name__)
            self.logger.info("=== Iniciando Trading Bot ===")
            
            # 2. Cargar configuración
            self.logger.info("Cargando configuración...")
            self.settings = Settings.load_from_file(self.config_path)
            self.trading_config = TradingConfig.from_settings(self.settings)
            
            # 3. Validar configuración
            if not self._validate_configuration():
                return False
            
            # 4. Inicializar sistema de notificaciones
            self.logger.info("Inicializando sistema de notificaciones...")
            self.notification_manager = NotificationManager(self.settings.notifications)
            
            # 5. Crear instancia del trading bot
            self.logger.info("Creando instancia del trading bot...")
            self.trading_bot = TradingBot(
                config=self.trading_config,
                notification_manager=self.notification_manager
            )
            
            # 6. Inicializar el bot
            self.logger.info("Inicializando trading bot...")
            if not self.trading_bot.initialize():
                self.logger.error("Error al inicializar el trading bot")
                return False
            
            # 7. Verificar conexión con MT5
            if not self.trading_bot.check_mt5_connection():
                self.logger.error("No se pudo establecer conexión con MetaTrader 5")
                return False
            
            # 8. Cargar modelos ML si están configurados
            if self.settings.ml_config.enabled:
                self.logger.info("Cargando modelos de Machine Learning...")
                self.trading_bot.load_ml_models()
            
            # 9. Inicializar dashboard si está habilitado
            if self.settings.dashboard.enabled:
                self.logger.info("Inicializando dashboard...")
                self.dashboard = TradingDashboard(self.trading_bot)
            
            # 10. Enviar notificación de inicio
            self.notification_manager.send_notification(
                "Sistema Iniciado",
                f"Trading Bot iniciado correctamente en {self.settings.environment} "
                f"a las {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                priority="info"
            )
            
            self.logger.info("=== Inicialización completada exitosamente ===")
            return True
            
        except Exception as e:
            self.logger.error(f"Error durante la inicialización: {str(e)}", exc_info=True)
            return False
    
    def _validate_configuration(self) -> bool:
        """Valida la configuración del sistema"""
        self.logger.info("Validando configuración...")
        
        # Verificar credenciales MT5
        if not all([
            self.settings.mt5.login,
            self.settings.mt5.password,
            self.settings.mt5.server
        ]):
            self.logger.error("Credenciales de MT5 incompletas")
            return False
        
        # Verificar símbolos configurados
        if not self.trading_config.symbols:
            self.logger.error("No hay símbolos configurados para operar")
            return False
        
        # Verificar estrategias
        if not self.trading_config.active_strategies:
            self.logger.error("No hay estrategias activas configuradas")
            return False
        
        # Verificar parámetros de riesgo
        if self.trading_config.risk_per_trade <= 0 or self.trading_config.risk_per_trade > 0.1:
            self.logger.error("risk_per_trade debe estar entre 0 y 0.1 (10%)")
            return False
        
        self.logger.info("Configuración validada correctamente")
        return True
    
    async def run_async(self):
        """Ejecuta el bot en modo asíncrono"""
        self.logger.info("Iniciando ejecución asíncrona del trading bot...")
        self.is_running = True
        
        try:
            # Crear tareas concurrentes
            tasks = []
            
            # Tarea principal del bot
            tasks.append(asyncio.create_task(
                self.trading_bot.run_async(),
                name="trading_bot_main"
            ))
            
            # Tarea del dashboard si está habilitado
            if self.dashboard:
                tasks.append(asyncio.create_task(
                    self._run_dashboard_async(),
                    name="dashboard"
                ))
            
            # Tarea de monitoreo de salud
            tasks.append(asyncio.create_task(
                self._health_monitor(),
                name="health_monitor"
            ))
            
            # Tarea de reportes programados
            if self.settings.reports.scheduled_reports_enabled:
                tasks.append(asyncio.create_task(
                    self._run_scheduled_reports(),
                    name="scheduled_reports"
                ))
            
            # Esperar a que todas las tareas terminen
            await asyncio.gather(*tasks)
            
        except asyncio.CancelledError:
            self.logger.info("Tareas asíncronas canceladas")
        except Exception as e:
            self.logger.error(f"Error en ejecución asíncrona: {str(e)}", exc_info=True)
            raise
        finally:
            self.is_running = False
    
    def run_sync(self):
        """Ejecuta el bot en modo síncrono"""
        self.logger.info("Iniciando ejecución síncrona del trading bot...")
        self.is_running = True
        
        try:
            # Ejecutar el bot
            self.trading_bot.run()
            
        except KeyboardInterrupt:
            self.logger.info("Interrupción por teclado detectada")
        except Exception as e:
            self.logger.error(f"Error en ejecución síncrona: {str(e)}", exc_info=True)
            raise
        finally:
            self.is_running = False
    
    async def _health_monitor(self):
        """Monitorea la salud del sistema"""
        check_interval = 60  # segundos
        
        while self.is_running:
            try:
                # Verificar estado del bot
                health_status = self.trading_bot.get_health_status()
                
                if not health_status['healthy']:
                    self.logger.warning(f"Problema de salud detectado: {health_status['issues']}")
                    
                    # Enviar notificación
                    self.notification_manager.send_notification(
                        "Alerta de Salud del Sistema",
                        f"Problemas detectados: {', '.join(health_status['issues'])}",
                        priority="high"
                    )
                    
                    # Intentar recuperación automática
                    if health_status['recoverable']:
                        self.logger.info("Intentando recuperación automática...")
                        recovery_result = await self.trading_bot.attempt_recovery()
                        
                        if recovery_result['success']:
                            self.logger.info("Recuperación exitosa")
                        else:
                            self.logger.error(f"Recuperación fallida: {recovery_result['error']}")
                
                # Esperar antes del siguiente chequeo
                await asyncio.sleep(check_interval)
                
            except Exception as e:
                self.logger.error(f"Error en monitor de salud: {str(e)}")
                await asyncio.sleep(check_interval)
    
    async def _run_dashboard_async(self):
        """Ejecuta el dashboard en modo asíncrono"""
        try:
            # El dashboard de Streamlit se ejecuta en su propio proceso
            import subprocess
            
            dashboard_process = subprocess.Popen([
                sys.executable,
                "-m",
                "streamlit",
                "run",
                "visualization/dashboard.py",
                "--server.port",
                str(self.settings.dashboard.port),
                "--server.address",
                self.settings.dashboard.host
            ])
            
            # Esperar a que el proceso termine o el bot se detenga
            while self.is_running and dashboard_process.poll() is None:
                await asyncio.sleep(1)
            
            # Terminar el proceso si sigue activo
            if dashboard_process.poll() is None:
                dashboard_process.terminate()
                
        except Exception as e:
            self.logger.error(f"Error ejecutando dashboard: {str(e)}")
    
    async def _run_scheduled_reports(self):
        """Ejecuta reportes programados"""
        while self.is_running:
            try:
                # Verificar reportes pendientes
                pending_reports = self.trading_bot.get_pending_reports()
                
                for report_config in pending_reports:
                    self.logger.info(f"Generando reporte programado: {report_config['name']}")
                    
                    # Generar reporte
                    report_result = await self.trading_bot.generate_report_async(report_config)
                    
                    if report_result['success']:
                        self.logger.info(f"Reporte generado: {report_result['file_path']}")
                        
                        # Enviar por email si está configurado
                        if report_config.get('send_email', False):
                            await self._send_report_email(
                                report_result['file_path'],
                                report_config['recipients']
                            )
                    else:
                        self.logger.error(f"Error generando reporte: {report_result['error']}")
                
                # Esperar hasta la próxima verificación
                await asyncio.sleep(300)  # 5 minutos
                
            except Exception as e:
                self.logger.error(f"Error en reportes programados: {str(e)}")
                await asyncio.sleep(300)
    
    def shutdown(self):
        """Cierra ordenadamente todos los componentes"""
        self.logger.info("Iniciando proceso de cierre...")
        self.is_running = False
        
        try:
            # 1. Detener el trading bot
            if self.trading_bot:
                self.logger.info("Deteniendo trading bot...")
                self.trading_bot.stop()
                
                # Cerrar todas las posiciones si está configurado
                if self.settings.shutdown.close_all_positions:
                    self.logger.info("Cerrando todas las posiciones abiertas...")
                    closed_positions = self.trading_bot.close_all_positions()
                    self.logger.info(f"Cerradas {len(closed_positions)} posiciones")
            
            # 2. Guardar estado
            if self.trading_bot:
                self.logger.info("Guardando estado del sistema...")
                self.trading_bot.save_state()
            
            # 3. Generar reporte de cierre
            if self.settings.shutdown.generate_final_report:
                self.logger.info("Generando reporte final...")
                self._generate_shutdown_report()
            
            # 4. Cerrar dashboard
            if self.dashboard:
                self.logger.info("Cerrando dashboard...")
                # El dashboard se cierra automáticamente cuando termina el proceso
            
            # 5. Enviar notificación de cierre
            if self.notification_manager:
                self.notification_manager.send_notification(
                    "Sistema Detenido",
                    f"Trading Bot detenido correctamente a las "
                    f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                    priority="info"
                )
            
            # 6. Cerrar conexiones
            if self.trading_bot:
                self.trading_bot.cleanup()
            
            self.logger.info("=== Cierre completado exitosamente ===")
            
        except Exception as e:
            self.logger.error(f"Error durante el cierre: {str(e)}", exc_info=True)
    
    def _generate_shutdown_report(self):
        """Genera reporte final al cerrar el sistema"""
        try:
            report_data = {
                'shutdown_time': datetime.now(),
                'session_duration': self.trading_bot.get_session_duration(),
                'total_trades': self.trading_bot.get_session_trade_count(),
                'session_pnl': self.trading_bot.get_session_pnl(),
                'open_positions': self.trading_bot.get_open_positions_summary()
            }
            
            # Guardar reporte
            report_path = Path("reports") / f"shutdown_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            report_path.parent.mkdir(exist_ok=True)
            
            import json
            with open(report_path, 'w') as f:
                json.dump(report_data, f, indent=2, default=str)
            
            self.logger.info(f"Reporte de cierre guardado en: {report_path}")
            
        except Exception as e:
            self.logger.error(f"Error generando reporte de cierre: {str(e)}")


def parse_arguments():
    """Parsea argumentos de línea de comandos"""
    parser = argparse.ArgumentParser(
        description='Trading Bot Cuantitativo para MetaTrader 5'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='config/settings.yaml',
        help='Ruta al archivo de configuración'
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        choices=['live', 'paper', 'backtest'],
        default='paper',
        help='Modo de ejecución del bot'
    )
    
    parser.add_argument(
        '--async',
        action='store_true',
        help='Ejecutar en modo asíncrono'
    )
    
    parser.add_argument(
        '--no-dashboard',
        action='store_true',
        help='Desactivar el dashboard web'
    )
    
    parser.add_argument(
        '--symbols',
        type=str,
        nargs='+',
        help='Símbolos a operar (override de configuración)'
    )
    
    parser.add_argument(
        '--strategies',
        type=str,
        nargs='+',
        help='Estrategias a activar (override de configuración)'
    )
    
    parser.add_argument(
        '--risk',
        type=float,
        help='Riesgo por operación (override de configuración)'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Ejecutar sin realizar operaciones reales'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Activar modo debug con logging detallado'
    )
    
    return parser.parse_args()


def apply_command_line_overrides(app: TradingBotApplication, args):
    """Aplica overrides de línea de comandos a la configuración"""
    
    # Modo de ejecución
    if args.mode:
        app.settings.environment = args.mode
    
    # Dashboard
    if args.no_dashboard:
        app.settings.dashboard.enabled = False
    
    # Símbolos
    if args.symbols:
        app.trading_config.symbols = args.symbols
    
    # Estrategias
    if args.strategies:
        app.trading_config.active_strategies = args.strategies
    
    # Riesgo
    if args.risk:
        app.trading_config.risk_per_trade = args.risk
    
    # Dry run
    if args.dry_run:
        app.settings.trading.dry_run = True
    
    # Debug
    if args.debug:
        app.settings.logging.level = 'DEBUG'
        logging.getLogger().setLevel(logging.DEBUG)


def main():
    """Función principal de entrada"""
    # Parsear argumentos
    args = parse_arguments()
    
    # Crear aplicación
    app = TradingBotApplication(config_path=args.config)
    
    try:
        # Inicializar
        if not app.initialize():
            print("Error: No se pudo inicializar el sistema")
            sys.exit(1)
        
        # Aplicar overrides de línea de comandos
        apply_command_line_overrides(app, args)
        
        # Ejecutar
        if args.async:
            # Modo asíncrono
            asyncio.run(app.run_async())
        else:
            # Modo síncrono
            app.run_sync()
        
    except Exception as e:
        print(f"Error fatal: {str(e)}")
        if app.logger:
            app.logger.error(f"Error fatal: {str(e)}", exc_info=True)
        sys.exit(1)
    
    finally:
        # Asegurar cierre ordenado
        app.shutdown()
        sys.exit(0)


if __name__ == "__main__":
    main()
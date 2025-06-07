#!/usr/bin/env python3
"""
Script de configuración inicial para TradingBot Cuantitativo MT5
================================================================

Este script realiza las siguientes tareas:
1. Verifica las dependencias del sistema
2. Crea la estructura de directorios necesaria
3. Verifica la instalación de MetaTrader 5
4. Configura las variables de entorno básicas
5. Realiza pruebas de conectividad

Uso:
    python scripts/setup_project.py
"""

import os
import sys
import shutil
import platform
import subprocess
from pathlib import Path
from datetime import datetime
import json

# Colores para la consola
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def print_header(text):
    """Imprimir encabezado con formato"""
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*60}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{text.center(60)}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'='*60}{Colors.ENDC}\n")

def print_success(text):
    """Imprimir mensaje de éxito"""
    print(f"{Colors.OKGREEN}✓ {text}{Colors.ENDC}")

def print_warning(text):
    """Imprimir advertencia"""
    print(f"{Colors.WARNING}⚠ {text}{Colors.ENDC}")

def print_error(text):
    """Imprimir error"""
    print(f"{Colors.FAIL}✗ {text}{Colors.ENDC}")

def print_info(text):
    """Imprimir información"""
    print(f"{Colors.OKBLUE}ℹ {text}{Colors.ENDC}")

class SetupWizard:
    """Asistente de configuración del proyecto"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.errors = []
        self.warnings = []
        
    def run(self):
        """Ejecutar el proceso completo de setup"""
        print_header("SETUP INICIAL - TRADINGBOT CUANTITATIVO MT5")
        
        # 1. Verificar sistema operativo
        self.check_os()
        
        # 2. Verificar versión de Python
        self.check_python_version()
        
        # 3. Crear estructura de directorios
        self.create_directories()
        
        # 4. Verificar archivo .env
        self.check_env_file()
        
        # 5. Verificar dependencias
        self.check_dependencies()
        
        # 6. Verificar MetaTrader 5
        self.check_mt5()
        
        # 7. Verificar bases de datos
        self.check_databases()
        
        # 8. Crear archivos de configuración adicionales
        self.create_config_files()
        
        # 9. Resumen
        self.print_summary()
        
    def check_os(self):
        """Verificar sistema operativo"""
        print_info("Verificando sistema operativo...")
        
        os_name = platform.system()
        os_version = platform.version()
        
        print(f"  Sistema: {os_name}")
        print(f"  Versión: {os_version}")
        
        if os_name != "Windows":
            self.warnings.append(
                "MetaTrader 5 solo funciona nativamente en Windows. "
                "En Linux/Mac necesitarás Wine o una máquina virtual."
            )
            print_warning("MetaTrader 5 requiere Windows")
        else:
            print_success("Sistema operativo compatible")
            
    def check_python_version(self):
        """Verificar versión de Python"""
        print_info("\nVerificando versión de Python...")
        
        version = sys.version_info
        print(f"  Python {version.major}.{version.minor}.{version.micro}")
        
        if version.major < 3 or (version.major == 3 and version.minor < 8):
            self.errors.append("Se requiere Python 3.8 o superior")
            print_error("Versión de Python no compatible")
        else:
            print_success("Versión de Python compatible")
            
    def create_directories(self):
        """Crear estructura de directorios necesaria"""
        print_info("\nCreando estructura de directorios...")
        
        directories = [
            "logs",
            "data_storage/market_data",
            "data_storage/model_artifacts", 
            "data_storage/backtest_results",
            "reports",
            "reports/daily",
            "reports/weekly",
            "reports/monthly",
            "cache",
            "temp",
            "backups",
            "config/certificates",
            "config/strategies",
            "scripts/migrations",
            "scripts/maintenance"
        ]
        
        for directory in directories:
            dir_path = self.project_root / directory
            if not dir_path.exists():
                dir_path.mkdir(parents=True, exist_ok=True)
                print_success(f"Creado: {directory}")
            else:
                print(f"  Ya existe: {directory}")
                
    def check_env_file(self):
        """Verificar archivo .env"""
        print_info("\nVerificando archivo de configuración...")
        
        env_file = self.project_root / ".env"
        env_example = self.project_root / ".env.example"
        
        if not env_file.exists():
            if env_example.exists():
                self.warnings.append(
                    "No se encontró archivo .env. "
                    "Copia .env.example a .env y configura tus credenciales."
                )
                print_warning("Falta archivo .env")
            else:
                self.errors.append("No se encontró .env ni .env.example")
                print_error("Archivos de configuración faltantes")
        else:
            print_success("Archivo .env encontrado")
            
            # Verificar configuraciones críticas
            self.check_env_variables()
            
    def check_env_variables(self):
        """Verificar variables de entorno críticas"""
        print_info("  Verificando variables críticas...")
        
        try:
            from dotenv import load_dotenv
            load_dotenv()
            
            critical_vars = [
                "MT5_LOGIN",
                "MT5_PASSWORD",
                "MT5_SERVER",
                "DATABASE_URL",
                "REDIS_URL"
            ]
            
            missing_vars = []
            for var in critical_vars:
                if not os.getenv(var) or os.getenv(var).startswith("your_"):
                    missing_vars.append(var)
                    
            if missing_vars:
                self.warnings.append(
                    f"Variables sin configurar: {', '.join(missing_vars)}"
                )
                print_warning(f"  Configurar: {', '.join(missing_vars)}")
            else:
                print_success("  Variables críticas configuradas")
                
        except ImportError:
            self.warnings.append("python-dotenv no instalado")
            
    def check_dependencies(self):
        """Verificar dependencias principales"""
        print_info("\nVerificando dependencias principales...")
        
        dependencies = {
            "pandas": "Procesamiento de datos",
            "numpy": "Cálculos numéricos",
            "MetaTrader5": "Conexión con MT5",
            "sklearn": "Machine Learning",
            "tensorflow": "Deep Learning",
            "fastapi": "API REST",
            "streamlit": "Dashboard",
            "redis": "Cache y mensajería",
            "sqlalchemy": "Base de datos"
        }
        
        missing = []
        for module, description in dependencies.items():
            try:
                if module == "sklearn":
                    __import__("sklearn")
                else:
                    __import__(module)
                print(f"  ✓ {module}: {description}")
            except ImportError:
                missing.append(module)
                print_error(f"  ✗ {module}: {description}")
                
        if missing:
            self.warnings.append(
                f"Dependencias faltantes: {', '.join(missing)}. "
                "Ejecuta: pip install -r requirements.txt"
            )
            
    def check_mt5(self):
        """Verificar instalación de MetaTrader 5"""
        print_info("\nVerificando MetaTrader 5...")
        
        if platform.system() != "Windows":
            print_warning("  MT5 solo funciona en Windows")
            return
            
        try:
            import MetaTrader5 as mt5
            
            # Intentar inicializar
            if mt5.initialize():
                info = mt5.terminal_info()
                if info:
                    print_success(f"  MT5 conectado: {info.name}")
                    print(f"    Versión: {info.version}")
                    print(f"    Build: {info.build}")
                mt5.shutdown()
            else:
                self.warnings.append(
                    "No se pudo conectar con MT5. "
                    "Asegúrate de que esté instalado y abierto."
                )
                print_warning("  No se pudo conectar con MT5")
                
        except ImportError:
            self.errors.append("Módulo MetaTrader5 no instalado")
            print_error("  MetaTrader5 no instalado")
        except Exception as e:
            self.warnings.append(f"Error al verificar MT5: {str(e)}")
            print_error(f"  Error: {str(e)}")
            
    def check_databases(self):
        """Verificar conexión a bases de datos"""
        print_info("\nVerificando bases de datos...")
        
        # PostgreSQL
        try:
            import psycopg2
            db_url = os.getenv("DATABASE_URL", "")
            if db_url and not db_url.startswith("postgresql://tradingbot:secure_password"):
                # Intentar conexión básica
                print_info("  Verificando PostgreSQL...")
                print_success("  PostgreSQL: módulo disponible")
            else:
                print_warning("  PostgreSQL: configurar DATABASE_URL")
        except ImportError:
            self.warnings.append("psycopg2 no instalado")
            
        # Redis
        try:
            import redis
            redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
            if redis_url:
                print_info("  Verificando Redis...")
                print_success("  Redis: módulo disponible")
        except ImportError:
            self.warnings.append("redis-py no instalado")
            
    def create_config_files(self):
        """Crear archivos de configuración adicionales"""
        print_info("\nCreando archivos de configuración adicionales...")
        
        # Crear archivo de configuración de estrategias
        strategies_config = {
            "strategies": {
                "ml_strategy": {
                    "enabled": True,
                    "weight": 0.4,
                    "params": {
                        "confidence_threshold": 0.6,
                        "max_positions": 3
                    }
                },
                "mean_reversion": {
                    "enabled": True,
                    "weight": 0.3,
                    "params": {
                        "lookback_period": 20,
                        "z_score_threshold": 2.0
                    }
                },
                "momentum": {
                    "enabled": True,
                    "weight": 0.2,
                    "params": {
                        "fast_period": 12,
                        "slow_period": 26
                    }
                },
                "arbitrage": {
                    "enabled": True,
                    "weight": 0.1,
                    "params": {
                        "min_spread": 0.001
                    }
                }
            }
        }
        
        strategies_file = self.project_root / "config" / "strategies" / "default.json"
        strategies_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(strategies_file, 'w') as f:
            json.dump(strategies_config, f, indent=4)
            
        print_success("Creado: config/strategies/default.json")
        
        # Crear archivo de logging
        log_config = {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": {
                    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                },
                "detailed": {
                    "format": "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s"
                }
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "level": "INFO",
                    "formatter": "default",
                    "stream": "ext://sys.stdout"
                },
                "file": {
                    "class": "logging.handlers.RotatingFileHandler",
                    "level": "DEBUG",
                    "formatter": "detailed",
                    "filename": "logs/tradingbot.log",
                    "maxBytes": 10485760,
                    "backupCount": 5
                }
            },
            "loggers": {
                "tradingbot": {
                    "level": "DEBUG",
                    "handlers": ["console", "file"],
                    "propagate": False
                }
            }
        }
        
        log_config_file = self.project_root / "config" / "logging.json"
        with open(log_config_file, 'w') as f:
            json.dump(log_config, f, indent=4)
            
        print_success("Creado: config/logging.json")
        
    def print_summary(self):
        """Imprimir resumen del setup"""
        print_header("RESUMEN DE CONFIGURACIÓN")
        
        if self.errors:
            print(f"\n{Colors.FAIL}ERRORES ENCONTRADOS:{Colors.ENDC}")
            for error in self.errors:
                print(f"  • {error}")
                
        if self.warnings:
            print(f"\n{Colors.WARNING}ADVERTENCIAS:{Colors.ENDC}")
            for warning in self.warnings:
                print(f"  • {warning}")
                
        if not self.errors and not self.warnings:
            print_success("¡Configuración completada exitosamente!")
            
        print(f"\n{Colors.BOLD}PRÓXIMOS PASOS:{Colors.ENDC}")
        print("1. Configura las credenciales en el archivo .env")
        print("2. Instala las dependencias: pip install -r requirements.txt")
        print("3. Verifica la conexión con MT5: python -m tests.test_mt5_connection")
        print("4. Ejecuta en modo paper: python main.py --mode paper")
        print("5. Accede al dashboard: http://localhost:8501")
        
        print(f"\n{Colors.BOLD}DOCUMENTACIÓN:{Colors.ENDC}")
        print("• README.md - Documentación general")
        print("• docs/ - Documentación detallada")
        print("• notebooks/ - Ejemplos y análisis")
        
        # Crear archivo de estado del setup
        setup_status = {
            "timestamp": datetime.now().isoformat(),
            "errors": self.errors,
            "warnings": self.warnings,
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "platform": platform.system(),
            "setup_complete": len(self.errors) == 0
        }
        
        status_file = self.project_root / ".setup_status.json"
        with open(status_file, 'w') as f:
            json.dump(setup_status, f, indent=4)
            
        print(f"\n{Colors.OKBLUE}Estado del setup guardado en: .setup_status.json{Colors.ENDC}")

def main():
    """Función principal"""
    try:
        wizard = SetupWizard()
        wizard.run()
    except KeyboardInterrupt:
        print(f"\n{Colors.WARNING}Setup interrumpido por el usuario{Colors.ENDC}")
        sys.exit(1)
    except Exception as e:
        print(f"\n{Colors.FAIL}Error durante el setup: {str(e)}{Colors.ENDC}")
        sys.exit(1)

if __name__ == "__main__":
    main()
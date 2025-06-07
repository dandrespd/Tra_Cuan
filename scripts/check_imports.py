#!/usr/bin/env python3
"""
Verificación de importaciones después de las correcciones
=========================================================

Este script verifica que todos los módulos se puedan importar correctamente.
"""

import sys
import importlib
from pathlib import Path

# Agregar el directorio raíz al path
sys.path.insert(0, str(Path(__file__).parent.parent))

def check_imports():
    """Verificar que todos los módulos se puedan importar"""
    
    modules_to_check = [
        # Config
        "config.settings",
        "config.trading_config",
        "config.mt5_config",
        
        # Core
        "core.trading_bot",
        "core.mt5_connector",
        "core.trade_executor",
        
        # Data
        "data.data_collector",
        "data.data_processor",
        "data.data_validator",
        "data.feature_engineer",
        
        # Models
        "models.base_model",
        "models.ml_models",
        "models.deep_models",
        "models.ensemble_models",
        
        # Strategies
        "strategies.base_strategy",
        "strategies.ml_strategy",
        "strategies.adaptive_strategy",
        "strategies.hybrid_strategy",
        
        # Risk
        "risk.risk_manager",
        "risk.position_sizer",
        "risk.portfolio_optimizer",
        
        # Analysis
        "analysis.market_analyzer",
        "analysis.performance_analyzer",
        "analysis.economic_calendar",
        
        # Control
        "control.adaptive_controller",
        "control.ml_optimizer",
        "control.online_learning",
        
        # Utils
        "utils.helpers",
        "utils.log_config",
        "utils.notifications",
        
        # Visualization
        "visualization.charts",
        "visualization.dashboard",
        "visualization.reports",
    ]
    
    print("VERIFICACIÓN DE IMPORTACIONES")
    print("=" * 50)
    
    success_count = 0
    error_count = 0
    errors = []
    
    for module_name in modules_to_check:
        try:
            importlib.import_module(module_name)
            print(f"✓ {module_name}")
            success_count += 1
        except ImportError as e:
            print(f"✗ {module_name}: {str(e)}")
            error_count += 1
            errors.append((module_name, str(e)))
        except Exception as e:
            print(f"✗ {module_name}: Error inesperado - {str(e)}")
            error_count += 1
            errors.append((module_name, str(e)))
    
    print("\n" + "=" * 50)
    print(f"RESUMEN: {success_count} éxitos, {error_count} errores")
    
    if errors:
        print("\nERRORES DETALLADOS:")
        for module, error in errors:
            print(f"\n{module}:")
            print(f"  {error}")
            
        print("\nPOSIBLES SOLUCIONES:")
        print("1. Instalar dependencias: pip install -r requirements.txt")
        print("2. Verificar que todos los archivos __init__.py existan")
        print("3. Revisar errores de sintaxis en los módulos")
    else:
        print("\n✅ ¡Todas las importaciones funcionan correctamente!")
    
    return error_count == 0

if __name__ == "__main__":
    success = check_imports()
    sys.exit(0 if success else 1)
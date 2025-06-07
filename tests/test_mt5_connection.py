"""
Test de conexión con MetaTrader 5
==================================

Este script verifica que la conexión con MT5 funcione correctamente.
"""

import sys
import os
from pathlib import Path

# Agregar el directorio raíz al path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import MetaTrader5 as mt5
    print("✓ Módulo MetaTrader5 importado correctamente")
except ImportError:
    print("✗ Error: No se pudo importar MetaTrader5")
    print("  Instala con: pip install MetaTrader5")
    sys.exit(1)

def test_mt5_connection():
    """Probar conexión con MT5"""
    print("\n" + "="*50)
    print("TEST DE CONEXIÓN CON METATRADER 5")
    print("="*50 + "\n")
    
    # Intentar inicializar MT5
    print("1. Inicializando MT5...")
    if not mt5.initialize():
        print("✗ Error: No se pudo inicializar MT5")
        print("  Asegúrate de que MetaTrader 5 esté instalado y abierto")
        error = mt5.last_error()
        if error:
            print(f"  Código de error: {error}")
        return False
    
    print("✓ MT5 inicializado correctamente")
    
    # Obtener información del terminal
    print("\n2. Información del terminal:")
    terminal_info = mt5.terminal_info()
    if terminal_info:
        print(f"  - Nombre: {terminal_info.name}")
        print(f"  - Empresa: {terminal_info.company}")
        print(f"  - Versión: {terminal_info.version}")
        print(f"  - Build: {terminal_info.build}")
        print(f"  - Conectado: {'Sí' if terminal_info.connected else 'No'}")
        print(f"  - Cuenta demo: {'Sí' if terminal_info.trade_allowed else 'No'}")
    
    # Información de la cuenta
    print("\n3. Información de la cuenta:")
    account_info = mt5.account_info()
    if account_info:
        print(f"  - Login: {account_info.login}")
        print(f"  - Servidor: {account_info.server}")
        print(f"  - Nombre: {account_info.name}")
        print(f"  - Empresa: {account_info.company}")
        print(f"  - Balance: {account_info.balance} {account_info.currency}")
        print(f"  - Apalancamiento: 1:{account_info.leverage}")
        print(f"  - Modo: {'Demo' if account_info.trade_mode == mt5.ACCOUNT_TRADE_MODE_DEMO else 'Real'}")
    else:
        print("✗ No se pudo obtener información de la cuenta")
        print("  Verifica las credenciales en el archivo .env")
    
    # Listar símbolos disponibles
    print("\n4. Símbolos disponibles:")
    symbols = mt5.symbols_get()
    if symbols:
        print(f"  Total de símbolos: {len(symbols)}")
        # Mostrar algunos símbolos populares
        popular_symbols = ["EURUSD", "GBPUSD", "USDJPY", "XAUUSD", "BTCUSD"]
        available_popular = []
        for symbol in symbols:
            if symbol.name in popular_symbols:
                available_popular.append(symbol.name)
        
        if available_popular:
            print(f"  Símbolos populares disponibles: {', '.join(available_popular)}")
        
        # Mostrar primeros 10 símbolos
        print("  Primeros 10 símbolos:")
        for i, symbol in enumerate(symbols[:10]):
            print(f"    - {symbol.name}: {symbol.description}")
    
    # Probar obtención de datos
    print("\n5. Prueba de obtención de datos:")
    symbol = "EURUSD"
    
    # Verificar si el símbolo existe
    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        # Buscar un símbolo alternativo
        for s in symbols[:10]:
            if "EUR" in s.name or "USD" in s.name:
                symbol = s.name
                break
    
    # Obtener últimas cotizaciones
    tick = mt5.symbol_info_tick(symbol)
    if tick:
        print(f"  Cotización actual de {symbol}:")
        print(f"    - Bid: {tick.bid}")
        print(f"    - Ask: {tick.ask}")
        print(f"    - Spread: {round((tick.ask - tick.bid) * 10000, 1)} pips")
        print(f"    - Tiempo: {tick.time}")
        print("✓ Datos obtenidos correctamente")
    else:
        print(f"✗ No se pudieron obtener datos para {symbol}")
    
    # Cerrar conexión
    mt5.shutdown()
    print("\n✓ Conexión cerrada correctamente")
    
    print("\n" + "="*50)
    print("✓ TODAS LAS PRUEBAS COMPLETADAS EXITOSAMENTE")
    print("="*50)
    
    return True

def test_env_configuration():
    """Verificar configuración del archivo .env"""
    print("\nVerificando configuración .env...")
    
    try:
        from dotenv import load_dotenv
        load_dotenv()
        
        mt5_login = os.getenv("MT5_LOGIN")
        mt5_password = os.getenv("MT5_PASSWORD")
        mt5_server = os.getenv("MT5_SERVER")
        
        config_ok = True
        
        if not mt5_login or mt5_login == "50000000":
            print("⚠ MT5_LOGIN no configurado o usando valor por defecto")
            config_ok = False
            
        if not mt5_password or mt5_password == "your_secure_password":
            print("⚠ MT5_PASSWORD no configurado o usando valor por defecto")
            config_ok = False
            
        if not mt5_server or mt5_server == "ICMarkets-Demo":
            print("ℹ MT5_SERVER usando valor por defecto (ICMarkets-Demo)")
            
        if config_ok:
            print("✓ Configuración .env parece correcta")
            
            # Intentar login con credenciales
            if mt5.initialize():
                authorized = mt5.login(
                    login=int(mt5_login),
                    password=mt5_password,
                    server=mt5_server
                )
                if authorized:
                    print("✓ Login exitoso con credenciales del .env")
                else:
                    print("✗ No se pudo hacer login con las credenciales del .env")
                    error = mt5.last_error()
                    if error:
                        print(f"  Error: {error}")
                mt5.shutdown()
                
    except ImportError:
        print("⚠ python-dotenv no instalado, no se puede verificar .env")
    except Exception as e:
        print(f"✗ Error al verificar .env: {str(e)}")

if __name__ == "__main__":
    print("Sistema operativo:", sys.platform)
    print("Versión de Python:", sys.version)
    
    # Verificar configuración
    test_env_configuration()
    
    # Ejecutar test
    success = test_mt5_connection()
    
    if not success:
        print("\n⚠ SOLUCIÓN DE PROBLEMAS:")
        print("1. Asegúrate de tener MetaTrader 5 instalado")
        print("2. Abre MetaTrader 5 antes de ejecutar este script")
        print("3. Verifica las credenciales en el archivo .env")
        print("4. Si usas Linux/Mac, necesitas Wine o una máquina virtual")
        sys.exit(1)
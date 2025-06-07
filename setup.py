"""
Setup configuration for TradingBot Cuantitative MT5
"""
import os
import sys
from pathlib import Path
from setuptools import setup, find_packages, Command
from setuptools.command.install import install
from setuptools.command.develop import develop


# Leer versión
def read_version():
    version_file = Path(__file__).parent / "tradingbot" / "__version__.py"
    with open(version_file, "r", encoding="utf-8") as f:
        exec(f.read())
    return locals()["__version__"]


# Leer README
def read_readme():
    readme_file = Path(__file__).parent / "README.md"
    if readme_file.exists():
        with open(readme_file, "r", encoding="utf-8") as f:
            return f.read()
    return ""


# Leer requirements
def read_requirements(filename="requirements.txt"):
    req_file = Path(__file__).parent / filename
    if req_file.exists():
        with open(req_file, "r", encoding="utf-8") as f:
            return [
                line.strip() 
                for line in f 
                if line.strip() and not line.startswith("#")
            ]
    return []


class PostInstallCommand(install):
    """Comandos post-instalación"""
    
    def run(self):
        install.run(self)
        self.post_install()
    
    def post_install(self):
        """Ejecutar tareas post-instalación"""
        print("\n" + "="*50)
        print("Trading Bot instalado exitosamente!")
        print("="*50)
        
        # Crear directorios necesarios
        self._create_directories()
        
        # Verificar dependencias del sistema
        self._check_system_dependencies()
        
        # Configurar TA-Lib si es necesario
        self._setup_talib()
        
        print("\nPara comenzar:")
        print("1. Copia .env.example a .env y configura tus credenciales")
        print("2. Ejecuta: tradingbot --help")
        print("3. O importa: from tradingbot import TradingBot")
        print("\n" + "="*50)
    
    def _create_directories(self):
        """Crear estructura de directorios necesaria"""
        directories = [
            "logs",
            "data_storage/market_data",
            "data_storage/model_artifacts",
            "data_storage/backtest_results",
            "reports",
            "notebooks"
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            print(f"✓ Directorio creado/verificado: {directory}")
    
    def _check_system_dependencies(self):
        """Verificar dependencias del sistema"""
        print("\nVerificando dependencias del sistema...")
        
        # Verificar Python version
        if sys.version_info < (3, 8):
            print("⚠ Advertencia: Python 3.8+ es recomendado")
        
        # Verificar sistema operativo para MT5
        if sys.platform != "win32":
            print("⚠ Advertencia: MetaTrader5 solo funciona nativamente en Windows")
            print("  Para Linux/Mac, considera usar Wine o una VM")
        
        # Verificar TA-Lib
        try:
            import talib
            print("✓ TA-Lib instalado correctamente")
        except ImportError:
            print("⚠ TA-Lib no encontrado. Instala los binarios del sistema:")
            print("  Windows: pip install TA-Lib-{version}-cp{python}-cp{python}-win_amd64.whl")
            print("  Linux: sudo apt-get install ta-lib")
            print("  MacOS: brew install ta-lib")


class DevelopCommand(develop):
    """Comandos para modo desarrollo"""
    
    def run(self):
        develop.run(self)
        self.setup_development()
    
    def setup_development(self):
        """Configurar ambiente de desarrollo"""
        print("\nConfigurando ambiente de desarrollo...")
        
        # Instalar pre-commit hooks
        os.system("pre-commit install")
        print("✓ Pre-commit hooks instalados")
        
        # Crear archivo de configuración de ejemplo
        example_config = Path(".env.example")
        if not Path(".env").exists() and example_config.exists():
            import shutil
            shutil.copy(".env.example", ".env")
            print("✓ Archivo .env creado desde .env.example")


class TestCommand(Command):
    """Comando personalizado para ejecutar tests"""
    
    description = "Ejecutar suite de tests"
    user_options = []
    
    def initialize_options(self):
        pass
    
    def finalize_options(self):
        pass
    
    def run(self):
        import subprocess
        errno = subprocess.call([sys.executable, "-m", "pytest"])
        raise SystemExit(errno)


# Configuración principal
setup(
    name="tradingbot-mt5",
    version=read_version(),
    author="Tu Nombre",
    author_email="tu.email@ejemplo.com",
    description="Sistema avanzado de trading algorítmico con ML para MetaTrader 5",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/tu-usuario/tradingbot-mt5",
    project_urls={
        "Bug Tracker": "https://github.com/tu-usuario/tradingbot-mt5/issues",
        "Documentation": "https://tradingbot-mt5.readthedocs.io/",
        "Source Code": "https://github.com/tu-usuario/tradingbot-mt5",
    },
    
    # Paquetes
    packages=find_packages(exclude=["tests", "tests.*", "notebooks", "docs"]),
    package_dir={"tradingbot": "."},
    
    # Incluir archivos adicionales
    package_data={
        "tradingbot": [
            "config/*.yaml",
            "config/*.json",
            "assets/*",
            "templates/*",
        ],
    },
    include_package_data=True,
    
    # Dependencias
    install_requires=read_requirements("requirements.txt"),
    extras_require={
        "dev": read_requirements("requirements-dev.txt"),
        "docs": [
            "sphinx>=7.2.6",
            "sphinx-rtd-theme>=2.0.0",
            "sphinx-autodoc-typehints>=1.25.2",
        ],
        "gpu": [
            "tensorflow[and-cuda]",
            "torch>=2.0.0+cu118",
        ],
    },
    
    # Clasificadores
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "Intended Audience :: Developers",
        "Topic :: Office/Business :: Financial :: Investment",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX :: Linux",
        "Framework :: AsyncIO",
    ],
    
    # Python version
    python_requires=">=3.8",
    
    # Entry points
    entry_points={
        "console_scripts": [
            "tradingbot=tradingbot.main:main",
            "tradingbot-backtest=tradingbot.tools.backtest:main",
            "tradingbot-optimize=tradingbot.tools.optimize:main",
            "tradingbot-dashboard=tradingbot.visualization.dashboard:main",
        ],
    },
    
    # Comandos personalizados
    cmdclass={
        "install": PostInstallCommand,
        "develop": DevelopCommand,
        "test": TestCommand,
    },
    
    # Otros metadatos
    license="MIT",
    keywords=[
        "trading", "algorithmic-trading", "quantitative-finance",
        "metatrader5", "mt5", "forex", "stocks", "machine-learning",
        "deep-learning", "backtesting", "risk-management"
    ],
    platforms=["Windows", "Linux", "MacOS"],
    zip_safe=False,
)


# Script de post-instalación adicional
if __name__ == "__main__":
    # Si se ejecuta directamente, mostrar información
    print("""
    Trading Bot Cuantitativo MT5
    ===========================
    
    Para instalar:
    -------------
    pip install .                  # Instalación normal
    pip install -e .              # Instalación en modo desarrollo
    pip install ".[dev]"          # Con dependencias de desarrollo
    pip install ".[gpu]"          # Con soporte GPU
    pip install ".[dev,gpu,docs]" # Todas las extras
    
    Para empaquetar:
    ---------------
    python setup.py sdist bdist_wheel
    
    Para publicar en PyPI:
    ---------------------
    python -m twine upload dist/*
    """)
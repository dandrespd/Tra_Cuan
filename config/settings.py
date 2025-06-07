# config/settings.py
import os
from pathlib import Path
from typing import Dict, Any
import json
from dataclasses import dataclass, field
from datetime import datetime

@dataclass
class Settings:
    """Configuración central del sistema de trading"""
    
    # Paths
    BASE_DIR: Path = Path(__file__).parent.parent
    DATA_DIR: Path = field(default_factory=lambda: Path(__file__).parent.parent / "data_storage")
    LOGS_DIR: Path = field(default_factory=lambda: Path(__file__).parent.parent / "logs")
    MODELS_DIR: Path = field(default_factory=lambda: Path(__file__).parent.parent / "data_storage" / "model_artifacts")
    
    # Trading Parameters
    INITIAL_BALANCE: float = 10000.0
    MAX_POSITIONS: int = 3
    RISK_PER_TRADE: float = 0.02  # 2% risk per trade
    MAX_DAILY_LOSS: float = 0.05  # 5% max daily loss
    
    # Model Parameters
    CONFIDENCE_THRESHOLD: float = 0.65
    RETRAINING_INTERVAL: int = 1000  # trades
    MODEL_UPDATE_FREQUENCY: str = "daily"
    
    # System Parameters
    DEBUG: bool = True
    NOTIFICATION_ENABLED: bool = True
    DASHBOARD_ENABLED: bool = True
    
    # MT5 Connection
    MT5_TIMEOUT: int = 60000
    MT5_DEVIATION: int = 20
    
    def __post_init__(self):
        """Crear directorios necesarios"""
        for directory in [self.DATA_DIR, self.LOGS_DIR, self.MODELS_DIR]:
            directory.mkdir(parents=True, exist_ok=True)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir configuración a diccionario"""
        return {
            k: str(v) if isinstance(v, Path) else v 
            for k, v in self.__dict__.items()
        }
    
    def save(self, filepath: str = None):
        """Guardar configuración en archivo"""
        if filepath is None:
            filepath = self.BASE_DIR / "config" / "settings.json"
        
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=4)
    
    @classmethod
    def load(cls, filepath: str = None) -> 'Settings':
        """Cargar configuración desde archivo"""
        if filepath is None:
            filepath = Path(__file__).parent / "settings.json"
        
        if not Path(filepath).exists():
            return cls()
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Convertir paths de string a Path
        for key in ['BASE_DIR', 'DATA_DIR', 'LOGS_DIR', 'MODELS_DIR']:
            if key in data:
                data[key] = Path(data[key])
        
        return cls(**data)

# Instancia global de configuración
settings = Settings()
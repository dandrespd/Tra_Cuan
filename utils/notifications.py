from utils.log_config import get_logger
import smtplib, requests
from abc import ABC, abstractmethod
from typing import Dict, List, Optional
from datetime import datetime

class NotificationType(Enum):
    TRADE_OPEN = "trade_open"
    TRADE_CLOSE = "trade_close"
    RISK_ALERT = "risk_alert"
    ERROR = "error"
    DAILY_REPORT = "daily_report"
    SYSTEM_STATUS = "system_status"

class NotificationPriority(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

@dataclass
class Notification:
    type: NotificationType
    priority: NotificationPriority
    title: str
    message: str
    data: Dict[str, Any]
    timestamp: datetime
    
class NotificationChannel(ABC):
    @abstractmethod
    async def send(self, notification: Notification) -> bool:
        pass
        
class EmailChannel(NotificationChannel):
    """Canal de email con templates HTML"""
    
class TelegramChannel(NotificationChannel):
    """Canal de Telegram con formato Markdown"""
    
class DiscordChannel(NotificationChannel):
    """Canal de Discord con embeds ricos"""
    
class NotificationManager:
    """Gestor central de notificaciones"""
    def __init__(self):
        self.channels: Dict[str, NotificationChannel] = {}
        self.rate_limiter = RateLimiter()
        self.templates = TemplateEngine()
        self.queue = PriorityQueue()
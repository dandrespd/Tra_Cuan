import requests
from datetime import datetime, timezone
import pandas as pd
from typing import List, Dict, Optional
from utils.log_config import get_logger

@dataclass
class EconomicEvent:
    """Evento económico individual"""
    event_id: str
    name: str
    country: str
    currency: str
    importance: str  # 'low', 'medium', 'high'
    scheduled_time: datetime
    previous_value: Optional[float]
    forecast_value: Optional[float]
    actual_value: Optional[float]
    description: str
    
    @property
    def is_high_impact(self) -> bool:
        return self.importance == 'high'
    
    @property
    def minutes_until(self) -> int:
        """Minutos hasta el evento"""
        return int((self.scheduled_time - datetime.now()).total_seconds() / 60)

class EconomicCalendarAPI:
    """Interface para diferentes proveedores de calendario"""
    
    def __init__(self, provider: str = "forexfactory"):
        self.provider = provider
        self.api_clients = {
            'forexfactory': ForexFactoryClient(),
            'investing': InvestingComClient(),
            'fxstreet': FXStreetClient()
        }
        
    def fetch_events(self, start_date: datetime, end_date: datetime,
                    currencies: List[str] = None) -> List[EconomicEvent]:
        """Obtiene eventos del proveedor seleccionado"""
        client = self.api_clients.get(self.provider)
        
        if not client:
            raise ValueError(f"Proveedor no soportado: {self.provider}")
        
        events = client.fetch_events(start_date, end_date)
        
        # Filtrar por currencies si se especifica
        if currencies:
            events = [e for e in events if e.currency in currencies]
        
        return events

class ForexFactoryClient:
    """Cliente para Forex Factory"""
    
    def fetch_events(self, start_date: datetime, end_date: datetime) -> List[EconomicEvent]:
        """Scraping de Forex Factory (con rate limiting)"""
        # Implementar web scraping responsable
        # Usar BeautifulSoup o Selenium
        # Cachear resultados
        
class EconomicCalendar:
    """Gestor principal del calendario económico"""
    
    def __init__(self, currencies: List[str], cache_dir: Path = None):
        self.currencies = currencies
        self.cache_dir = cache_dir or Path("data_storage/economic_calendar")
        self.cache_dir.mkdir(exist_ok=True)
        
        self.api = EconomicCalendarAPI()
        self.events_cache = {}
        self.impact_analyzer = EventImpactAnalyzer()
        
    def get_upcoming_events(self, hours_ahead: int = 24) -> List[EconomicEvent]:
        """Obtiene eventos próximos"""
        start = datetime.now()
        end = start + timedelta(hours=hours_ahead)
        
        # Verificar cache primero
        cache_key = f"{start.date()}_{end.date()}"
        
        if cache_key in self.events_cache:
            events = self.events_cache[cache_key]
        else:
            events = self.api.fetch_events(start, end, self.currencies)
            self.events_cache[cache_key] = events
            
            # Guardar en disco
            self._save_to_cache(events, cache_key)
        
        # Filtrar eventos pasados
        return [e for e in events if e.scheduled_time > datetime.now()]
    
    def should_restrict_trading(self, symbol: str, 
                              minutes_before: int = 30,
                              minutes_after: int = 30) -> Tuple[bool, Optional[str]]:
        """Determina si se debe restringir trading por eventos próximos"""
        
        # Obtener currencies del símbolo
        base_currency = symbol[:3]
        quote_currency = symbol[3:6]
        
        # Verificar eventos próximos
        upcoming = self.get_upcoming_events(hours_ahead=1)
        
        for event in upcoming:
            if event.currency not in [base_currency, quote_currency]:
                continue
            
            if not event.is_high_impact:
                continue
            
            # Verificar ventana de tiempo
            minutes_to_event = event.minutes_until
            
            if -minutes_after <= minutes_to_event <= minutes_before:
                return True, f"Evento de alto impacto: {event.name} en {minutes_to_event} minutos"
        
        return False, None
    
    def analyze_historical_impact(self, event_type: str, symbol: str) -> Dict[str, Any]:
        """Analiza impacto histórico de tipos de eventos"""
        return self.impact_analyzer.analyze_event_impact(event_type, symbol)

class EventImpactAnalyzer:
    """Analiza el impacto histórico de eventos en precios"""
    
    def __init__(self):
        self.impact_database = {}
        
    def analyze_event_impact(self, event_type: str, symbol: str, 
                           lookback_events: int = 50) -> Dict[str, Any]:
        """Analiza cómo eventos pasados afectaron al precio"""
        
        # Cargar eventos históricos
        historical_events = self._load_historical_events(event_type, lookback_events)
        
        impact_stats = {
            'avg_volatility_increase': 0,
            'avg_pip_movement': 0,
            'directional_bias': 0,  # % de veces que el precio subió
            'avg_spread_widening': 0,
            'recovery_time_minutes': 0
        }
        
        for event in historical_events:
            # Analizar movimiento de precio alrededor del evento
            price_impact = self._analyze_price_movement(
                symbol, event.scheduled_time, 
                minutes_before=60, minutes_after=60
            )
            
            # Actualizar estadísticas
            impact_stats['avg_volatility_increase'] += price_impact['volatility_ratio']
            impact_stats['avg_pip_movement'] += price_impact['max_movement_pips']
            
            if price_impact['direction'] > 0:
                impact_stats['directional_bias'] += 1
        
        # Promediar resultados
        n_events = len(historical_events)
        if n_events > 0:
            for key in impact_stats:
                if key != 'directional_bias':
                    impact_stats[key] /= n_events
                else:
                    impact_stats[key] = impact_stats[key] / n_events * 100  # Porcentaje
        
        return impact_stats

class TradingFilter:
    """Filtros de trading basados en calendario económico"""
    
    def __init__(self, calendar: EconomicCalendar):
        self.calendar = calendar
        self.active_filters = {
            'pre_event': True,
            'post_event': True,
            'high_impact_only': True,
            'multiple_events': True  # Múltiples eventos simultáneos
        }
        
    def can_trade(self, symbol: str, strategy_type: str = None) -> Tuple[bool, str]:
        """Determina si se puede operar según filtros activos"""
        
        # Verificar restricción por eventos
        restricted, reason = self.calendar.should_restrict_trading(symbol)
        
        if restricted and self.active_filters['pre_event']:
            return False, reason
        
        # Verificar múltiples eventos
        if self.active_filters['multiple_events']:
            upcoming = self.calendar.get_upcoming_events(hours_ahead=2)
            high_impact_count = sum(1 for e in upcoming if e.is_high_impact)
            
            if high_impact_count >= 3:
                return False, f"Múltiples eventos de alto impacto próximos: {high_impact_count}"
        
        # Verificar volatilidad post-evento
        if self.active_filters['post_event']:
            recent_events = self._get_recent_events(minutes=60)
            if recent_events:
                return False, "Período de volatilidad post-evento"
        
        return True, "Trading permitido"

class EventScheduler:
    """Programa acciones basadas en eventos económicos"""
    
    def __init__(self, calendar: EconomicCalendar):
        self.calendar = calendar
        self.scheduled_actions = []
        
    def schedule_pre_event_actions(self, event: EconomicEvent, 
                                 actions: List[Callable]):
        """Programa acciones antes de un evento"""
        for minutes_before, action in actions:
            scheduled_time = event.scheduled_time - timedelta(minutes=minutes_before)
            
            self.scheduled_actions.append({
                'time': scheduled_time,
                'action': action,
                'event': event,
                'type': 'pre_event'
            })
    
    def execute_scheduled_actions(self):
        """Ejecuta acciones programadas"""
        current_time = datetime.now()
        
        # Ejecutar acciones que ya es hora
        due_actions = [
            a for a in self.scheduled_actions 
            if a['time'] <= current_time
        ]
        
        for action_data in due_actions:
            try:
                action_data['action'](action_data['event'])
                self.scheduled_actions.remove(action_data)
            except Exception as e:
                logger.error(f"Error ejecutando acción programada: {e}")
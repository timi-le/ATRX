"""
Economic Calendar Data Loader for FX AI-Quant Trading System.

This module provides functionality to load economic calendar data from various sources
including APIs, web scraping, and mock data for testing purposes.
"""

import asyncio
import json
import aiohttp
import pandas as pd
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import asdict
import structlog
import re

from core.interfaces.macro_interfaces import (
    MacroDataProvider, 
    EconomicEvent, 
    Currency, 
    EventType, 
    ImpactLevel
)


class MockEconomicDataProvider(MacroDataProvider):
    """Mock economic data provider for testing and development."""
    
    def __init__(self, logger: Optional[structlog.stdlib.BoundLogger] = None):
        self.logger = logger or structlog.get_logger(__name__)
        self.events_cache: Dict[str, EconomicEvent] = {}
        self.historical_events: List[EconomicEvent] = []
        
        # Initialize with some realistic mock events
        self._generate_mock_historical_data()
    
    def _generate_mock_historical_data(self):
        """Generate realistic mock historical economic data."""
        base_time = datetime.now(timezone.utc) - timedelta(days=30)
        
        # Common economic indicators with typical values
        event_templates = [
            # US Events
            {
                "currency": Currency.USD,
                "event_type": EventType.EMPLOYMENT,
                "name": "Non-Farm Payrolls",
                "impact": ImpactLevel.HIGH,
                "unit": "K",
                "typical_forecast": 200.0,
                "volatility": 50.0
            },
            {
                "currency": Currency.USD,
                "event_type": EventType.INFLATION,
                "name": "Consumer Price Index (CPI)",
                "impact": ImpactLevel.HIGH,
                "unit": "%",
                "typical_forecast": 3.2,
                "volatility": 0.2
            },
            {
                "currency": Currency.USD,
                "event_type": EventType.GDP,
                "name": "Gross Domestic Product (GDP)",
                "impact": ImpactLevel.HIGH,
                "unit": "%",
                "typical_forecast": 2.1,
                "volatility": 0.3
            },
            {
                "currency": Currency.USD,
                "event_type": EventType.INTEREST_RATE,
                "name": "Federal Funds Rate",
                "impact": ImpactLevel.HIGH,
                "unit": "%",
                "typical_forecast": 5.25,
                "volatility": 0.25
            },
            
            # EUR Events
            {
                "currency": Currency.EUR,
                "event_type": EventType.INFLATION,
                "name": "Consumer Price Index (CPI)",
                "impact": ImpactLevel.HIGH,
                "unit": "%",
                "typical_forecast": 2.4,
                "volatility": 0.2
            },
            {
                "currency": Currency.EUR,
                "event_type": EventType.PMI,
                "name": "Manufacturing PMI",
                "impact": ImpactLevel.MEDIUM,
                "unit": "",
                "typical_forecast": 48.5,
                "volatility": 2.0
            },
            {
                "currency": Currency.EUR,
                "event_type": EventType.INTEREST_RATE,
                "name": "ECB Interest Rate Decision",
                "impact": ImpactLevel.HIGH,
                "unit": "%",
                "typical_forecast": 4.0,
                "volatility": 0.25
            },
            
            # GBP Events
            {
                "currency": Currency.GBP,
                "event_type": EventType.INFLATION,
                "name": "Consumer Price Index (CPI)",
                "impact": ImpactLevel.HIGH,
                "unit": "%",
                "typical_forecast": 4.2,
                "volatility": 0.3
            },
            {
                "currency": Currency.GBP,
                "event_type": EventType.EMPLOYMENT,
                "name": "Unemployment Rate",
                "impact": ImpactLevel.MEDIUM,
                "unit": "%",
                "typical_forecast": 4.2,
                "volatility": 0.2
            },
            
            # JPY Events
            {
                "currency": Currency.JPY,
                "event_type": EventType.INFLATION,
                "name": "Consumer Price Index (CPI)",
                "impact": ImpactLevel.HIGH,
                "unit": "%",
                "typical_forecast": 3.1,
                "volatility": 0.2
            },
            {
                "currency": Currency.JPY,
                "event_type": EventType.INTEREST_RATE,
                "name": "BoJ Interest Rate Decision",
                "impact": ImpactLevel.HIGH,
                "unit": "%",
                "typical_forecast": -0.1,
                "volatility": 0.1
            }
        ]
        
        import random
        random.seed(42)  # For reproducible mock data
        
        # Generate events for the past 30 days
        for day_offset in range(30):
            event_date = base_time + timedelta(days=day_offset)
            
            # Generate 1-3 events per day
            num_events = random.randint(1, 3)
            
            for i in range(num_events):
                template = random.choice(event_templates)
                
                # Add some time variation
                event_time = event_date.replace(
                    hour=random.randint(8, 16),
                    minute=random.choice([0, 30]),
                    second=0,
                    microsecond=0
                )
                
                # Generate forecast and actual values
                forecast = template["typical_forecast"] + random.gauss(0, template["volatility"] * 0.2)
                actual = forecast + random.gauss(0, template["volatility"] * 0.5)
                previous = forecast + random.gauss(0, template["volatility"] * 0.3)
                
                event = EconomicEvent(
                    event_id=f"mock_{template['currency'].value}_{template['event_type'].value}_{day_offset}_{i}",
                    timestamp=event_time,
                    currency=template["currency"],
                    event_type=template["event_type"],
                    name=template["name"],
                    impact=template["impact"],
                    actual=actual,
                    forecast=forecast,
                    previous=previous,
                    unit=template["unit"],
                    source="mock_provider"
                )
                
                self.historical_events.append(event)
                self.events_cache[event.event_id] = event
        
        self.logger.info(f"Generated {len(self.historical_events)} mock historical events")
    
    async def get_economic_calendar(
        self, 
        start_date: datetime, 
        end_date: datetime,
        currencies: Optional[List[Currency]] = None,
        impact_levels: Optional[List[ImpactLevel]] = None
    ) -> List[EconomicEvent]:
        """Fetch economic calendar events from mock data."""
        
        filtered_events = []
        
        for event in self.historical_events:
            # Date filter
            if not (start_date <= event.timestamp <= end_date):
                continue
            
            # Currency filter
            if currencies and event.currency not in currencies:
                continue
            
            # Impact level filter
            if impact_levels and event.impact not in impact_levels:
                continue
            
            filtered_events.append(event)
        
        self.logger.debug(
            f"Retrieved {len(filtered_events)} events",
            start_date=start_date.isoformat(),
            end_date=end_date.isoformat()
        )
        
        return sorted(filtered_events, key=lambda x: x.timestamp)
    
    async def get_live_events(self) -> List[EconomicEvent]:
        """Get live/real-time economic events (mock implementation)."""
        now = datetime.now(timezone.utc)
        
        # Return events from the last hour as "live"
        recent_events = [
            event for event in self.historical_events
            if (now - event.timestamp).total_seconds() < 3600
        ]
        
        return recent_events
    
    async def update_event_actual(self, event_id: str, actual_value: float) -> bool:
        """Update an event with actual released value."""
        if event_id in self.events_cache:
            self.events_cache[event_id].actual = actual_value
            self.logger.info(f"Updated event {event_id} with actual value {actual_value}")
            return True
        
        self.logger.warning(f"Event {event_id} not found for update")
        return False


class ForexFactoryDataProvider(MacroDataProvider):
    """Forex Factory economic calendar data provider (placeholder for real implementation)."""
    
    def __init__(self, logger: Optional[structlog.stdlib.BoundLogger] = None):
        self.logger = logger or structlog.get_logger(__name__)
        self.base_url = "https://www.forexfactory.com"
        self.session: Optional[aiohttp.ClientSession] = None
    
    async def _ensure_session(self):
        """Ensure aiohttp session is available."""
        if self.session is None:
            self.session = aiohttp.ClientSession()
    
    async def get_economic_calendar(
        self, 
        start_date: datetime, 
        end_date: datetime,
        currencies: Optional[List[Currency]] = None,
        impact_levels: Optional[List[ImpactLevel]] = None
    ) -> List[EconomicEvent]:
        """Fetch economic calendar from Forex Factory (placeholder)."""
        
        # This is a placeholder implementation
        # In production, this would scrape or use Forex Factory API
        
        self.logger.warning("ForexFactoryDataProvider is not yet implemented - using mock data")
        
        # Fallback to mock data for demonstration
        mock_provider = MockEconomicDataProvider(self.logger)
        return await mock_provider.get_economic_calendar(
            start_date, end_date, currencies, impact_levels
        )
    
    async def get_live_events(self) -> List[EconomicEvent]:
        """Get live events from Forex Factory (placeholder)."""
        self.logger.warning("ForexFactoryDataProvider live events not yet implemented")
        return []
    
    async def update_event_actual(self, event_id: str, actual_value: float) -> bool:
        """Update event with actual value (placeholder)."""
        self.logger.warning("ForexFactoryDataProvider update not yet implemented")
        return False
    
    async def close(self):
        """Close the aiohttp session."""
        if self.session:
            await self.session.close()


class TradingEconomicsProvider(MacroDataProvider):
    """Trading Economics API provider (placeholder for real implementation)."""
    
    def __init__(self, api_key: Optional[str] = None, logger: Optional[structlog.stdlib.BoundLogger] = None):
        self.api_key = api_key
        self.logger = logger or structlog.get_logger(__name__)
        self.base_url = "https://api.tradingeconomics.com"
        self.session: Optional[aiohttp.ClientSession] = None
    
    async def _ensure_session(self):
        """Ensure aiohttp session is available."""
        if self.session is None:
            headers = {}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            
            self.session = aiohttp.ClientSession(headers=headers)
    
    async def get_economic_calendar(
        self, 
        start_date: datetime, 
        end_date: datetime,
        currencies: Optional[List[Currency]] = None,
        impact_levels: Optional[List[ImpactLevel]] = None
    ) -> List[EconomicEvent]:
        """Fetch economic calendar from Trading Economics API (placeholder)."""
        
        if not self.api_key:
            self.logger.warning("No API key provided for Trading Economics - using mock data")
            mock_provider = MockEconomicDataProvider(self.logger)
            return await mock_provider.get_economic_calendar(
                start_date, end_date, currencies, impact_levels
            )
        
        # Placeholder for real API implementation
        self.logger.warning("TradingEconomicsProvider API integration not yet implemented")
        return []
    
    async def get_live_events(self) -> List[EconomicEvent]:
        """Get live events from Trading Economics API (placeholder)."""
        self.logger.warning("TradingEconomicsProvider live events not yet implemented")
        return []
    
    async def update_event_actual(self, event_id: str, actual_value: float) -> bool:
        """Update event with actual value (placeholder)."""
        self.logger.warning("TradingEconomicsProvider update not yet implemented")
        return False
    
    async def close(self):
        """Close the aiohttp session."""
        if self.session:
            await self.session.close()


class EconomicCalendarLoader:
    """Main economic calendar data loader that aggregates multiple providers."""
    
    def __init__(
        self, 
        providers: Optional[List[MacroDataProvider]] = None,
        logger: Optional[structlog.stdlib.BoundLogger] = None
    ):
        self.logger = logger or structlog.get_logger(__name__)
        self.providers = providers or [MockEconomicDataProvider(self.logger)]
        
        self.logger.info(f"Initialized with {len(self.providers)} data providers")
    
    async def get_events(
        self,
        start_date: datetime,
        end_date: datetime,
        currencies: Optional[List[Currency]] = None,
        impact_levels: Optional[List[ImpactLevel]] = None
    ) -> List[EconomicEvent]:
        """Get economic events from all providers and deduplicate."""
        
        all_events = []
        
        for provider in self.providers:
            try:
                events = await provider.get_economic_calendar(
                    start_date, end_date, currencies, impact_levels
                )
                all_events.extend(events)
                
                self.logger.debug(
                    f"Retrieved {len(events)} events from {provider.__class__.__name__}"
                )
                
            except Exception as e:
                self.logger.error(
                    f"Failed to get events from {provider.__class__.__name__}: {e}"
                )
        
        # Deduplicate events by event_id
        unique_events = {}
        for event in all_events:
            unique_events[event.event_id] = event
        
        final_events = list(unique_events.values())
        final_events.sort(key=lambda x: x.timestamp)
        
        self.logger.info(
            f"Retrieved {len(final_events)} unique events from {len(self.providers)} providers"
        )
        
        return final_events
    
    async def get_live_events(self) -> List[EconomicEvent]:
        """Get live events from all providers."""
        all_events = []
        
        for provider in self.providers:
            try:
                events = await provider.get_live_events()
                all_events.extend(events)
            except Exception as e:
                self.logger.error(
                    f"Failed to get live events from {provider.__class__.__name__}: {e}"
                )
        
        # Deduplicate
        unique_events = {}
        for event in all_events:
            unique_events[event.event_id] = event
        
        return list(unique_events.values())
    
    def add_provider(self, provider: MacroDataProvider):
        """Add a new data provider."""
        self.providers.append(provider)
        self.logger.info(f"Added provider: {provider.__class__.__name__}")
    
    def remove_provider(self, provider_class: type):
        """Remove a data provider by class."""
        self.providers = [
            p for p in self.providers 
            if not isinstance(p, provider_class)
        ]
        self.logger.info(f"Removed provider: {provider_class.__name__}")
    
    async def close(self):
        """Close all providers that support cleanup."""
        for provider in self.providers:
            if hasattr(provider, 'close'):
                try:
                    await provider.close()
                except Exception as e:
                    self.logger.error(f"Error closing provider {provider.__class__.__name__}: {e}")


# Utility functions
def parse_currency_from_text(text: str) -> List[Currency]:
    """Extract currency mentions from text."""
    currency_patterns = {
        Currency.USD: r'\b(USD|US\s*Dollar|Dollar|United\s*States)\b',
        Currency.EUR: r'\b(EUR|Euro|European|Eurozone)\b',
        Currency.GBP: r'\b(GBP|Pound|Sterling|British|UK|United\s*Kingdom)\b',
        Currency.JPY: r'\b(JPY|Yen|Japanese|Japan)\b',
        Currency.AUD: r'\b(AUD|Australian|Australia)\b',
        Currency.CAD: r'\b(CAD|Canadian|Canada)\b',
        Currency.CHF: r'\b(CHF|Swiss|Switzerland|Franc)\b',
        Currency.NZD: r'\b(NZD|New\s*Zealand)\b',
    }
    
    found_currencies = []
    text_upper = text.upper()
    
    for currency, pattern in currency_patterns.items():
        if re.search(pattern, text_upper, re.IGNORECASE):
            found_currencies.append(currency)
    
    return found_currencies


def map_event_name_to_type(event_name: str) -> EventType:
    """Map event name to EventType enum."""
    name_upper = event_name.upper()
    
    # Define keyword mappings
    type_keywords = {
        EventType.GDP: ['GDP', 'GROSS DOMESTIC PRODUCT', 'ECONOMIC GROWTH'],
        EventType.INFLATION: ['CPI', 'INFLATION', 'CONSUMER PRICE', 'PPI', 'PRODUCER PRICE'],
        EventType.EMPLOYMENT: ['UNEMPLOYMENT', 'PAYROLL', 'EMPLOYMENT', 'JOBS', 'JOBLESS'],
        EventType.INTEREST_RATE: ['INTEREST RATE', 'FEDERAL FUNDS', 'ECB RATE', 'BOJ RATE', 'BANK RATE'],
        EventType.PMI: ['PMI', 'PURCHASING MANAGERS', 'MANUFACTURING INDEX'],
        EventType.RETAIL_SALES: ['RETAIL SALES', 'CONSUMER SPENDING'],
        EventType.TRADE_BALANCE: ['TRADE BALANCE', 'IMPORTS', 'EXPORTS', 'CURRENT ACCOUNT'],
        EventType.CENTRAL_BANK: ['CENTRAL BANK', 'FOMC', 'ECB', 'BOJ', 'BOE'],
        EventType.CONSUMER_CONFIDENCE: ['CONSUMER CONFIDENCE', 'CONSUMER SENTIMENT'],
        EventType.MANUFACTURING: ['MANUFACTURING', 'INDUSTRIAL PRODUCTION'],
    }
    
    for event_type, keywords in type_keywords.items():
        for keyword in keywords:
            if keyword in name_upper:
                return event_type
    
    # Default fallback
    return EventType.MANUFACTURING 
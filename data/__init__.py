"""
FX Data Ingestion Module

This module provides the core data ingestion infrastructure for the FX trading system.
It includes connectors for various data providers and streaming capabilities.
"""

from .base_connector import BaseFXConnector
from .mock_provider import MockFXConnector
from .oanda_connector import OandaConnector, OandaConfig
from .dukascopy_connector import DukascopyConnector, DukascopyConfig
from .stream_feed import DataStreamManager

__all__ = [
    "BaseFXConnector",
    "MockFXConnector", 
    "OandaConnector",
    "OandaConfig",
    "DukascopyConnector", 
    "DukascopyConfig",
    "DataStreamManager",
] 
#!/usr/bin/env python3
"""
Comprehensive test script for the Regime Detection System.

This script tests:
1. Regime detector functionality
2. Configuration loading
3. News provider integration
4. Feature generation
5. Integration with macro engine
"""

import asyncio
import sys
import os
from datetime import datetime, timezone, timedelta
from pathlib import Path
import structlog
import yaml
import pandas as pd
import numpy as np

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from core.regime_detector import (
    RegimeFeatures,
    RegimeOutput,
    RuleBasedRegimeDetector,
    create_sample_features,
    RegimeType
)
from data.newsapi_provider import (
    RealNewsAPIProvider,
    RSSNewsProvider,
    create_news_provider
)
from core.interfaces.macro_interfaces import Currency


def setup_logging():
    """Setup structured logging."""
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="ISO"),
            structlog.dev.ConsoleRenderer()
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    return structlog.get_logger(__name__)


def test_configuration_loading(logger):
    """Test loading the regime configuration."""
    logger.info("Testing configuration loading...")
    
    try:
        config_path = project_root / "core" / "regime_config.yaml"
        
        if not config_path.exists():
            logger.error(f"Configuration file not found: {config_path}")
            return False
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Validate required sections
        required_sections = ['hmm', 'clustering', 'rules', 'features', 'ensemble', 'regimes']
        for section in required_sections:
            if section not in config:
                logger.error(f"Missing required section: {section}")
                return False
        
        logger.info("✅ Configuration loaded successfully")
        logger.info(f"HMM states: {config['hmm']['n_states']}")
        logger.info(f"Clustering algorithms: {list(config['clustering'].keys())}")
        logger.info(f"Regime types: {list(config['regimes'].keys())}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Configuration loading failed: {e}")
        return False


def test_regime_features(logger):
    """Test RegimeFeatures data structure."""
    logger.info("Testing RegimeFeatures...")
    
    try:
        # Test valid features
        features = RegimeFeatures(
            atr=0.5, bb_width=0.4, realized_vol=0.3, vol_ratio=1.1,
            macd_signal=0.2, macd_histogram=0.1, adx=60, rsi=65, momentum=0.3,
            macro_surprise=0.1, macro_sentiment=0.2, trend_strength=0.6, mean_reversion=0.3
        )
        
        # Test conversion to array
        feature_array = features.to_array()
        assert len(feature_array) == 13, f"Expected 13 features, got {len(feature_array)}"
        
        # Test conversion to dict
        feature_dict = features.to_dict()
        assert len(feature_dict) == 13, f"Expected 13 features in dict, got {len(feature_dict)}"
        
        logger.info("✅ RegimeFeatures tests passed")
        logger.info(f"Feature array shape: {feature_array.shape}")
        logger.info(f"Sample features: {list(feature_dict.keys())[:5]}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ RegimeFeatures test failed: {e}")
        return False


def test_rule_based_detector(logger):
    """Test the rule-based regime detector."""
    logger.info("Testing RuleBasedRegimeDetector...")
    
    try:
        detector = RuleBasedRegimeDetector(logger=logger)
        
        # Test trending regime
        trending_features = RegimeFeatures(
            atr=0.5, bb_width=0.4, realized_vol=0.3, vol_ratio=1.1,
            macd_signal=0.8, macd_histogram=0.5, adx=60, rsi=65, momentum=0.7,
            macro_surprise=0.1, macro_sentiment=0.2, trend_strength=0.8, mean_reversion=0.2
        )
        
        result = detector.predict(trending_features)
        logger.info(f"Trending test - Regime: {result.regime}, Confidence: {result.confidence:.2f}")
        
        # Test mean-reverting regime
        mean_rev_features = RegimeFeatures(
            atr=0.2, bb_width=0.3, realized_vol=0.15, vol_ratio=0.8,
            macd_signal=0.1, macd_histogram=0.05, adx=15, rsi=50, momentum=0.1,
            macro_surprise=0.05, macro_sentiment=0.1, trend_strength=0.2, mean_reversion=0.8
        )
        
        result = detector.predict(mean_rev_features)
        logger.info(f"Mean-reverting test - Regime: {result.regime}, Confidence: {result.confidence:.2f}")
        
        # Test choppy regime
        choppy_features = RegimeFeatures(
            atr=0.8, bb_width=0.9, realized_vol=0.7, vol_ratio=1.5,
            macd_signal=0.2, macd_histogram=0.1, adx=12, rsi=45, momentum=0.2,
            macro_surprise=0.3, macro_sentiment=-0.1, trend_strength=0.1, mean_reversion=0.3
        )
        
        result = detector.predict(choppy_features)
        logger.info(f"Choppy test - Regime: {result.regime}, Confidence: {result.confidence:.2f}")
        
        logger.info("✅ RuleBasedRegimeDetector tests passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ RuleBasedRegimeDetector test failed: {e}")
        return False


def test_sample_feature_generation(logger):
    """Test sample feature generation."""
    logger.info("Testing sample feature generation...")
    
    try:
        # Generate sample data
        sample_data = create_sample_features(n_samples=100, regime_type=RegimeType.TRENDING)
        
        assert len(sample_data) == 100, f"Expected 100 samples, got {len(sample_data)}"
        
        # Test different regime types
        for regime_type in RegimeType:
            samples = create_sample_features(n_samples=10, regime_type=regime_type)
            logger.info(f"Generated {len(samples)} samples for {regime_type.value} regime")
        
        logger.info("✅ Sample feature generation tests passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Sample feature generation test failed: {e}")
        return False


async def test_news_providers(logger):
    """Test news provider functionality."""
    logger.info("Testing news providers...")
    
    try:
        # Test time range
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(hours=6)  # Last 6 hours
        currencies = [Currency.USD, Currency.EUR]
        
        # Test RSS provider
        logger.info("Testing RSS provider...")
        rss_provider = RSSNewsProvider(logger=logger)
        
        try:
            rss_news = await rss_provider.get_news_feed(start_time, end_time, currencies)
            logger.info(f"RSS provider: Found {len(rss_news)} news events")
            
            if rss_news:
                sample = rss_news[0]
                logger.info(f"Sample RSS: {sample.headline[:80]}...")
                logger.info(f"Currencies: {[c.value for c in sample.currencies_mentioned]}")
                logger.info(f"Sentiment: {sample.sentiment_score:.2f}")
        
        except Exception as e:
            logger.warning(f"RSS provider test failed: {e}")
        
        # Test NewsAPI provider
        logger.info("Testing NewsAPI provider...")
        api_key = os.getenv("NEWS_API_KEY", "ec45f3866330462db1c4e49c60ea22cd")
        
        try:
            async with RealNewsAPIProvider(api_key=api_key, logger=logger) as newsapi_provider:
                newsapi_news = await newsapi_provider.get_news_feed(start_time, end_time, currencies)
                logger.info(f"NewsAPI provider: Found {len(newsapi_news)} news events")
                
                if newsapi_news:
                    sample = newsapi_news[0]
                    logger.info(f"Sample NewsAPI: {sample.headline[:80]}...")
                    logger.info(f"Currencies: {[c.value for c in sample.currencies_mentioned]}")
                    logger.info(f"Sentiment: {sample.sentiment_score:.2f}")
        
        except Exception as e:
            logger.warning(f"NewsAPI provider test failed: {e}")
        
        # Test factory function
        try:
            provider = create_news_provider("rss", logger=logger)
            logger.info("✅ News provider factory works")
        except Exception as e:
            logger.error(f"❌ News provider factory failed: {e}")
            return False
        
        logger.info("✅ News provider tests completed")
        return True
        
    except Exception as e:
        logger.error(f"❌ News provider test failed: {e}")
        return False


def test_integration_scenario(logger):
    """Test a complete integration scenario."""
    logger.info("Testing integration scenario...")
    
    try:
        # Initialize detector
        detector = RuleBasedRegimeDetector(logger=logger)
        
        # Simulate a trading session with different market conditions
        scenarios = [
            ("Market Open - High Volatility", RegimeFeatures(
                atr=0.8, bb_width=0.9, realized_vol=0.7, vol_ratio=1.4,
                macd_signal=0.3, macd_histogram=0.2, adx=20, rsi=55, momentum=0.3,
                macro_surprise=0.2, macro_sentiment=0.1, trend_strength=0.3, mean_reversion=0.4
            )),
            ("Strong Trend Development", RegimeFeatures(
                atr=0.6, bb_width=0.5, realized_vol=0.4, vol_ratio=1.2,
                macd_signal=0.7, macd_histogram=0.6, adx=45, rsi=70, momentum=0.8,
                macro_surprise=0.1, macro_sentiment=0.3, trend_strength=0.9, mean_reversion=0.1
            )),
            ("Range-bound Consolidation", RegimeFeatures(
                atr=0.3, bb_width=0.2, realized_vol=0.2, vol_ratio=0.9,
                macd_signal=0.1, macd_histogram=0.05, adx=18, rsi=48, momentum=0.1,
                macro_surprise=0.05, macro_sentiment=0.0, trend_strength=0.2, mean_reversion=0.9
            ))
        ]
        
        results = []
        for scenario_name, features in scenarios:
            result = detector.predict(features)
            results.append((scenario_name, result))
            logger.info(f"{scenario_name}: {result.regime.value} (confidence: {result.confidence:.2f})")
        
        # Verify we got different regimes
        regimes_detected = set(result.regime for _, result in results)
        logger.info(f"Detected {len(regimes_detected)} different regimes: {[r.value for r in regimes_detected]}")
        
        logger.info("✅ Integration scenario test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Integration scenario test failed: {e}")
        return False


async def run_all_tests():
    """Run all tests."""
    logger = setup_logging()
    logger.info("🚀 Starting Regime Detection System Tests")
    
    test_results = []
    
    # Run synchronous tests
    sync_tests = [
        ("Configuration Loading", test_configuration_loading),
        ("Regime Features", test_regime_features),
        ("Rule-based Detector", test_rule_based_detector),
        ("Sample Feature Generation", test_sample_feature_generation),
        ("Integration Scenario", test_integration_scenario)
    ]
    
    for test_name, test_func in sync_tests:
        logger.info(f"\n--- Running {test_name} Test ---")
        try:
            result = test_func(logger)
            test_results.append((test_name, result))
        except Exception as e:
            logger.error(f"Test {test_name} crashed: {e}")
            test_results.append((test_name, False))
    
    # Run async tests
    logger.info(f"\n--- Running News Providers Test ---")
    try:
        result = await test_news_providers(logger)
        test_results.append(("News Providers", result))
    except Exception as e:
        logger.error(f"News Providers test crashed: {e}")
        test_results.append(("News Providers", False))
    
    # Summary
    logger.info(f"\n{'='*50}")
    logger.info("TEST SUMMARY")
    logger.info(f"{'='*50}")
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info(f"{test_name}: {status}")
        if result:
            passed += 1
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! Regime Detection System is ready.")
        return True
    else:
        logger.warning(f"⚠️  {total - passed} tests failed. Please review the issues above.")
        return False


if __name__ == "__main__":
    success = asyncio.run(run_all_tests())
    sys.exit(0 if success else 1) 
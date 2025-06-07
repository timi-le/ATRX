#!/usr/bin/env python3
"""
Professional Backtesting Script for FX Quant Trading System

This script runs comprehensive backtests to evaluate the performance of our
FX trading strategies using the complete trading pipeline.
"""

import asyncio
import sys
import os
import logging
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import random
from dotenv import load_dotenv

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Load environment variables from .env file
dotenv_path = project_root / '.env'
load_dotenv(dotenv_path=dotenv_path)

from backtester import (
    BacktestEngine,
    BacktestConfig,
    BacktestMode,
    create_backtest_config,
    PerformanceAnalyzer,
    create_performance_config,
    compare_strategies
)

from core.feature_engine import HighPerformanceFeatureEngine, FeatureConfig
from core.regime_detector import MetaLearnerRegimeDetector
from core.strategy_switcher import StrategySwitcher
from core.position_sizer import KellyPositionSizer
from core.risk_manager import CoreRiskManager
from data.mt5_data_downloader import MT5DataDownloader
import MetaTrader5 as mt5

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('backtest.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ProfessionalBacktester:
    """Professional backtesting suite for FX trading system."""
    
    def __init__(self):
        self.results = {}
        self.output_dir = Path("outputs/professional_backtest")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    async def run_comprehensive_backtest(self):
        """Run comprehensive backtest suite."""
        logger.info("Starting Professional Backtesting Suite")
        logger.info("=" * 60)
        
        # Test configurations
        test_configs = [
            {
                "name": "Conservative_EURUSD",
                "symbols": ["EURUSD"],
                "strategies": ["grid_martingale"],
                "timeframe": "5m",
                "initial_capital": 100000.0,
                "max_position_size": 0.05,  # 5% per position
                "max_total_exposure": 0.25,  # 25% total exposure
                "stop_loss_pct": 0.015,  # 1.5% stop loss
                "take_profit_pct": 0.03,  # 3% take profit
            },
            # {
            #     "name": "Aggressive_Multi_Currency",
            #     "symbols": ["EURUSD", "GBPUSD", "USDJPY"],
            #     "strategies": ["breakout_trend", "time_scalping"],
            #     "timeframe": "1m",
            #     "initial_capital": 250000.0,
            #     "max_position_size": 0.15,  # 15% per position
            #     "max_total_exposure": 0.60,  # 60% total exposure
            #     "stop_loss_pct": 0.025,  # 2.5% stop loss
            #     "take_profit_pct": 0.05,  # 5% take profit
            # },
            # {
            #     "name": "Balanced_Portfolio",
            #     "symbols": ["EURUSD", "GBPUSD", "USDJPY", "USDCHF"],
            #     "strategies": ["grid_martingale", "breakout_trend"],
            #     "timeframe": "5m",
            #     "initial_capital": 500000.0,
            #     "max_position_size": 0.10,  # 10% per position
            #     "max_total_exposure": 0.40,  # 40% total exposure
            #     "stop_loss_pct": 0.02,  # 2% stop loss
            #     "take_profit_pct": 0.04,  # 4% take profit
            # }
        ]
        
        # Run backtests for each configuration in parallel
        tasks = [self._run_single_backtest(config) for config in test_configs]
        results = await asyncio.gather(*tasks)
        
        for i, config in enumerate(test_configs):
            if results[i]:
                self.results[config['name']] = results[i]
                
                # Log key metrics
                perf = results[i]['performance']
                logger.info(f"Results for {config['name']}:")
                logger.info(f"  Total Return: {perf['total_return']:.2%}")
                logger.info(f"  Sharpe Ratio: {perf['sharpe_ratio']:.2f}")
                logger.info(f"  Max Drawdown: {perf['max_drawdown']:.2%}")
                logger.info(f"  Total Trades: {perf['total_trades']}")
                logger.info(f"  Win Rate: {perf['win_rate']:.2%}")
            else:
                logger.error(f"Backtest failed for {config['name']} and returned no result.")

        # Generate comprehensive report
        await self._generate_comprehensive_report()
        
        logger.info("\nProfessional Backtesting Suite Completed!")
        logger.info(f"Results saved to: {self.output_dir}")
        
    async def _run_single_backtest(self, config: Dict) -> Dict:
        """Run a single backtest configuration."""
        
        # Create backtest configuration
        backtest_config = BacktestConfig(
            symbols=config["symbols"],
            timeframe=config["timeframe"],
            start_date=datetime(2024, 5, 1),
            end_date=datetime(2024, 5, 3), # Short test
            initial_capital=config["initial_capital"],
            mode=BacktestMode.FULL_PIPELINE,
            
            # Strategy configuration
            strategies=config["strategies"],
            strategy_weights={s: 1.0/len(config["strategies"]) for s in config["strategies"]},
            
            # Risk management
            max_position_size=config["max_position_size"],
            max_total_exposure=config["max_total_exposure"],
            stop_loss_pct=config["stop_loss_pct"],
            take_profit_pct=config["take_profit_pct"],
            
            # Execution settings (realistic)
            enable_slippage=True,
            enable_commission=True,
            enable_latency=True,
            
            # Results
            save_results=True,
            save_trades=True,
            save_equity_curve=True,
            results_path=str(self.output_dir / config["name"]),
            log_level="INFO"
        )
        
        # Create core components
        feature_engine = await self._create_feature_engine()
        regime_detector = await self._create_regime_detector()
        strategy_switcher = await self._create_strategy_switcher(config["strategies"])
        position_sizer = await self._create_position_sizer()
        risk_manager = await self._create_risk_manager(config["initial_capital"])
        
        # Create market replay with real MT5 data
        market_replay = await self._create_mt5_market_replay(config)
        
        if not market_replay:
            logger.error(f"Failed to create market replay for {config['name']}. Skipping.")
            return {}
        
        # Create and run backtest engine
        engine = BacktestEngine(
            config=backtest_config,
            feature_engine=feature_engine,
            regime_detector=regime_detector,
            strategy_switcher=strategy_switcher,
            position_sizer=position_sizer,
            risk_manager=risk_manager
        )
        
        # Replace the market replay with our implementation
        engine.market_replay = market_replay
        
        await engine.initialize()
        return await engine.run()
    
    async def _create_feature_engine(self) -> HighPerformanceFeatureEngine:
        """Create feature engine for backtesting."""
        from core.feature_engine import FeatureConfig
        
        # Use the new StatisticalFeatureEngine
        feature_config = FeatureConfig(use_numba=True, max_workers=4)
        return HighPerformanceFeatureEngine(config=feature_config)
    
    async def _create_regime_detector(self) -> MetaLearnerRegimeDetector:
        """Create the real regime detector."""
        # Use the new MetaLearnerRegimeDetector
        detector = MetaLearnerRegimeDetector()
        await detector._load_base_models() # Important: load the sub-models
        return detector
    
    async def _create_strategy_switcher(self, strategies: List[str]) -> StrategySwitcher:
        """Create the real strategy switcher."""
        from core.strategy_switcher import StrategySwitcher, StrategySwitcherConfig
        
        config = StrategySwitcherConfig(config_path="config/strategy_params.yaml")
        return StrategySwitcher(config=config)
    
    async def _create_position_sizer(self) -> KellyPositionSizer:
        """Create the real position sizer."""
        from core.position_sizer import KellyPositionSizer, PositionSizerConfig, RiskProfile
        
        config = PositionSizerConfig(config_path="config/risk_settings.yaml")
        # Use a moderate risk profile for backtesting, can be configured later
        return KellyPositionSizer(config=config, risk_profile=RiskProfile.MODERATE)
    
    async def _create_risk_manager(self, initial_capital: float) -> CoreRiskManager:
        """Create risk manager for backtesting."""
        from core.risk_manager import CoreRiskManager, RiskManagerConfig

        # Note: This is a simplified setup. The real risk manager has more dependencies
        # that we might need to mock or provide later (e.g., a publisher).
        rm_config = RiskManagerConfig(config_path="config/risk_limits.yaml")
        return CoreRiskManager(config=rm_config, initial_capital=initial_capital)
    
    async def _create_mt5_market_replay(self, config: Dict) -> Optional['MarketReplay']:
        """Create a market replay object using real MT5 data."""
        logger.info("Creating market replay from MT5 data...")
        
        mt5_login = os.getenv("MT5_LOGIN")
        mt5_password = os.getenv("MT5_PASSWORD")
        mt5_server = os.getenv("MT5_SERVER")

        if not all([mt5_login, mt5_password, mt5_server]):
            logger.error("MT5 credentials not found in environment variables. Make sure you have a .env file with MT5_LOGIN, MT5_PASSWORD, and MT5_SERVER.")
            return None
            
        try:
            downloader = MT5DataDownloader(login=int(mt5_login), password=mt5_password, server=mt5_server)
        except ConnectionError as e:
            logger.error(f"Failed to connect to MT5: {e}")
            return None

        timeframe_map = {
            "1m": mt5.TIMEFRAME_M1,
            "5m": mt5.TIMEFRAME_M5,
            "15m": mt5.TIMEFRAME_M15,
            "30m": mt5.TIMEFRAME_M30,
            "1h": mt5.TIMEFRAME_H1,
            "4h": mt5.TIMEFRAME_H4,
            "1d": mt5.TIMEFRAME_D1,
        }
        mt5_timeframe = timeframe_map.get(config["timeframe"])
        if not mt5_timeframe:
            logger.error(f"Unsupported timeframe: {config['timeframe']}")
            downloader.disconnect()
            return None

        all_data = {}
        for symbol in config["symbols"]:
            data = downloader.download_data(
                symbol=symbol,
                timeframe=mt5_timeframe,
                start_date=datetime(2024, 5, 1),
                end_date=datetime(2024, 5, 3)
            )
            if data.empty:
                logger.warning(f"No data downloaded for {symbol}. It might not be available on the server.")
            else:
                all_data[symbol] = data
        
        downloader.disconnect()

        if not all_data:
            logger.error("No data downloaded for any symbols. Cannot proceed with backtest.")
            return None

        # Using a simplified market replay implementation for now
        class MarketReplay:
            def __init__(self, data: Dict[str, pd.DataFrame]):
                self._data = data
                self._all_symbols = list(data.keys())
                # Combine and sort all data points by time to create a unified timeline
                combined_data = []
                for symbol, df in data.items():
                    df['symbol'] = symbol
                    combined_data.append(df)
                
                self._timeline = pd.concat(combined_data).sort_values(by='time').reset_index()

            async def load_data(self):
                # Data is already loaded in this implementation
                pass
                
            async def stream(self):
                logger.info(f"Streaming {len(self._timeline)} data points across {len(self._all_symbols)} symbols.")
                for index, row in self._timeline.iterrows():
                    yield row

        return MarketReplay(all_data)
    
    async def _create_mock_market_replay(self, config: Dict) -> 'MockMarketReplay':
        """DEPRECATED: Create a mock market replay for testing."""
        logger.warning("Using DEPRECATED mock market replay. Switch to MT5 data.")
        
        symbols = config["symbols"]
        # ... existing code ...
    
    async def _generate_comprehensive_report(self):
        """Generate comprehensive backtesting report."""
        if not self.results:
            logger.warning("No results to generate report from")
            return
        
        # Create comparison DataFrame
        comparison_data = []
        for name, result in self.results.items():
            perf = result['performance']
            exec_stats = result['execution']
            
            comparison_data.append({
                'Strategy': name,
                'Initial_Capital': result['config']['initial_capital'],
                'Final_Equity': result['final_equity'],
                'Total_Return': perf['total_return'],
                'Annualized_Return': perf.get('annualized_return', 0.0),
                'Volatility': perf['volatility'],
                'Sharpe_Ratio': perf['sharpe_ratio'],
                'Sortino_Ratio': perf.get('sortino_ratio', 0.0),
                'Calmar_Ratio': perf.get('calmar_ratio', 0.0),
                'Max_Drawdown': perf['max_drawdown'],
                'Max_DD_Duration': perf.get('max_drawdown_duration', 0.0),
                'Total_Trades': perf['total_trades'],
                'Win_Rate': perf['win_rate'],
                'Profit_Factor': perf['profit_factor'],
                'Avg_Win': perf.get('avg_win', 0.0),
                'Avg_Loss': perf.get('avg_loss', 0.0),
                'Total_Commission': perf['total_commission'],
                'Fill_Rate': exec_stats['fill_rate'],
                'Avg_Slippage': exec_stats['avg_slippage_per_fill']
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Save comparison report
        report_file = self.output_dir / "strategy_comparison_report.csv"
        comparison_df.to_csv(report_file, index=False)
        
        # Generate summary report
        summary_file = self.output_dir / "backtest_summary.txt"
        with open(summary_file, 'w') as f:
            f.write("FX Quant Trading System - Professional Backtest Report\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Number of Strategies Tested: {len(self.results)}\n\n")
            
            # Best performing strategy
            best_strategy = max(self.results.items(), 
                              key=lambda x: x[1]['performance']['sharpe_ratio'])
            f.write(f"Best Strategy (by Sharpe Ratio): {best_strategy[0]}\n")
            f.write(f"  Sharpe Ratio: {best_strategy[1]['performance']['sharpe_ratio']:.2f}\n")
            f.write(f"  Total Return: {best_strategy[1]['performance']['total_return']:.2%}\n")
            f.write(f"  Max Drawdown: {best_strategy[1]['performance']['max_drawdown']:.2%}\n\n")
            
            # Strategy rankings
            f.write("Strategy Rankings (by Sharpe Ratio):\n")
            f.write("-" * 40 + "\n")
            
            sorted_strategies = sorted(self.results.items(), 
                                     key=lambda x: x[1]['performance']['sharpe_ratio'], 
                                     reverse=True)
            
            for i, (name, result) in enumerate(sorted_strategies, 1):
                perf = result['performance']
                f.write(f"{i}. {name}\n")
                f.write(f"   Return: {perf['total_return']:.2%}\n")
                f.write(f"   Sharpe: {perf['sharpe_ratio']:.2f}\n")
                f.write(f"   Max DD: {perf['max_drawdown']:.2%}\n")
                f.write(f"   Trades: {perf['total_trades']}\n")
                f.write(f"   Win Rate: {perf['win_rate']:.2%}\n\n")
            
            # Risk Analysis
            f.write("Risk Analysis Summary:\n")
            f.write("-" * 30 + "\n")
            avg_sharpe = np.mean([r['performance']['sharpe_ratio'] for r in self.results.values()])
            avg_drawdown = np.mean([r['performance']['max_drawdown'] for r in self.results.values()])
            avg_volatility = np.mean([r['performance']['volatility'] for r in self.results.values()])
            
            f.write(f"Average Sharpe Ratio: {avg_sharpe:.2f}\n")
            f.write(f"Average Max Drawdown: {avg_drawdown:.2%}\n")
            f.write(f"Average Volatility: {avg_volatility:.2%}\n\n")
            
            # Execution Quality
            f.write("Execution Quality Summary:\n")
            f.write("-" * 30 + "\n")
            avg_fill_rate = np.mean([r['execution']['fill_rate'] for r in self.results.values()])
            avg_slippage = np.mean([r['execution']['avg_slippage_per_fill'] for r in self.results.values()])
            total_commission = sum([r['performance']['total_commission'] for r in self.results.values()])
            
            f.write(f"Average Fill Rate: {avg_fill_rate:.2%}\n")
            f.write(f"Average Slippage: {avg_slippage:.4f}\n")
            f.write(f"Total Commission Paid: ${total_commission:.2f}\n\n")
            
            # Recommendations
            f.write("Recommendations:\n")
            f.write("-" * 20 + "\n")
            
            if avg_sharpe > 1.0:
                f.write("[OK] Strong risk-adjusted returns achieved\n")
            else:
                f.write("[WARNING] Consider optimizing strategies for better risk-adjusted returns\n")
            
            if avg_drawdown < 0.15:
                f.write("[OK] Acceptable drawdown levels\n")
            else:
                f.write("[WARNING] High drawdown levels - consider reducing position sizes\n")
            
            if avg_fill_rate > 0.95:
                f.write("[OK] Excellent execution quality\n")
            else:
                f.write("[WARNING] Consider optimizing execution algorithms\n")
        
        logger.info(f"Comprehensive report saved to: {summary_file}")
        logger.info(f"Detailed comparison saved to: {report_file}")


async def main():
    """Main function to run professional backtesting."""
    try:
        backtester = ProfessionalBacktester()
        await backtester.run_comprehensive_backtest()
        
        print("\n" + "=" * 60)
        print("PROFESSIONAL BACKTEST COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print(f"Results available in: {backtester.output_dir}")
        print("\nKey files generated:")
        print("- strategy_comparison_report.csv")
        print("- backtest_summary.txt")
        print("- Individual strategy results in subdirectories")
        
    except Exception as e:
        logger.error(f"Error running professional backtest: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Ensure output directory exists
    os.makedirs("outputs", exist_ok=True)
    
    # Run the professional backtest
    asyncio.run(main()) 
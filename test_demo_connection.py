#!/usr/bin/env python3
"""
Demo Trading Connection Test
Test MT5 connection with user's demo account credentials
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))

try:
    pass

    import MetaTrader5 as mt5

    from services.execution.mt5_connector import create_mt5_connector

    def test_mt5_connection():
        print("🔌 Testing MT5 Demo Connection...")
        print("📦 MetaTrader5 package: ✅ Available")

        # Create connector
        try:
            connector = create_mt5_connector()
            print(f"📋 Config loaded:")
            print(f"   Login: {connector.config.login}")
            print(f"   Server: {connector.config.server}")
            print(f"   Magic Number: {connector.config.magic_number}")
        except Exception as e:
            print(f"❌ Failed to load config: {e}")
            return False

        # Test basic MT5 initialization
        print(f"🚀 Testing MT5 initialization...")
        if not mt5.initialize():
            error = mt5.last_error()
            print(f"❌ MT5 initialization failed: {error}")
            print("   Make sure MetaTrader 5 terminal is installed and running")
            return False

        print("✅ MT5 terminal initialized successfully")

        # Try to login
        print(f"🔐 Attempting login to {connector.config.server}...")
        login_result = mt5.login(
            connector.config.login,
            password=connector.config.password,
            server=connector.config.server,
        )

        if not login_result:
            error = mt5.last_error()
            print(f"❌ Login failed: {error}")
            print("   Check your credentials in config_mt5.yaml")
            print("   Make sure your demo account is active")
            mt5.shutdown()
            return False

        print("✅ Login successful!")

        # Get account info
        account_info = mt5.account_info()
        if account_info:
            print(f"📊 Account Info:")
            print(f"   Balance: ${account_info.balance:,.2f}")
            print(f"   Equity: ${account_info.equity:,.2f}")
            print(f"   Server: {account_info.server}")
            print(f"   Company: {account_info.company}")
            print(f"   Currency: {account_info.currency}")
            print(f"   Leverage: 1:{account_info.leverage}")
            print(f"   Trade Allowed: {account_info.trade_allowed}")
        else:
            print("⚠️ Could not retrieve account info")

        # Test symbol access
        print(f"📈 Testing symbol access...")
        symbol_info = mt5.symbol_info("EURUSD")
        if symbol_info:
            print(f"   EURUSD Bid: {symbol_info.bid:.5f}")
            print(f"   EURUSD Ask: {symbol_info.ask:.5f}")
            print(f"   Spread: {symbol_info.spread} points")
            print(f"   Trading Allowed: {symbol_info.visible}")
        else:
            print("⚠️ Could not access EURUSD symbol info")

        # Test getting available symbols
        symbols = mt5.symbols_get()
        if symbols:
            print(f"   Available symbols: {len(symbols)} total")
            major_pairs = [
                s.name
                for s in symbols
                if s.name in ["EURUSD", "GBPUSD", "USDJPY", "USDCHF"]
            ]
            print(f"   Major pairs available: {major_pairs}")
        else:
            print("⚠️ Could not retrieve symbols list")

        # Test market hours
        print(f"🕐 Testing market status...")
        terminal_info = mt5.terminal_info()
        if terminal_info:
            print(f"   Terminal: {terminal_info.name}")
            print(f"   Build: {terminal_info.build}")
            print(f"   Connected: {terminal_info.connected}")
            print(f"   Trade Allowed: {terminal_info.trade_allowed}")

        mt5.shutdown()
        print("🎉 Demo connection test PASSED! Ready for trading.")
        return True

    if __name__ == "__main__":
        success = test_mt5_connection()
        if success:
            print("\n✅ All tests passed! Your demo account is ready.")
            print("💡 Next steps:")
            print("   1. Run demo trades with the execution engine")
            print("   2. Monitor real-time positions and orders")
            print("   3. Test different order types and algorithms")
        else:
            print("\n❌ Connection test failed. Please check:")
            print("   1. MetaTrader 5 terminal is installed and running")
            print("   2. Demo account credentials are correct")
            print("   3. Internet connection is stable")
            print("   4. Demo account is active and not expired")

except ImportError as e:
    print(f"❌ Import error: {e}")
    print("   Run: pip install MetaTrader5")
except Exception as e:
    print(f"❌ Connection test failed: {str(e)}")
    import traceback

    traceback.print_exc()

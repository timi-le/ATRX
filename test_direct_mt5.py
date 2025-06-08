#!/usr/bin/env python3
"""
Direct MT5 Connection Test
Test MT5 connection directly with hardcoded credentials
"""

import MetaTrader5 as mt5
import yaml


def test_direct_connection():
    print("🔌 Direct MT5 Connection Test...")

    # Load config directly
    with open("services/execution/config_mt5.yaml") as f:
        config = yaml.safe_load(f)

    mt5_config = config["mt5"]
    login = mt5_config["login"]
    password = mt5_config["password"]
    server = mt5_config["server"]

    print(f"📋 Loaded credentials:")
    print(f"   Login: {login}")
    print(f"   Password: {password}")
    print(f"   Server: {server}")

    # Test MT5 initialization
    print(f"🚀 Initializing MT5...")
    if not mt5.initialize():
        error = mt5.last_error()
        print(f"❌ MT5 initialization failed: {error}")
        return False

    print("✅ MT5 initialized successfully")

    # Test login
    print(f"🔐 Attempting login...")
    login_result = mt5.login(login, password=password, server=server)

    if not login_result:
        error = mt5.last_error()
        print(f"❌ Login failed: {error}")
        print("   Possible issues:")
        print("   - Demo account expired or inactive")
        print("   - Wrong server name")
        print("   - Wrong credentials")
        print("   - Network connectivity issues")
        mt5.shutdown()
        return False

    print("✅ Login successful!")

    # Get account info
    account_info = mt5.account_info()
    if account_info:
        print(f"📊 Account Details:")
        print(f"   Balance: ${account_info.balance:,.2f}")
        print(f"   Equity: ${account_info.equity:,.2f}")
        print(f"   Server: {account_info.server}")
        print(f"   Company: {account_info.company}")
        print(f"   Currency: {account_info.currency}")
        print(f"   Leverage: 1:{account_info.leverage}")
        print(f"   Trade Allowed: {account_info.trade_allowed}")
        print(f"   Expert Advisors Allowed: {account_info.trade_expert}")

    # Test symbol access
    print(f"📈 Testing symbol access...")
    symbol_info = mt5.symbol_info("EURUSD")
    if symbol_info:
        print(f"   EURUSD:")
        print(f"     Bid: {symbol_info.bid:.5f}")
        print(f"     Ask: {symbol_info.ask:.5f}")
        print(f"     Spread: {symbol_info.spread} points")
        print(f"     Visible: {symbol_info.visible}")
        print(f"     Trade Mode: {symbol_info.trade_mode}")

    # Test terminal info
    terminal_info = mt5.terminal_info()
    if terminal_info:
        print(f"🖥️ Terminal Info:")
        print(f"   Name: {terminal_info.name}")
        print(f"   Build: {terminal_info.build}")
        print(f"   Connected: {terminal_info.connected}")
        print(f"   Trade Allowed: {terminal_info.trade_allowed}")
        print(f"   Expert Advisors Enabled: {terminal_info.tradeapi_disabled}")

    mt5.shutdown()
    print("🎉 Direct connection test completed successfully!")
    return True


if __name__ == "__main__":
    success = test_direct_connection()
    if success:
        print("\n✅ Demo account is working! Ready for trading.")
    else:
        print("\n❌ Connection failed. Please check your demo account status.")

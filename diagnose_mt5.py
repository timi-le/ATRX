#!/usr/bin/env python3
"""
MT5 Connection Diagnostics
Comprehensive diagnostics for MT5 connection issues
"""

import MetaTrader5 as mt5
import yaml
import socket
import subprocess
import sys
from pathlib import Path

def check_mt5_installation():
    """Check if MT5 is properly installed."""
    print("🔍 Checking MT5 Installation...")
    
    try:
        # Check if MT5 package is available
        import MetaTrader5
        print(f"✅ MetaTrader5 Python package: v{MetaTrader5.__version__}")
    except ImportError:
        print("❌ MetaTrader5 Python package not installed")
        return False
    
    # Check common MT5 installation paths
    common_paths = [
        "C:\\Program Files\\MetaTrader 5\\terminal64.exe",
        "C:\\Program Files (x86)\\MetaTrader 5\\terminal64.exe",
        "C:\\Users\\%USERNAME%\\AppData\\Roaming\\MetaQuotes\\Terminal\\*\\terminal64.exe"
    ]
    
    mt5_found = False
    for path in common_paths:
        if Path(path.replace("%USERNAME%", Path.home().name)).exists():
            print(f"✅ MT5 Terminal found: {path}")
            mt5_found = True
            break
    
    if not mt5_found:
        print("⚠️ MT5 Terminal not found in common locations")
        print("   Please ensure MetaTrader 5 is installed")
    
    return True

def check_network_connectivity():
    """Check network connectivity to common MT5 servers."""
    print("\n🌐 Checking Network Connectivity...")
    
    test_hosts = [
        ("google.com", 80),
        ("mt5.exness.com", 443),
        ("exness.com", 443)
    ]
    
    for host, port in test_hosts:
        try:
            socket.create_connection((host, port), timeout=5)
            print(f"✅ Connection to {host}:{port} successful")
        except Exception as e:
            print(f"❌ Connection to {host}:{port} failed: {e}")

def test_mt5_initialization():
    """Test MT5 initialization."""
    print("\n🚀 Testing MT5 Initialization...")
    
    if not mt5.initialize():
        error = mt5.last_error()
        print(f"❌ MT5 initialization failed: {error}")
        print("   Possible solutions:")
        print("   1. Start MetaTrader 5 terminal manually")
        print("   2. Enable 'Allow DLL imports' in MT5 settings")
        print("   3. Enable 'Allow automated trading' in MT5 settings")
        return False
    
    print("✅ MT5 initialization successful")
    
    # Get terminal info
    terminal_info = mt5.terminal_info()
    if terminal_info:
        print(f"   Terminal: {terminal_info.name}")
        print(f"   Build: {terminal_info.build}")
        print(f"   Connected: {terminal_info.connected}")
        print(f"   Trade Allowed: {terminal_info.trade_allowed}")
        print(f"   Expert Advisors: {'Enabled' if not terminal_info.tradeapi_disabled else 'Disabled'}")
    
    return True

def test_server_connection():
    """Test connection to different servers."""
    print("\n🔐 Testing Server Connections...")
    
    # Load config
    with open('services/execution/config_mt5.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    mt5_config = config['mt5']
    login = mt5_config['login']
    password = mt5_config['password']
    server = mt5_config['server']
    
    print(f"📋 Testing with credentials:")
    print(f"   Login: {login}")
    print(f"   Server: {server}")
    
    # Test the configured server
    print(f"\n🔌 Attempting login to {server}...")
    login_result = mt5.login(login, password=password, server=server)
    
    if login_result:
        print("✅ Login successful!")
        account_info = mt5.account_info()
        if account_info:
            print(f"   Balance: ${account_info.balance:,.2f}")
            print(f"   Server: {account_info.server}")
            print(f"   Company: {account_info.company}")
        return True
    else:
        error = mt5.last_error()
        print(f"❌ Login failed: {error}")
        
        # Try alternative server names
        alternative_servers = [
            "Exness-MT5Trial",
            "Exness-MT5Trial1",
            "Exness-MT5Trial2", 
            "Exness-MT5Trial3",
            "Exness-MT5Trial4",
            "Exness-MT5Trial5",
            "Exness-MT5Trial6",
            "Exness-MT5Trial7",
            "Exness-MT5Trial8",
            "Exness-MT5Trial10",
            "Exness-Demo",
            "ExnessTrial"
        ]
        
        print(f"\n🔄 Trying alternative server names...")
        for alt_server in alternative_servers:
            print(f"   Testing: {alt_server}")
            if mt5.login(login, password=password, server=alt_server):
                print(f"✅ Success with server: {alt_server}")
                print(f"   Update your config to use: {alt_server}")
                account_info = mt5.account_info()
                if account_info:
                    print(f"   Balance: ${account_info.balance:,.2f}")
                return True
        
        print("❌ All server attempts failed")
        return False

def provide_troubleshooting_tips():
    """Provide troubleshooting tips."""
    print("\n💡 Troubleshooting Tips:")
    print("1. **Demo Account Status:**")
    print("   - Demo accounts typically expire after 30 days")
    print("   - Check if your account is still active")
    print("   - Create a new demo account if expired")
    
    print("\n2. **MetaTrader 5 Settings:**")
    print("   - Open MT5 terminal manually")
    print("   - Go to Tools > Options > Expert Advisors")
    print("   - Enable 'Allow automated trading'")
    print("   - Enable 'Allow DLL imports'")
    
    print("\n3. **Server Connection:**")
    print("   - Verify server name in MT5 terminal")
    print("   - Check File > Login to Account for correct server")
    print("   - Try logging in manually first")
    
    print("\n4. **Firewall/Antivirus:**")
    print("   - Check if firewall is blocking MT5")
    print("   - Add MT5 to antivirus exceptions")
    print("   - Temporarily disable firewall for testing")

def main():
    """Run comprehensive diagnostics."""
    print("🔧 MT5 Connection Diagnostics")
    print("=" * 50)
    
    # Run all diagnostic checks
    check_mt5_installation()
    check_network_connectivity()
    
    if test_mt5_initialization():
        success = test_server_connection()
        mt5.shutdown()
        
        if success:
            print("\n🎉 Connection successful! Your demo account is working.")
            print("💡 You can now proceed with demo trading.")
        else:
            provide_troubleshooting_tips()
    else:
        print("\n❌ MT5 initialization failed. Please check MT5 installation.")

if __name__ == "__main__":
    main() 
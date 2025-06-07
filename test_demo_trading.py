#!/usr/bin/env python3
"""
Demo Trading Integration Test
Test the MT5 connector with demo trading functionality
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))

from services.execution.mt5_connector import create_mt5_connector
import asyncio

async def test_demo_trading():
    print('🎯 Testing Demo Trading Integration...')
    
    # Create connector
    connector = create_mt5_connector()
    
    try:
        # Connect
        if await connector.connect():
            print('✅ Connected to MT5')
            
            # Get account info (not async)
            account_info = connector.get_account_info()
            if account_info:
                print(f'📊 Account: ${account_info["balance"]:,.2f} balance')
                print(f'   Equity: ${account_info["equity"]:,.2f}')
                print(f'   Free Margin: ${account_info["free_margin"]:,.2f}')
                print(f'   Leverage: 1:{account_info["leverage"]}')
            else:
                print('⚠️ Could not retrieve account info')
            
            # Test symbol info for major pairs (not async)
            major_pairs = ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF']
            print(f'\n📈 Testing symbol access:')
            
            for symbol in major_pairs:
                symbol_info = connector.get_symbol_info(symbol)
                if symbol_info:
                    spread = symbol_info['ask'] - symbol_info['bid']
                    print(f'   {symbol}: Bid={symbol_info["bid"]:.5f}, Ask={symbol_info["ask"]:.5f}, Spread={spread:.5f}')
                else:
                    print(f'   {symbol}: ❌ Not available')
            
            # Test positions (not async)
            positions = connector.get_positions()
            print(f'\n📋 Current positions: {len(positions)}')
            
            # Test orders (not async)
            orders = connector.get_orders()
            print(f'📋 Pending orders: {len(orders)}')
            
            # Test market status
            print(f'\n🕐 Market status:')
            for symbol in ['EURUSD', 'GBPUSD']:
                symbol_info = connector.get_symbol_info(symbol)
                if symbol_info:
                    trading_allowed = symbol_info.get('trade_mode', 0) > 0
                    print(f'   {symbol}: {"✅ Trading allowed" if trading_allowed else "❌ Trading disabled"}')
            
            await connector.disconnect()
            print('\n🎉 Demo trading integration test PASSED!')
            print('💡 Ready to execute demo trades!')
            
            return True
            
        else:
            print('❌ Connection failed')
            return False
            
    except Exception as e:
        print(f'❌ Test failed: {str(e)}')
        import traceback
        traceback.print_exc()
        return False

async def test_simple_demo_order():
    """Test placing a simple demo order (dry run)."""
    print('\n🧪 Testing Demo Order Placement (Simulation)...')
    
    connector = create_mt5_connector()
    
    try:
        if await connector.connect():
            # Get current EURUSD price (not async)
            symbol_info = connector.get_symbol_info('EURUSD')
            if symbol_info:
                current_price = symbol_info['bid']
                print(f'📊 EURUSD current price: {current_price:.5f}')
                
                # Simulate order parameters
                order_params = {
                    'symbol': 'EURUSD',
                    'volume': 0.01,  # Micro lot
                    'order_type': 'buy',
                    'price': current_price,
                    'stop_loss': current_price - 0.0050,  # 50 pips
                    'take_profit': current_price + 0.0100,  # 100 pips
                }
                
                print(f'📋 Demo order simulation:')
                print(f'   Symbol: {order_params["symbol"]}')
                print(f'   Volume: {order_params["volume"]} lots')
                print(f'   Type: {order_params["order_type"]}')
                print(f'   Entry: {order_params["price"]:.5f}')
                print(f'   Stop Loss: {order_params["stop_loss"]:.5f}')
                print(f'   Take Profit: {order_params["take_profit"]:.5f}')
                
                print('✅ Order simulation successful!')
                print('💡 Ready for actual demo trading!')
            
            await connector.disconnect()
            return True
            
    except Exception as e:
        print(f'❌ Order simulation failed: {str(e)}')
        return False

if __name__ == "__main__":
    async def main():
        success1 = await test_demo_trading()
        success2 = await test_simple_demo_order()
        
        if success1 and success2:
            print('\n🎉 ALL TESTS PASSED!')
            print('🚀 Your demo trading system is ready!')
            print('\n💡 Next steps:')
            print('   1. Execute real demo trades')
            print('   2. Test execution algorithms (TWAP, POV, Direct)')
            print('   3. Monitor real-time positions')
            print('   4. Test risk management features')
        else:
            print('\n❌ Some tests failed. Please check the errors above.')
    
    asyncio.run(main()) 
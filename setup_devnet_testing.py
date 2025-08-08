#!/usr/bin/env python3
"""
Devnet Testing Setup for TrenchBotAi
Creates test wallet and requests faucet SOL
"""
import os
import json
import asyncio
import aiohttp
from solders.keypair import Keypair
from solders.pubkey import Pubkey
from dotenv import load_dotenv

load_dotenv()

class DevnetSetup:
    def __init__(self):
        self.devnet_rpc = "https://api.devnet.solana.com"
        self.helius_devnet = os.getenv('HELIUS_RPC_HTTP')
        self.wallet_path = "./test_wallet.json"
        
    def create_test_wallet(self):
        """Create a new devnet test wallet"""
        print("🔑 Creating devnet test wallet...")
        
        # Generate new keypair
        keypair = Keypair()
        pubkey = keypair.pubkey()
        
        # Save keypair to file
        keypair_bytes = list(keypair.secret())
        wallet_data = {
            "keypair": keypair_bytes,
            "pubkey": str(pubkey),
            "network": "devnet"
        }
        
        with open(self.wallet_path, 'w') as f:
            json.dump(wallet_data, f, indent=2)
        
        print(f"✅ Wallet created!")
        print(f"   Address: {pubkey}")
        print(f"   Saved to: {self.wallet_path}")
        
        return str(pubkey)
    
    async def request_faucet_sol(self, pubkey_str):
        """Request devnet SOL from faucet"""
        print("💰 Requesting devnet SOL from faucet...")
        
        try:
            # Request 2 SOL from devnet faucet
            async with aiohttp.ClientSession() as session:
                payload = {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "requestAirdrop",
                    "params": [
                        pubkey_str,
                        2000000000  # 2 SOL in lamports
                    ]
                }
                
                async with session.post(self.devnet_rpc, json=payload) as response:
                    if response.status == 200:
                        data = await response.json()
                        if 'result' in data:
                            tx_signature = data['result']
                            print(f"✅ Airdrop requested!")
                            print(f"   Transaction: {tx_signature}")
                            print("   Waiting for confirmation...")
                            
                            # Wait for confirmation
                            await self.wait_for_confirmation(tx_signature)
                            return True
                        else:
                            print(f"❌ Faucet error: {data.get('error', 'Unknown error')}")
                            return False
                    else:
                        print(f"❌ Faucet request failed: HTTP {response.status}")
                        return False
        except Exception as e:
            print(f"❌ Faucet request failed: {str(e)}")
            return False
    
    async def wait_for_confirmation(self, signature, max_attempts=30):
        """Wait for transaction confirmation"""
        for attempt in range(max_attempts):
            try:
                async with aiohttp.ClientSession() as session:
                    payload = {
                        "jsonrpc": "2.0",
                        "id": 1,
                        "method": "getSignatureStatuses",
                        "params": [[signature]]
                    }
                    
                    async with session.post(self.devnet_rpc, json=payload) as response:
                        if response.status == 200:
                            data = await response.json()
                            statuses = data.get('result', {}).get('value', [])
                            
                            if statuses and statuses[0] and statuses[0].get('confirmationStatus'):
                                status = statuses[0]['confirmationStatus']
                                if status in ['confirmed', 'finalized']:
                                    print(f"✅ Transaction confirmed ({status})")
                                    return True
                                else:
                                    print(f"⏳ Status: {status} (attempt {attempt+1}/{max_attempts})")
                            else:
                                print(f"⏳ Waiting for confirmation (attempt {attempt+1}/{max_attempts})")
            except:
                pass
            
            await asyncio.sleep(2)  # Wait 2 seconds between attempts
        
        print("⚠️ Transaction confirmation timeout")
        return False
    
    async def check_balance(self, pubkey_str):
        """Check wallet balance"""
        print("💰 Checking wallet balance...")
        
        try:
            async with aiohttp.ClientSession() as session:
                payload = {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "getBalance",
                    "params": [pubkey_str]
                }
                
                async with session.post(self.devnet_rpc, json=payload) as response:
                    if response.status == 200:
                        data = await response.json()
                        balance_lamports = data.get('result', {}).get('value', 0)
                        balance_sol = balance_lamports / 1_000_000_000  # Convert to SOL
                        
                        print(f"✅ Balance: {balance_sol} SOL ({balance_lamports:,} lamports)")
                        return balance_sol
                    else:
                        print(f"❌ Balance check failed: HTTP {response.status}")
                        return 0
        except Exception as e:
            print(f"❌ Balance check failed: {str(e)}")
            return 0
    
    async def test_devnet_apis(self):
        """Test devnet API connectivity"""
        print("🔗 Testing devnet API connectivity...")
        
        # Test standard devnet RPC
        try:
            async with aiohttp.ClientSession() as session:
                payload = {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "getHealth"
                }
                async with session.post(self.devnet_rpc, json=payload) as response:
                    if response.status == 200:
                        print("✅ Devnet RPC: Connected")
                    else:
                        print(f"❌ Devnet RPC: HTTP {response.status}")
        except Exception as e:
            print(f"❌ Devnet RPC: {str(e)}")
        
        # Test Helius devnet
        if self.helius_devnet:
            try:
                async with aiohttp.ClientSession() as session:
                    payload = {
                        "jsonrpc": "2.0",
                        "id": 1,
                        "method": "getHealth"
                    }
                    async with session.post(self.helius_devnet, json=payload) as response:
                        if response.status == 200:
                            print("✅ Helius Devnet: Connected")
                        else:
                            print(f"❌ Helius Devnet: HTTP {response.status}")
            except Exception as e:
                print(f"❌ Helius Devnet: {str(e)}")
    
    async def setup_complete_devnet_environment(self):
        """Complete devnet setup"""
        print("🚀 TrenchBotAi Devnet Setup")
        print("=" * 50)
        
        # Test APIs first
        await self.test_devnet_apis()
        print()
        
        # Create or load wallet
        if os.path.exists(self.wallet_path):
            print("📁 Loading existing test wallet...")
            with open(self.wallet_path, 'r') as f:
                wallet_data = json.load(f)
                pubkey_str = wallet_data['pubkey']
                print(f"✅ Loaded wallet: {pubkey_str}")
        else:
            pubkey_str = self.create_test_wallet()
        
        print()
        
        # Check current balance
        current_balance = await self.check_balance(pubkey_str)
        print()
        
        # Request SOL if balance is low
        if current_balance < 1.0:
            print("💳 Balance low, requesting faucet SOL...")
            success = await self.request_faucet_sol(pubkey_str)
            if success:
                print()
                final_balance = await self.check_balance(pubkey_str)
            else:
                print("❌ Could not get faucet SOL")
                final_balance = current_balance
        else:
            print("✅ Sufficient balance for testing")
            final_balance = current_balance
        
        print()
        print("=" * 50)
        print("📋 DEVNET SETUP SUMMARY")
        print("=" * 50)
        print(f"🔑 Wallet: {pubkey_str}")
        print(f"💰 Balance: {final_balance} SOL")
        print(f"🌐 Network: Devnet")
        print(f"📁 Wallet file: {self.wallet_path}")
        print()
        
        if final_balance >= 1.0:
            print("🎉 DEVNET SETUP COMPLETE!")
            print("✅ Ready for safe testing with devnet SOL")
            print("🎯 Next: Run your bot tests on devnet")
            print()
            print("Commands to test:")
            print(f"  export SOLANA_NETWORK=devnet")
            print(f"  export TEST_WALLET={pubkey_str}")
            print(f"  python3 test_bot_locally.py")
        else:
            print("⚠️ Setup incomplete - insufficient SOL")
            print("Try requesting from web faucet:")
            print(f"  https://faucet.solana.com")
            print(f"  Address: {pubkey_str}")
        
        return final_balance >= 1.0

async def main():
    setup = DevnetSetup()
    await setup.setup_complete_devnet_environment()

if __name__ == "__main__":
    asyncio.run(main())
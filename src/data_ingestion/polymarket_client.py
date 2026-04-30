"""
Author: AI Assistant
Date: 2026-03-17
Description: Polymarket API Client with retry and rate limiting logic.
"""

import time
import requests
import random
from loguru import logger
from typing import Dict, Any, Optional, List
from src.config.settings import POLYMARKET_API_KEY, POLYMARKET_PRIVATE_KEYS, POLYMARKET_CLOB_API_KEYS, POLYMARKET_CLOB_SECRETS, POLYMARKET_CLOB_PASSPHRASES, ALLOW_MOCK_DATA
from py_clob_client.client import ClobClient
from py_clob_client.clob_types import ApiCreds, OpenOrderParams
from py_clob_client.signing.model import ClobAuth

class PolymarketClient:
    """Client for interacting with Polymarket endpoints using multi-key rotation."""
    
    GAMMA_API_URL = "https://gamma-api.polymarket.com"
    CLOB_API_URL = "https://clob.polymarket.com"
    CHAIN_ID = 137 # Polygon Mainnet

    def __init__(self, api_key: str = POLYMARKET_API_KEY, max_retries: int = 3, backoff_factor: float = 1.0):
        """
        Initialize the client.
        """
        self.api_key = api_key
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.session = requests.Session()
        
        self.clob_clients = []
        
        # 1. Try initializing with direct L2 credentials first
        if POLYMARKET_CLOB_API_KEYS and POLYMARKET_CLOB_SECRETS and POLYMARKET_CLOB_PASSPHRASES:
            for key, secret, passphrase in zip(POLYMARKET_CLOB_API_KEYS, POLYMARKET_CLOB_SECRETS, POLYMARKET_CLOB_PASSPHRASES):
                try:
                    creds = ClobAuth(api_key=key, api_secret=secret, api_passphrase=passphrase)
                    client = ClobClient(self.CLOB_API_URL, chain_id=self.CHAIN_ID, creds=creds)
                    self.clob_clients.append(client)
                except Exception as e:
                    logger.error(f"Failed to initialize CLOB client with L2 creds for key {key[:5]}... : {e}")
                    
        # 2. Try initializing with L1 private keys if L2 isn't available
        elif POLYMARKET_PRIVATE_KEYS:
            for pk in POLYMARKET_PRIVATE_KEYS:
                try:
                    # Initialize client with raw private key
                    client = ClobClient(
                        self.CLOB_API_URL, 
                        chain_id=self.CHAIN_ID, 
                        key=pk
                    )
                    
                    # Auto derive or create credentials
                    creds_obj = client.create_or_derive_api_creds()
                    client.set_api_creds(creds_obj)
                    
                    self.clob_clients.append(client)
                except Exception as e:
                    logger.error(f"Failed to initialize CLOB client for private key ending in ...{pk[-4:] if len(pk)>4 else ''}: {e}")
        
        if self.clob_clients:
            logger.info(f"Initialized {len(self.clob_clients)} CLOB API clients for rotation.")
        else:
            logger.warning("No valid CLOB API clients initialized. CLOB endpoints will use Mock Data or fail.")
            
        self._current_client_idx = 0

    def _get_active_clob_client(self) -> Optional[ClobClient]:
        """Returns the currently active CLOB client."""
        if not self.clob_clients:
            return None
        return self.clob_clients[self._current_client_idx]
        
    def _rotate_clob_client(self):
        """Rotates to the next available CLOB client."""
        if not self.clob_clients:
            return
        self._current_client_idx = (self._current_client_idx + 1) % len(self.clob_clients)
        logger.info(f"Rotated CLOB API client to index {self._current_client_idx}")

    def _request(self, endpoint: str, params: Optional[Dict] = None, base_url: str = GAMMA_API_URL) -> Optional[Dict|List]:
        """
        Internal method to make requests with retry and rate limit handling.
        """
        url = f"{base_url}{endpoint}"
        retries = 0
        
        while retries <= self.max_retries:
            try:
                response = self.session.get(url, params=params, timeout=10)
                
                # Check for rate limiting
                if response.status_code == 429:
                    retry_after = int(response.headers.get("Retry-After", self.backoff_factor * (2 ** retries)))
                    logger.warning(f"Rate limited (429) on {url}. Retrying in {retry_after} seconds...")
                    time.sleep(retry_after)
                    retries += 1
                    continue
                
                response.raise_for_status()
                return response.json()
            except requests.exceptions.RequestException as e:
                if hasattr(e, 'response') and e.response is not None and e.response.status_code == 405:
                    logger.warning(f"Endpoint not allowed (405): {url}. Missing Auth or Invalid Method.")
                    return None
                logger.error(f"Request failed: {url} | Error: {e}")
                retries += 1
                if retries <= self.max_retries:
                    sleep_time = self.backoff_factor * (2 ** retries)
                    logger.info(f"Retrying in {sleep_time} seconds (Attempt {retries}/{self.max_retries})...")
                    time.sleep(sleep_time)
                else:
                    logger.error(f"Max retries reached for {url}")
                    return None
        return None

    def get_transaction_records(self, address: str, start_time: Optional[str] = None, end_time: Optional[str] = None) -> List[Dict]:
        """
        Fetch transaction records for a specific address.
        """
        logger.info(f"Fetching transaction records for {address}")
        
        clob = self._get_active_clob_client()
        if clob:
            retries = 0
            while retries <= self.max_retries:
                try:
                    # using get_orders instead of raw _request for proper authentication payload creation
                    orders = clob.get_orders(OpenOrderParams())
                    return orders
                except Exception as e:
                    # Identify rate limit or auth error
                    err_str = str(e).lower()
                    if "429" in err_str or "rate limit" in err_str:
                        logger.warning(f"Rate limited on CLOB Client API. Rotating key...")
                        self._rotate_clob_client()
                        clob = self._get_active_clob_client() # Next client
                        time.sleep(1.0)
                        retries += 1
                    else:
                        logger.error(f"CLOB Request failed for address {address}: {e}")
                        break
        
        # Mock strategy fallback - ONLY if ALLOW_MOCK_DATA is enabled
        if not ALLOW_MOCK_DATA:
            logger.error(f"Failed to fetch real data for {address} and ALLOW_MOCK_DATA=false. Raising error.")
            raise DataFetchError(f"No CLOB credentials available and mock data is disabled for address {address}")

        logger.warning(f"[MOCK MODE] Failed to fetch real data for {address}. Using mock transaction data for testing ONLY.")
        return [
            {"transaction_id": f"tx001_{address}", "address": address, "market_id": "mkt1", "side": "BUY", "amount": 500, "price": 0.45, "timestamp": "2024-01-15T12:00:00Z"},
            {"transaction_id": f"tx002_{address}", "address": address, "market_id": "mkt1", "side": "SELL", "amount": 500, "price": 0.60, "timestamp": "2024-01-20T12:00:00Z"}
        ]

    def get_market_snapshot(self, market_id: str) -> Optional[Dict]:
        """
        Get snapshot data for a specific market.
        
        Args:
            market_id: The Polymarket condition_id.
            
        Returns:
            Dictionary containing market snapshot info.
        """
        logger.info(f"Fetching snapshot for market {market_id}")
        endpoint = f"/markets/{market_id}"
        return self._request(endpoint)

    def get_market_resolution(self, market_id: str) -> Optional[Dict]:
        """
        Get final resolution data for a market.
        
        Args:
            market_id: The market identifier.
            
        Returns:
            Market resolution details.
        """
        logger.info(f"Fetching resolution for market {market_id}")
        endpoint = f"/markets/{market_id}"
        res = self._request(endpoint)
        if res:
             return {"market_id": market_id, "resolved": res.get("closed"), "resolution": res.get("tokens_resolved")}
        return None

    def get_top_markets(self, limit: int = 50, closed: bool = True) -> List[str]:
        """
        Fetch the top markets by volume.
        
        Args:
            limit: Maximum number of markets to return.
            closed: If true, fetch resolved markets. If false, fetch active open markets.
            
        Returns:
            List of market condition_ids.
        """
        logger.info(f"Fetching top {limit} markets (closed={closed}) by volume")
        endpoint = "/markets"
        params = {
            "active": "false" if closed else "true",
            "closed": "true" if closed else "false",
            "limit": limit,
            "order": "volume",
            "ascending": "false"
        }
        res = self._request(endpoint, params=params)
        import json
        markets = []
        if res:
             for m in res:
                  clob_tokens = m.get('clobTokenIds')
                  if clob_tokens:
                       try:
                           tokens = json.loads(clob_tokens)
                           markets.extend(tokens)
                       except:
                           pass
        return markets

# data_fetch.py
import requests
import logging
import threading
import time
from datetime import datetime
from typing import Optional, Dict, Any
from requests.adapters import HTTPAdapter
from urllib3.util import Retry
import pandas as pd
from backend.appfiles.config import Config
from utils.logger import setup_logger

logger = setup_logger(__name__)

class APIError(Exception):
    """Custom exception for API-related errors."""
    pass

class RateLimiter:
    def __init__(self, rate: int, interval: int):
        self.rate = rate
        self.interval = interval
        self.tokens = rate
        self.last_update = time.time()
        self.lock = threading.Lock()

    def acquire(self):
        with self.lock:
            now = time.time()
            time_passed = now - self.last_update
            self.tokens = min(self.rate, self.tokens + time_passed * (self.rate / self.interval))
            self.last_update = now

            if self.tokens < 1:
                wait_time = (1 - self.tokens) * (self.interval / self.rate)
                time.sleep(wait_time)
                self.tokens = 0
            else:
                self.tokens -= 1

class DataFetcher:
    def __init__(self):
        self.rate_limiters = {
            'alpha_vantage': RateLimiter(Config.RATE_LIMIT['alpha_vantage']['requests_per_minute'], 60),
            'finnhub': RateLimiter(Config.RATE_LIMIT['finnhub']['requests_per_minute'], 60)
        }
        
        self.session = self._create_robust_session()
        self._cache = {}
        self._cache_lock = threading.Lock()
        self._data_lock = threading.Lock()
        logger.info("DataFetcher initialized successfully")

    def _create_robust_session(self) -> requests.Session:
        session = requests.Session()
        retries = Retry(
            total=3,
            backoff_factor=0.5,
            status_forcelist=[500, 502, 503, 504],
            allowed_methods=frozenset(['GET', 'POST'])
        )
        
        adapter = HTTPAdapter(
            max_retries=retries,
            pool_connections=10,
            pool_maxsize=10
        )
        
        session.mount('http://', adapter)
        session.mount('https://', adapter)
        session.headers.update({
            'User-Agent': 'TradingBot/1.0',
            'Accept': 'application/json',
            'Content-Type': 'application/json'
        })
        return session

    def _get_from_cache(self, key: str) -> Optional[pd.DataFrame]:
        with self._cache_lock:
            if key in self._cache:
                data, timestamp = self._cache[key]
                if datetime.now().timestamp() - timestamp < 300:
                    return data
                del self._cache[key]
        return None

    def _add_to_cache(self, key: str, data: Any, expire: int = 300) -> None:
        with self._cache_lock:
            self._cache[key] = (data, datetime.now().timestamp())

    def cleanup(self) -> None:
        try:
            if self.session:
                self.session.close()
            with self._cache_lock:
                self._cache.clear()
            logger.info("DataFetcher cleanup completed")
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")

    def __enter__(self) -> 'DataFetcher':
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.cleanup()
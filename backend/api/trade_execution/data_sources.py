# data_sources.py

import pandas as pd
import requests
from typing import Dict, Any, Optional
from datetime import datetime
import threading
from decimal import Decimal

from backend.appfiles.config import Config
from utils.logger import setup_logger
from exceptions import APIError

logger = setup_logger(__name__)

class DataSourceHandler:
    def __init__(self):
        self.session = self._create_robust_session()
        self.rate_limiters = {
            'alpha_vantage': RateLimiter(
                Config.RATE_LIMIT['alpha_vantage']['requests_per_minute'], 
                60
            ),
            'finnhub': RateLimiter(
                Config.RATE_LIMIT['finnhub']['requests_per_minute'], 
                60
            )
        }
        self._cache = {}
        self._cache_lock = threading.Lock()

    def _create_robust_session(self) -> requests.Session:
        """Create a session with retry logic."""
        session = requests.Session()
        retries = requests.adapters.Retry(
            total=3,
            backoff_factor=0.5,
            status_forcelist=[500, 502, 503, 504]
        )
        adapter = requests.adapters.HTTPAdapter(max_retries=retries)
        session.mount('http://', adapter)
        session.mount('https://', adapter)
        return session

    def _get_from_cache(self, key: str) -> Optional[Any]:
        """Get data from cache if not expired."""
        with self._cache_lock:
            if key in self._cache:
                data, timestamp = self._cache[key]
                if datetime.now().timestamp() - timestamp < 300:  # 5 min expiry
                    return data
                del self._cache[key]
        return None

    def _add_to_cache(self, key: str, data: Any) -> None:
        """Add data to cache with timestamp."""
        with self._cache_lock:
            self._cache[key] = (data, datetime.now().timestamp())

    def _fetch_with_fallback(self, symbol: str, timeframe: str, limit: int) -> Dict[str, Any]:
        """Fetch data with fallback between multiple sources."""
        exceptions = []

        try:
            interval = self._timeframe_to_alpha_vantage(timeframe)
            data = self._fetch_alpha_vantage_data(symbol, interval)
            if data:
                return data
        except Exception as e:
            exceptions.append(('Alpha Vantage', str(e)))

        try:
            resolution = self._timeframe_to_finnhub(timeframe)
            data = self._fetch_finnhub_data(symbol, resolution)
            if data:
                return data
        except Exception as e:
            exceptions.append(('Finnhub', str(e)))

        error_msg = '; '.join([f"{src}: {err}" for src, err in exceptions])
        raise APIError(f"All data sources failed. {error_msg}")

    def _fetch_alpha_vantage_data(self, symbol: str, interval: str) -> Dict:
        """Fetch data from Alpha Vantage API."""
        cache_key = f"av_{symbol}_{interval}"
        cached_data = self._get_from_cache(cache_key)
        if cached_data:
            return cached_data

        try:
            self.rate_limiters['alpha_vantage'].acquire()
            url = (
                f"https://www.alphavantage.co/query"
                f"?function=TIME_SERIES_INTRADAY"
                f"&symbol={symbol}"
                f"&interval={interval}"
                f"&apikey={Config.API.ALPHA_VANTAGE_API_KEY}"
                f"&outputsize=full"
            )

            response = self.session.get(url)
            response.raise_for_status()
            data = response.json()

            if 'Error Message' in data:
                raise APIError(data['Error Message'])
            if 'Note' in data:
                raise APIError("Alpha Vantage API limit reached")

            self._add_to_cache(cache_key, data)
            return data

        except requests.exceptions.RequestException as e:
            logger.error(f"Alpha Vantage request error for {symbol}: {str(e)}")
            raise APIError(f"Alpha Vantage request failed: {str(e)}")

    def _fetch_finnhub_data(self, symbol: str, resolution: str) -> Dict:
        """Fetch data from Finnhub API."""
        cache_key = f"fh_{symbol}_{resolution}"
        cached_data = self._get_from_cache(cache_key)
        if cached_data:
            return cached_data

        try:
            self.rate_limiters['finnhub'].acquire()
            end_time = int(datetime.now().timestamp())
            start_time = end_time - (86400 * 30)  # Last 30 days

            url = (
                f"https://finnhub.io/api/v1/stock/candle"
                f"?symbol={symbol}"
                f"&resolution={resolution}"
                f"&from={start_time}"
                f"&to={end_time}"
                f"&token={Config.API.FINNHUB_API_KEY}"
            )

            response = self.session.get(url)
            response.raise_for_status()
            data = response.json()

            if data.get('s') == 'no_data':
                raise APIError(f"No data available for {symbol}")

            self._add_to_cache(cache_key, data)
            return data

        except Exception as e:
            logger.error(f"Finnhub API error for {symbol}: {str(e)}")
            raise

    def _process_raw_data(self, data: Dict[str, Any], timeframe: str) -> pd.DataFrame:
        """Process raw API data into standardized DataFrame."""
        try:
            if 'Time Series' in data:
                time_series_key = f"Time Series ({timeframe})"
                if time_series_key not in data:
                    time_series_key = list(filter(
                        lambda x: 'Time Series' in x, 
                        data.keys()
                    ))[0]
                    
                time_series = data[time_series_key]
                df = pd.DataFrame.from_dict(time_series, orient='index')
                
                column_map = {
                    '1. open': 'open',
                    '2. high': 'high',
                    '3. low': 'low',
                    '4. close': 'close',
                    '5. volume': 'volume'
                }
                df.rename(columns=column_map, inplace=True)
                
            else:  # Finnhub format
                df = pd.DataFrame({
                    'timestamp': pd.to_datetime(data['t'], unit='s'),
                    'open': data['o'],
                    'high': data['h'],
                    'low': data['l'],
                    'close': data['c'],
                    'volume': data['v']
                })
                df.set_index('timestamp', inplace=True)

            # Convert to numeric and calculate indicators
            for col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            df.dropna(inplace=True)
            df.sort_index(inplace=True)
            
            df['returns'] = df['close'].pct_change()
            df['volatility'] = df['returns'].rolling(window=20).std()
            df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
            
            df['sma_20'] = df['close'].rolling(window=20).mean()
            df['sma_50'] = df['close'].rolling(window=50).mean()
            df['rsi'] = self._calculate_rsi(df['close'])
            
            df['volume_sma'] = df['volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma']

            return df

        except Exception as e:
            logger.error(f"Error processing raw data: {str(e)}")
            raise

    @staticmethod
    def _calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index."""
        try:
            delta = prices.astype(float).diff()
            gain = (delta.where(delta > 0, 0.0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0.0)).rolling(window=period).mean()
            
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi
        except Exception as e:
            logger.error(f"Error calculating RSI: {str(e)}")
            return pd.Series(index=prices.index)

    @staticmethod
    def _timeframe_to_alpha_vantage(timeframe: str) -> str:
        """Convert timeframe to Alpha Vantage format."""
        mapping = {
            '1min': '1min',
            '5min': '5min',
            '15min': '15min',
            '30min': '30min',
            '60min': '60min',
            'daily': 'daily',
            'weekly': 'weekly',
            'monthly': 'monthly'
        }
        return mapping.get(timeframe, 'daily')

    @staticmethod
    def _timeframe_to_finnhub(timeframe: str) -> str:
        """Convert timeframe to Finnhub format."""
        mapping = {
            '1min': '1',
            '5min': '5',
            '15min': '15',
            '30min': '30',
            '60min': '60',
            'daily': 'D',
            'weekly': 'W',
            'monthly': 'M'
        }
        return mapping.get(timeframe, 'D')
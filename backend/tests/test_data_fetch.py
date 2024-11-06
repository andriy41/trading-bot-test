# tests/test_data_fetch.py

import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
from backend.api.trade_execution.data_fetch import DataFetcher, APIError

class TestDataFetcher(unittest.TestCase):
    def setUp(self):
        """Setup test environment."""
        self.fetcher = DataFetcher()
        self.test_symbol = 'AAPL'
        self.test_timeframe = 'daily'
        self.test_data_av = {
            'Time Series (Daily)': {
                '2023-01-01': {
                    '1. open': '100.0',
                    '2. high': '105.0',
                    '3. low': '95.0',
                    '4. close': '102.0',
                    '5. volume': '1000000'
                },
                '2023-01-02': {
                    '1. open': '102.0',
                    '2. high': '107.0',
                    '3. low': '97.0',
                    '4. close': '104.0',
                    '5. volume': '1100000'
                }
            }
        }
        self.test_data_fh = {
            'c': [102.0, 104.0],
            'h': [105.0, 107.0],
            'l': [95.0, 97.0],
            'o': [100.0, 102.0],
            'v': [1000000, 1100000],
            't': [1672531200, 1672617600]  # 2023-01-01, 2023-01-02
        }

    def tearDown(self):
        """Cleanup after tests."""
        self.fetcher.cleanup()

    @patch('requests.Session.get')
    def test_alpha_vantage_fetch(self, mock_get):
        """Test Alpha Vantage data fetching."""
        # Setup mock response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = self.test_data_av
        mock_get.return_value = mock_response

        # Test fetch
        data = self.fetcher._fetch_alpha_vantage_data(self.test_symbol, 'daily')
        
        # Verify response
        self.assertIsInstance(data, dict)
        self.assertIn('Time Series (Daily)', data)
        self.assertEqual(len(data['Time Series (Daily)']), 2)
        
        # Test error handling
        mock_response.json.return_value = {'Error Message': 'Invalid API call'}
        with self.assertRaises(APIError):
            self.fetcher._fetch_alpha_vantage_data(self.test_symbol, 'daily')

    @patch('requests.Session.get')
    def test_finnhub_fetch(self, mock_get):
        """Test Finnhub data fetching."""
        # Setup mock response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = self.test_data_fh
        mock_get.return_value = mock_response

        # Test fetch
        data = self.fetcher._fetch_finnhub_data(self.test_symbol, 'D')
        
        # Verify response
        self.assertIsInstance(data, dict)
        self.assertTrue(all(key in data for key in ['c', 'h', 'l', 'o', 'v', 't']))
        self.assertEqual(len(data['c']), 2)
        
        # Test error handling
        mock_response.json.return_value = {'s': 'no_data'}
        with self.assertRaises(APIError):
            self.fetcher._fetch_finnhub_data(self.test_symbol, 'D')

    def test_process_raw_data(self):
        """Test data processing for both data sources."""
        # Test Alpha Vantage format
        df_av = self.fetcher._process_raw_data(self.test_data_av, 'daily')
        self.assertIsInstance(df_av, pd.DataFrame)
        self.assertTrue(all(col in df_av.columns for col in [
            'open', 'high', 'low', 'close', 'volume',
            'returns', 'volatility', 'typical_price',
            'sma_20', 'sma_50', 'rsi', 'volume_sma', 'volume_ratio'
        ]))
        
        # Test Finnhub format
        df_fh = self.fetcher._process_raw_data(self.test_data_fh, 'daily')
        self.assertIsInstance(df_fh, pd.DataFrame)
        self.assertTrue(all(col in df_fh.columns for col in [
            'open', 'high', 'low', 'close', 'volume',
            'returns', 'volatility', 'typical_price',
            'sma_20', 'sma_50', 'rsi', 'volume_sma', 'volume_ratio'
        ]))

    def test_cache_operations(self):
        """Test cache operations."""
        test_data = pd.DataFrame({'test': [1, 2, 3]})
        test_key = 'test_key'

        # Test adding to cache
        self.fetcher._add_to_cache(test_key, test_data)
        cached_data = self.fetcher._get_from_cache(test_key)
        self.assertIsNotNone(cached_data)
        pd.testing.assert_frame_equal(cached_data, test_data)

        # Test cache expiration
        with patch('datetime.datetime') as mock_datetime:
            mock_datetime.now.return_value = datetime.now() + timedelta(minutes=6)
            cached_data = self.fetcher._get_from_cache(test_key)
            self.assertIsNone(cached_data)

        # Test cache clear
        self.fetcher._add_to_cache('key1', test_data)
        self.fetcher._add_to_cache('key2', test_data)
        self.fetcher.clear_cache(pattern='key1')
        self.assertIsNone(self.fetcher._get_from_cache('key1'))
        self.assertIsNotNone(self.fetcher._get_from_cache('key2'))

    def test_batch_operations(self):
        """Test batch fetch operations."""
        test_symbols = ['AAPL', 'MSFT', 'GOOGL']
        mock_df = pd.DataFrame({'test': [1, 2, 3]})
        
        # Test batch fetch
        with patch.object(self.fetcher, 'fetch_data', return_value=mock_df):
            results = self.fetcher.fetch_batch(test_symbols, self.test_timeframe)
            
            self.assertIsInstance(results, dict)
            self.assertEqual(len(results), len(test_symbols))
            for symbol in test_symbols:
                self.assertIn(symbol, results)
                pd.testing.assert_frame_equal(results[symbol], mock_df)

        # Test batch fetch with errors
        def mock_fetch(symbol, *args, **kwargs):
            if symbol == 'MSFT':
                raise APIError("Test error")
            return mock_df

        with patch.object(self.fetcher, 'fetch_data', side_effect=mock_fetch):
            results = self.fetcher.fetch_batch(test_symbols, self.test_timeframe)
            self.assertIn('AAPL', results)
            self.assertIn('GOOGL', results)
            self.assertNotIn('MSFT', results)

    def test_rate_limiting(self):
        """Test rate limiting functionality."""
        # Test rate limiter initialization
        limiter = self.fetcher.rate_limiters['alpha_vantage']
        self.assertEqual(limiter.tokens, limiter.rate)

        # Test rate limiting timing
        start_time = time.time()
        for _ in range(3):
            limiter.acquire()
        elapsed_time = time.time() - start_time
        
        # Verify rate limiting is working (should take at least some time)
        self.assertGreater(elapsed_time, 0)

    def test_error_handling(self):
        """Test error handling scenarios."""
        # Test invalid symbol
        with self.assertRaises(ValueError):
            self.fetcher.fetch_data('')

        # Test API error handling
        with patch.object(self.fetcher, '_fetch_with_fallback', side_effect=APIError("Test error")):
            with self.assertRaises(APIError):
                self.fetcher.fetch_data(self.test_symbol)

        # Test network error handling
        with patch('requests.Session.get', side_effect=Exception("Network error")):
            with self.assertRaises(APIError):
                self.fetcher._fetch_alpha_vantage_data(self.test_symbol, 'daily')

    def test_data_validation(self):
        """Test data validation methods."""
        # Test valid Alpha Vantage data
        self.assertTrue(self.fetcher._validate_data(self.test_data_av))
        
        # Test valid Finnhub data
        self.assertTrue(self.fetcher._validate_data(self.test_data_fh))
        
        # Test invalid data
        invalid_data = {'invalid': 'data'}
        self.assertFalse(self.fetcher._validate_data(invalid_data))

    def test_timeframe_conversion(self):
        """Test timeframe conversion methods."""
        # Test Alpha Vantage conversion
        self.assertEqual(self.fetcher._timeframe_to_alpha_vantage('daily'), 'daily')
        self.assertEqual(self.fetcher._timeframe_to_alpha_vantage('1min'), '1min')
        self.assertEqual(self.fetcher._timeframe_to_alpha_vantage('invalid'), 'daily')

        # Test Finnhub conversion
        self.assertEqual(self.fetcher._timeframe_to_finnhub('daily'), 'D')
        self.assertEqual(self.fetcher._timeframe_to_finnhub('1min'), '1')
        self.assertEqual(self.fetcher._timeframe_to_finnhub('invalid'), 'D')

    def test_cleanup(self):
        """Test cleanup functionality."""
        # Add some test data to cache
        self.fetcher._add_to_cache('test_key', pd.DataFrame())
        
        # Run cleanup
        self.fetcher.cleanup()
        
        # Verify cache is cleared
        self.assertEqual(len(self.fetcher._cache), 0)
        
        # Verify session is closed
        self.assertIsNone(self.fetcher.session._transport)

if __name__ == '__main__':
    unittest.main()

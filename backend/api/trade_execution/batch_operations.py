# batch_operations.py
import threading
import queue
from typing import Dict, List, Optional, Callable
import pandas as pd
from datetime import datetime
from utils.logger import setup_logger

logger = setup_logger(__name__)

class BatchOperationsHandler:
    def __init__(self):
        self._batch_queue = queue.Queue()
        self._cache = {}
        self._cache_lock = threading.Lock()
        self._data_lock = threading.Lock()

    def fetch_batch(
        self,
        symbols: List[str],
        timeframe: str = 'daily',
        limit: int = 100
    ) -> Dict[str, pd.DataFrame]:
        results = {}
        errors = {}
        threads = []

        def fetch_symbol(symbol):
            try:
                data = self.fetch_data(symbol, timeframe, limit)
                with self._data_lock:
                    results[symbol] = data
            except Exception as e:
                with self._data_lock:
                    errors[symbol] = str(e)

        for symbol in symbols:
            thread = threading.Thread(target=fetch_symbol, args=(symbol,))
            thread.start()
            threads.append(thread)

        for thread in threads:
            thread.join()

        if errors:
            logger.error(f"Batch fetch errors: {errors}")

        return results

    def schedule_batch(
        self,
        symbols: List[str],
        timeframe: str = 'daily',
        limit: int = 100,
        callback: Optional[Callable] = None
    ) -> None:
        self._batch_queue.put((symbols, timeframe, limit, callback))

    def _process_batch_queue(self) -> None:
        while True:
            try:
                batch = self._batch_queue.get()
                if batch is None:
                    break
                
                symbols, timeframe, limit, callback = batch
                results = self.fetch_batch(symbols, timeframe, limit)
                
                if callback:
                    try:
                        callback(results)
                    except Exception as e:
                        logger.error(f"Batch callback error: {str(e)}")
                        
            except Exception as e:
                logger.error(f"Batch processing error: {str(e)}")
            finally:
                self._batch_queue.task_done()

    def _fetch_batch(
        self,
        symbols: List[str],
        timeframe: str,
        limit: int
    ) -> Dict[str, pd.DataFrame]:
        try:
            results = {}
            for symbol in symbols:
                key = f"batch_{symbol}_{timeframe}"
                cached_data = self._get_from_cache(key)
                if cached_data is not None:
                    results[symbol] = cached_data
                else:
                    data = self.fetch_data(symbol, timeframe, limit)
                    if data is not None:
                        results[symbol] = data
                        self._add_to_cache(key, data)
            return results
        except Exception as e:
            logger.error(f"Error fetching batch: {str(e)}")
            return {}

    def _get_from_cache(self, key: str) -> Optional[pd.DataFrame]:
        with self._cache_lock:
            if key in self._cache:
                data, timestamp = self._cache[key]
                if datetime.now().timestamp() - timestamp < 300:
                    return data
                del self._cache[key]
        return None

    def _add_to_cache(self, key: str, data: pd.DataFrame) -> None:
        with self._cache_lock:
            self._cache[key] = (data, datetime.now().timestamp())

    def get_cache_stats(self) -> Dict[str, int]:
        with self._cache_lock:
            total_entries = len(self._cache)
            expired_entries = sum(
                1 for _, (_, timestamp) in self._cache.items()
                if datetime.now().timestamp() - timestamp >= 300
            )
            return {
                'total_entries': total_entries,
                'active_entries': total_entries - expired_entries,
                'expired_entries': expired_entries
            }

    def clear_cache(self, pattern: Optional[str] = None) -> None:
        with self._cache_lock:
            if pattern:
                keys_to_delete = [
                    key for key in list(self._cache.keys())
                    if pattern in key
                ]
                for key in keys_to_delete:
                    del self._cache[key]
            else:
                self._cache.clear()
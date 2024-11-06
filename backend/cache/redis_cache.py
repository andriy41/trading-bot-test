# cache/redis_cache.py

import redis.asyncio as redis
import json
import pickle
from typing import Any, Optional, Union
import asyncio
from backend.appfiles.config import Config
from utils.logger import setup_logger

logger = setup_logger()

class RedisCache:
    """Asynchronous Redis cache handler."""

    def __init__(self, url: str = None):
        self.url = url or f"redis://{Config.REDIS.HOST}:{Config.REDIS.PORT}/{Config.REDIS.DB}"
        self._client = None

    async def _get_client(self):
        if self._client is None:
            self._client = redis.Redis.from_url(
                self.url,
                password=getattr(Config.REDIS, 'PASSWORD', None),
                decode_responses=True
            )
        return self._client

    async def get(self, key: str, default: Any = None) -> Any:
        """
        Retrieve data from cache with automatic deserialization.
        """
        try:
            client = await self._get_client()
            data = await client.get(key)
            if data is None:
                return default
            return self._deserialize(data)
        except Exception as e:
            logger.error(f"Cache get error: {str(e)}")
            return default

    async def set(
        self,
        key: str,
        value: Any,
        expire: int = 3600,
        nx: bool = False
    ) -> bool:
        """
        Store data in cache with automatic serialization.
        """
        try:
            client = await self._get_client()
            serialized_value = self._serialize(value)
            result = await client.set(
                name=key,
                value=serialized_value,
                ex=expire,
                nx=nx
            )
            return result
        except Exception as e:
            logger.error(f"Cache set error: {str(e)}")
            return False

    async def mget(self, keys: list) -> list:
        """
        Retrieve multiple keys efficiently.
        """
        try:
            client = await self._get_client()
            values = await client.mget(*keys)
            return [self._deserialize(v) if v else None for v in values]
        except Exception as e:
            logger.error(f"Cache mget error: {str(e)}")
            return [None] * len(keys)

    async def pipeline_set(self, mapping: dict, expire: int = 3600) -> bool:
        """
        Batch set operations using pipeline.
        """
        try:
            client = await self._get_client()
            async with client.pipeline() as pipe:
                for key, value in mapping.items():
                    serialized_value = self._serialize(value)
                    pipe.set(name=key, value=serialized_value, ex=expire)
                await pipe.execute()
            return True
        except Exception as e:
            logger.error(f"Pipeline set error: {str(e)}")
            return False

    async def get_or_compute(
        self,
        key: str,
        compute_func,
        expire: int = 3600
    ) -> Any:
        """
        Get from cache or compute and store if missing.
        """
        value = await self.get(key)
        if value is None:
            value = await compute_func()
            await self.set(key, value, expire=expire)
        return value

    async def invalidate_pattern(self, pattern: str) -> int:
        """
        Invalidate all keys matching pattern.
        """
        try:
            client = await self._get_client()
            keys = await client.keys(pattern)
            if keys:
                count = await client.delete(*keys)
                return count
            return 0
        except Exception as e:
            logger.error(f"Pattern invalidation error: {str(e)}")
            return 0

    def _serialize(self, value: Any) -> Union[str, bytes]:
        """
        Serialize data for storage.
        """
        if isinstance(value, (dict, list)):
            return json.dumps(value)
        elif isinstance(value, (str, int, float)):
            return str(value)
        else:
            return pickle.dumps(value)

    def _deserialize(self, value: Union[str, bytes]) -> Any:
        """
        Deserialize data from storage.
        """
        if isinstance(value, bytes):
            value = value.decode('utf-8')
        try:
            return json.loads(value)
        except (json.JSONDecodeError, TypeError):
            try:
                return pickle.loads(value.encode('utf-8'))
            except Exception:
                return value

    async def close(self):
        """
        Clean up resources.
        """
        if self._client:
            await self._client.close()
            self._client = None


class CacheManager:
    def __init__(self, redis_cache: RedisCache):
        self.cache = redis_cache
        self.prefix = getattr(Config, 'CACHE_KEY_PREFIX', 'cache')

    def get_key(self, *parts: str) -> str:
        """
        Generate standardized cache keys.
        """
        return f"{self.prefix}:{':'.join(parts)}"

    async def warm_cache(self, keys: list, compute_funcs: dict):
        """
        Pre-warm cache with computed values.
        """
        tasks = []
        for key in keys:
            if key in compute_funcs:
                tasks.append(
                    self.cache.get_or_compute(key, compute_funcs[key])
                )
        await asyncio.gather(*tasks)

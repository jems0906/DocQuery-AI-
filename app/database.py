from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool
from contextlib import contextmanager
import redis
from typing import Generator
from .config import get_settings
from .models import Base
from loguru import logger

settings = get_settings()

# SQLAlchemy Database Setup
if settings.database_url.startswith("sqlite"):
    engine = create_engine(
        settings.database_url,
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
        echo=settings.debug
    )
else:
    engine = create_engine(
        settings.database_url,
        echo=settings.debug
    )

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Redis Setup
try:
    redis_client = redis.from_url(settings.redis_url, decode_responses=True)
    redis_client.ping()
    logger.info("Redis connection established")
except Exception as e:
    logger.warning(f"Redis connection failed: {e}. Using memory cache fallback.")
    redis_client = None


def create_tables():
    """Create all database tables"""
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables created successfully")


def get_db() -> Generator[Session, None, None]:
    """Dependency to get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@contextmanager
def get_db_session() -> Generator[Session, None, None]:
    """Context manager for database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


class CacheManager:
    """Cache management with Redis fallback to memory"""
    
    def __init__(self):
        self._memory_cache = {}
        self.redis_client = redis_client
    
    async def get(self, key: str) -> str | None:
        """Get value from cache"""
        if self.redis_client:
            try:
                return self.redis_client.get(key)
            except Exception as e:
                logger.warning(f"Redis get failed: {e}")
        
        return self._memory_cache.get(key)
    
    async def set(self, key: str, value: str, ttl: int = None) -> bool:
        """Set value in cache"""
        if self.redis_client:
            try:
                if ttl:
                    return self.redis_client.setex(key, ttl, value)
                else:
                    return self.redis_client.set(key, value)
            except Exception as e:
                logger.warning(f"Redis set failed: {e}")
        
        # Fallback to memory cache
        self._memory_cache[key] = value
        return True
    
    async def delete(self, key: str) -> bool:
        """Delete key from cache"""
        if self.redis_client:
            try:
                return bool(self.redis_client.delete(key))
            except Exception as e:
                logger.warning(f"Redis delete failed: {e}")
        
        return bool(self._memory_cache.pop(key, None))
    
    async def clear(self) -> bool:
        """Clear all cache"""
        if self.redis_client:
            try:
                self.redis_client.flushdb()
            except Exception as e:
                logger.warning(f"Redis clear failed: {e}")
        
        self._memory_cache.clear()
        return True


# Global cache manager instance
cache_manager = CacheManager()
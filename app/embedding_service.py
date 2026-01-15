import asyncio
import numpy as np
from typing import List, Dict, Any, Optional, Union
from abc import ABC, abstractmethod
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer
import openai
from openai import AsyncOpenAI
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential
from .config import get_settings

settings = get_settings()


@dataclass
class EmbeddingResult:
    embeddings: List[List[float]]
    model_name: str
    dimension: int
    processing_time: float


class BaseEmbeddingProvider(ABC):
    """Abstract base class for embedding providers"""
    
    @abstractmethod
    async def encode(self, texts: List[str]) -> EmbeddingResult:
        pass
    
    @abstractmethod
    def get_dimension(self) -> int:
        pass
    
    @abstractmethod
    def get_model_name(self) -> str:
        pass


class OpenAIEmbeddingProvider(BaseEmbeddingProvider):
    """OpenAI embedding provider with retry logic"""
    
    def __init__(self):
        if not settings.openai_api_key:
            raise ValueError("OpenAI API key is required")
        
        self.client = AsyncOpenAI(api_key=settings.openai_api_key)
        self.model_name = settings.openai_model
        self._dimension = 1536 if "3-small" in self.model_name else 1536
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    async def encode(self, texts: List[str]) -> EmbeddingResult:
        start_time = asyncio.get_event_loop().time()
        
        try:
            # Process in batches to avoid token limits
            batch_size = settings.batch_size
            all_embeddings = []
            
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                
                response = await self.client.embeddings.create(
                    input=batch_texts,
                    model=self.model_name
                )
                
                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)
            
            processing_time = asyncio.get_event_loop().time() - start_time
            
            return EmbeddingResult(
                embeddings=all_embeddings,
                model_name=self.model_name,
                dimension=self._dimension,
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"OpenAI embedding failed: {e}")
            raise
    
    def get_dimension(self) -> int:
        return self._dimension
    
    def get_model_name(self) -> str:
        return self.model_name


class HuggingFaceEmbeddingProvider(BaseEmbeddingProvider):
    """Hugging Face Sentence Transformers provider"""
    
    def __init__(self, model_name: str = None):
        self.model_name = model_name or settings.embedding_model
        self.model = None
        self._dimension = None
    
    async def _load_model(self):
        """Lazy load model to avoid blocking startup"""
        if self.model is None:
            logger.info(f"Loading embedding model: {self.model_name}")
            loop = asyncio.get_event_loop()
            self.model = await loop.run_in_executor(
                None, 
                lambda: SentenceTransformer(self.model_name)
            )
            self._dimension = self.model.get_sentence_embedding_dimension()
            logger.info(f"Model loaded. Dimension: {self._dimension}")
    
    async def encode(self, texts: List[str]) -> EmbeddingResult:
        await self._load_model()
        start_time = asyncio.get_event_loop().time()
        
        try:
            # Process in executor to avoid blocking
            loop = asyncio.get_event_loop()
            embeddings = await loop.run_in_executor(
                None, 
                lambda: self.model.encode(texts, show_progress_bar=False).tolist()
            )
            
            processing_time = asyncio.get_event_loop().time() - start_time
            
            return EmbeddingResult(
                embeddings=embeddings,
                model_name=self.model_name,
                dimension=self._dimension,
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"HuggingFace embedding failed: {e}")
            raise
    
    def get_dimension(self) -> int:
        if self._dimension is None:
            # Estimate dimension based on model name
            if "MiniLM" in self.model_name:
                return 384
            elif "base" in self.model_name:
                return 768
            else:
                return 512
        return self._dimension
    
    def get_model_name(self) -> str:
        return self.model_name


class EmbeddingService:
    """Main embedding service with provider fallback"""
    
    def __init__(self):
        self.providers = []
        self._setup_providers()
        self.primary_provider = self.providers[0] if self.providers else None
    
    def _setup_providers(self):
        """Initialize embedding providers with fallback order"""
        # Try OpenAI first if API key is available
        if settings.openai_api_key:
            try:
                self.providers.append(OpenAIEmbeddingProvider())
                logger.info("OpenAI embedding provider initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize OpenAI provider: {e}")
        
        # Add HuggingFace as fallback
        try:
            self.providers.append(HuggingFaceEmbeddingProvider())
            logger.info("HuggingFace embedding provider initialized")
        except Exception as e:
            logger.error(f"Failed to initialize HuggingFace provider: {e}")
        
        if not self.providers:
            raise RuntimeError("No embedding providers available")
    
    async def encode_texts(self, texts: List[str]) -> EmbeddingResult:
        """Encode texts using primary provider with fallback"""
        if not texts:
            return EmbeddingResult([], "none", 0, 0.0)
        
        for i, provider in enumerate(self.providers):
            try:
                logger.debug(f"Using embedding provider {i + 1}: {provider.get_model_name()}")
                result = await provider.encode(texts)
                
                # Update primary provider if fallback was used
                if i > 0:
                    self.primary_provider = provider
                    logger.info(f"Switched to provider: {provider.get_model_name()}")
                
                return result
                
            except Exception as e:
                logger.warning(f"Provider {provider.get_model_name()} failed: {e}")
                if i == len(self.providers) - 1:
                    raise Exception(f"All embedding providers failed. Last error: {e}")
                continue
    
    async def encode_query(self, query: str) -> List[float]:
        """Encode single query text"""
        result = await self.encode_texts([query])
        return result.embeddings[0] if result.embeddings else []
    
    async def encode_documents(self, documents: List[str]) -> List[List[float]]:
        """Encode multiple documents"""
        result = await self.encode_texts(documents)
        return result.embeddings
    
    def get_dimension(self) -> int:
        """Get embedding dimension from primary provider"""
        if self.primary_provider:
            return self.primary_provider.get_dimension()
        return settings.index_dimension  # fallback
    
    def get_model_name(self) -> str:
        """Get current model name"""
        if self.primary_provider:
            return self.primary_provider.get_model_name()
        return "unknown"


# Global embedding service instance
embedding_service = EmbeddingService()
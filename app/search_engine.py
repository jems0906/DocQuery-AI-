import asyncio
import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from loguru import logger
from .config import get_settings
from .embedding_service import embedding_service
from .vector_store import VectorStoreManager, SearchResult
from .query_processor import QueryProcessor, ProcessedQuery
from .ranking_system import hybrid_ranker, RankedResult
from .database import get_db_session
from .models import QueryLog, DocumentChunk

settings = get_settings()


@dataclass
class SearchResponse:
    query: str
    processed_query: ProcessedQuery
    results: List[RankedResult]
    total_results: int
    search_time_ms: float
    processing_time_ms: float
    total_time_ms: float
    used_reranking: bool
    metadata: Dict[str, Any]


class SearchEngine:
    """Main search engine orchestrating all components"""
    
    def __init__(self):
        self.vector_store = None
        self.query_processor = QueryProcessor()
        self.embedding_service = embedding_service
        self.ranker = hybrid_ranker
        
        # Performance tracking
        self.stats = {
            'total_searches': 0,
            'avg_search_time': 0.0,
            'cache_hits': 0,
            'cache_misses': 0
        }
    
    async def initialize(self):
        """Initialize search engine components"""
        try:
            # Initialize vector store
            embedding_dim = self.embedding_service.get_dimension()
            self.vector_store = VectorStoreManager(embedding_dim)
            
            logger.info(f"Search engine initialized with embedding dimension: {embedding_dim}")
            logger.info(f"Using embedding model: {self.embedding_service.get_model_name()}")
            
        except Exception as e:
            logger.error(f"Search engine initialization failed: {e}")
            raise
    
    async def search(self, query: str, document_id: Optional[int] = None, 
                    max_results: int = 5, use_reranking: bool = True) -> SearchResponse:
        """Perform comprehensive search with ranking"""
        start_time = time.time()
        
        try:
            # Initialize if needed
            if self.vector_store is None:
                await self.initialize()
            
            # Process query
            processing_start = time.time()
            processed_query = await self.query_processor.process_query(query)
            processing_time = (time.time() - processing_start) * 1000
            
            logger.debug(f"Query processed: {processed_query.query_type.value}, confidence: {processed_query.confidence:.3f}")
            
            # Generate query embedding
            search_start = time.time()
            query_embedding = await self.embedding_service.encode_query(processed_query.expanded_query)
            
            if not query_embedding:
                return self._empty_response(query, processed_query, processing_time)
            
            # Vector search
            search_results = await self.vector_store.search_similar(
                query_embedding, 
                top_k=settings.search_top_k,
                document_id=document_id
            )
            
            search_time = (time.time() - search_start) * 1000
            
            if not search_results:
                return self._empty_response(query, processed_query, processing_time, search_time)
            
            # Rank results
            ranking_start = time.time()
            
            # Decide whether to use reranking based on query complexity and result count
            should_rerank = (
                use_reranking and 
                len(search_results) > 2 and 
                processed_query.confidence > 0.6 and
                len(search_results) <= settings.search_top_k
            )
            
            ranked_results = await self.ranker.rank_results(
                processed_query, search_results, use_reranking=should_rerank
            )
            
            # Limit results
            final_results = ranked_results[:max_results]
            
            ranking_time = (time.time() - ranking_start) * 1000
            total_time = (time.time() - start_time) * 1000
            
            # Log query
            asyncio.create_task(self._log_query(
                query, processed_query, final_results, search_time, total_time, document_id
            ))
            
            # Update stats
            self.stats['total_searches'] += 1
            self.stats['avg_search_time'] = (
                (self.stats['avg_search_time'] * (self.stats['total_searches'] - 1) + total_time) / 
                self.stats['total_searches']
            )
            
            logger.info(
                f"Search completed: {len(final_results)} results, "
                f"processing: {processing_time:.1f}ms, search: {search_time:.1f}ms, "
                f"ranking: {ranking_time:.1f}ms, total: {total_time:.1f}ms"
            )
            
            return SearchResponse(
                query=query,
                processed_query=processed_query,
                results=final_results,
                total_results=len(search_results),
                search_time_ms=search_time,
                processing_time_ms=processing_time,
                total_time_ms=total_time,
                used_reranking=should_rerank,
                metadata={
                    'embedding_model': self.embedding_service.get_model_name(),
                    'vector_store_stats': await self.vector_store.get_stats(),
                    'ranking_time_ms': ranking_time
                }
            )
            
        except Exception as e:
            logger.error(f"Search failed for query '{query}': {e}")
            total_time = (time.time() - start_time) * 1000
            return self._empty_response(query, processed_query, 0, 0, total_time, str(e))
    
    def _empty_response(self, query: str, processed_query: ProcessedQuery = None,
                       processing_time: float = 0, search_time: float = 0, 
                       total_time: float = 0, error: str = None) -> SearchResponse:
        """Create empty search response"""
        if processed_query is None:
            processed_query = ProcessedQuery(
                original_query=query,
                cleaned_query=query,
                query_type="general",
                key_terms=[],
                entities=[],
                intent_keywords=[],
                question_words=[],
                expanded_query=query,
                confidence=0.0,
                metadata={}
            )
        
        return SearchResponse(
            query=query,
            processed_query=processed_query,
            results=[],
            total_results=0,
            search_time_ms=search_time,
            processing_time_ms=processing_time,
            total_time_ms=total_time,
            used_reranking=False,
            metadata={'error': error} if error else {}
        )
    
    async def _log_query(self, query: str, processed_query: ProcessedQuery,
                        results: List[RankedResult], search_time: float,
                        total_time: float, document_id: Optional[int] = None):
        """Log query for analytics"""
        try:
            import hashlib
            
            query_hash = hashlib.sha256(query.encode('utf-8')).hexdigest()
            relevant_chunks = [r.search_result.chunk_id for r in results]
            
            with get_db_session() as db:
                log_entry = QueryLog(
                    query_text=query,
                    query_hash=query_hash,
                    document_id=document_id,
                    relevant_chunks=relevant_chunks,
                    search_time_ms=search_time,
                    total_time_ms=total_time
                )
                db.add(log_entry)
                db.commit()
                
        except Exception as e:
            logger.warning(f"Query logging failed: {e}")
    
    async def add_document(self, document_id: int, chunks: List[Dict[str, Any]]) -> bool:
        """Add document chunks to search index"""
        try:
            if self.vector_store is None:
                await self.initialize()
            
            # Extract text content for embedding
            texts = [chunk['content'] for chunk in chunks]
            
            if not texts:
                return True
            
            # Generate embeddings
            embeddings = await self.embedding_service.encode_documents(texts)
            
            if not embeddings:
                raise Exception("Failed to generate embeddings")
            
            # Add to vector store
            success = await self.vector_store.add_document_chunks(
                document_id, chunks, embeddings
            )
            
            if success:
                # Update ranker corpus (for TF-IDF)
                await self.ranker.fit_corpus(texts)
                logger.info(f"Added {len(chunks)} chunks for document {document_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to add document {document_id} to search index: {e}")
            return False
    
    async def remove_document(self, document_id: int) -> bool:
        """Remove document from search index"""
        try:
            if self.vector_store is None:
                return True
            
            success = await self.vector_store.delete_document(document_id)
            
            if success:
                logger.info(f"Removed document {document_id} from search index")
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to remove document {document_id} from search index: {e}")
            return False
    
    async def get_search_stats(self) -> Dict[str, Any]:
        """Get search engine statistics"""
        try:
            vector_stats = await self.vector_store.get_stats() if self.vector_store else {}
            
            return {
                'search_stats': self.stats,
                'vector_store_stats': vector_stats,
                'embedding_model': self.embedding_service.get_model_name(),
                'embedding_dimension': self.embedding_service.get_dimension()
            }
            
        except Exception as e:
            logger.error(f"Failed to get search stats: {e}")
            return {'error': str(e)}
    
    async def suggest_queries(self, partial_query: str, limit: int = 5) -> List[str]:
        """Suggest query completions (basic implementation)"""
        try:
            # Simple suggestion based on recent queries
            with get_db_session() as db:
                recent_queries = db.query(QueryLog.query_text).filter(
                    QueryLog.query_text.ilike(f"%{partial_query}%")
                ).distinct().limit(limit * 2).all()
                
                suggestions = [
                    q[0] for q in recent_queries 
                    if q[0].lower().startswith(partial_query.lower())
                ][:limit]
                
                return suggestions
                
        except Exception as e:
            logger.warning(f"Query suggestions failed: {e}")
            return []


# Global search engine instance
search_engine = SearchEngine()
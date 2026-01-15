import pytest
import asyncio
from app.search_engine import SearchEngine
from app.query_processor import QueryProcessor
from app.embedding_service import embedding_service


class TestSearchEngine:
    """Test cases for the search engine"""
    
    @pytest.fixture
    def search_engine(self):
        """Create search engine instance"""
        engine = SearchEngine()
        return engine
    
    @pytest.fixture
    def sample_documents(self):
        """Sample documents for testing"""
        return [
            {
                'content': 'This is a document about artificial intelligence and machine learning.',
                'document_id': 1,
                'chunk_index': 0,
                'page_number': 1,
                'token_count': 12
            },
            {
                'content': 'Python is a popular programming language for data science.',
                'document_id': 2,
                'chunk_index': 0,
                'page_number': 1,
                'token_count': 10
            }
        ]
    
    @pytest.mark.asyncio
    async def test_search_initialization(self, search_engine):
        """Test search engine initialization"""
        await search_engine.initialize()
        
        assert search_engine.vector_store is not None
        assert search_engine.query_processor is not None
        assert search_engine.embedding_service is not None
    
    @pytest.mark.asyncio
    async def test_document_indexing(self, search_engine, sample_documents):
        """Test document indexing"""
        await search_engine.initialize()
        
        # Add documents to index
        success = await search_engine.add_document(1, sample_documents[:1])
        assert success is True
        
        # Check vector store stats
        stats = await search_engine.get_search_stats()
        assert stats['vector_store_stats']['total_chunks'] >= 1
    
    @pytest.mark.asyncio
    async def test_search_functionality(self, search_engine, sample_documents):
        """Test search functionality"""
        await search_engine.initialize()
        
        # Add documents
        await search_engine.add_document(1, sample_documents)
        
        # Perform search
        response = await search_engine.search(
            query="What is artificial intelligence?",
            max_results=5
        )
        
        assert response is not None
        assert response.query == "What is artificial intelligence?"
        assert len(response.results) >= 0
        assert response.total_time_ms > 0


class TestQueryProcessor:
    """Test cases for query processing"""
    
    @pytest.fixture
    def query_processor(self):
        """Create query processor instance"""
        return QueryProcessor()
    
    @pytest.mark.asyncio
    async def test_query_processing(self, query_processor):
        """Test query processing"""
        query = "What is machine learning?"
        
        processed = await query_processor.process_query(query)
        
        assert processed.original_query == query
        assert processed.cleaned_query is not None
        assert processed.confidence > 0
        assert len(processed.key_terms) > 0
    
    @pytest.mark.asyncio
    async def test_query_type_classification(self, query_processor):
        """Test query type classification"""
        queries = [
            "What is artificial intelligence?",  # DEFINITIONAL
            "How do I create a neural network?",  # PROCEDURAL
            "Compare Python and Java",  # COMPARATIVE
            "Why does this algorithm work?",  # ANALYTICAL
        ]
        
        for query in queries:
            processed = await query_processor.process_query(query)
            assert processed.query_type is not None
            assert processed.confidence > 0


class TestEmbeddingService:
    """Test cases for embedding service"""
    
    @pytest.mark.asyncio
    async def test_text_encoding(self):
        """Test text encoding"""
        texts = ["This is a test", "Another test sentence"]
        
        embeddings = await embedding_service.encode_documents(texts)
        
        assert len(embeddings) == len(texts)
        assert all(len(emb) > 0 for emb in embeddings)
        assert all(isinstance(emb[0], float) for emb in embeddings)
    
    @pytest.mark.asyncio
    async def test_query_encoding(self):
        """Test query encoding"""
        query = "What is machine learning?"
        
        embedding = await embedding_service.encode_query(query)
        
        assert len(embedding) > 0
        assert isinstance(embedding[0], float)
    
    def test_model_info(self):
        """Test model information"""
        model_name = embedding_service.get_model_name()
        dimension = embedding_service.get_dimension()
        
        assert isinstance(model_name, str)
        assert isinstance(dimension, int)
        assert dimension > 0


if __name__ == "__main__":
    pytest.main(["-v", __file__])
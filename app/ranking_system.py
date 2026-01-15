import asyncio
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from sentence_transformers import CrossEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from loguru import logger
from .config import get_settings
from .vector_store import SearchResult
from .query_processor import ProcessedQuery

settings = get_settings()


@dataclass
class RankedResult:
    search_result: SearchResult
    relevance_score: float
    ranking_features: Dict[str, float]
    final_rank: int


class FeatureExtractor:
    """Extract ranking features from query-document pairs"""
    
    def __init__(self):
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=1000,
            ngram_range=(1, 2),
            stop_words='english',
            lowercase=True
        )
        self._fitted = False
    
    def fit(self, documents: List[str]):
        """Fit TF-IDF vectorizer on document corpus"""
        if documents:
            try:
                self.tfidf_vectorizer.fit(documents)
                self._fitted = True
                logger.debug(f"TF-IDF fitted on {len(documents)} documents")
            except Exception as e:
                logger.warning(f"TF-IDF fitting failed: {e}")
    
    def extract_features(self, query: str, document: str, 
                        search_result: SearchResult) -> Dict[str, float]:
        """Extract ranking features for query-document pair"""
        features = {}
        
        try:
            # Basic text features
            features['doc_length'] = len(document)
            features['query_length'] = len(query)
            features['query_doc_length_ratio'] = len(query) / max(len(document), 1)
            
            # Term overlap features
            query_words = set(query.lower().split())
            doc_words = set(document.lower().split())
            
            if query_words:
                features['term_overlap'] = len(query_words & doc_words) / len(query_words)
                features['query_coverage'] = len(query_words & doc_words) / len(query_words)
            else:
                features['term_overlap'] = 0.0
                features['query_coverage'] = 0.0
            
            # Position-based features
            features['embedding_score'] = search_result.score
            features['log_embedding_score'] = np.log1p(max(0, search_result.score))
            
            # Document metadata features
            metadata = search_result.metadata or {}
            features['page_number'] = metadata.get('page_number', 1)
            features['chunk_index'] = metadata.get('chunk_index', 0)
            features['token_count'] = metadata.get('token_count', len(document.split()))
            
            # TF-IDF similarity (if fitted)
            if self._fitted and query and document:
                try:
                    query_tfidf = self.tfidf_vectorizer.transform([query])
                    doc_tfidf = self.tfidf_vectorizer.transform([document])
                    tfidf_sim = cosine_similarity(query_tfidf, doc_tfidf)[0][0]
                    features['tfidf_similarity'] = float(tfidf_sim)
                except Exception as e:
                    logger.debug(f"TF-IDF similarity calculation failed: {e}")
                    features['tfidf_similarity'] = 0.0
            else:
                features['tfidf_similarity'] = 0.0
            
            # Text quality features
            features['has_numbers'] = float(any(c.isdigit() for c in document))
            features['has_questions'] = float('?' in document)
            features['sentence_count'] = document.count('.') + document.count('!') + document.count('?')
            features['avg_word_length'] = np.mean([len(w) for w in document.split()]) if document.split() else 0
            
        except Exception as e:
            logger.warning(f"Feature extraction failed: {e}")
            # Return basic features on error
            features = {
                'embedding_score': search_result.score,
                'doc_length': len(document),
                'query_length': len(query),
                'term_overlap': 0.0,
                'tfidf_similarity': 0.0,
                'page_number': 1,
                'chunk_index': 0,
                'token_count': len(document.split())
            }
        
        return features


class CrossEncoderReranker:
    """Cross-encoder based reranking for improved relevance"""
    
    def __init__(self, model_name: str = None):
        self.model_name = model_name or settings.rerank_model
        self.model = None
        self.max_length = 512
    
    async def _load_model(self):
        """Lazy load the cross-encoder model"""
        if self.model is None:
            try:
                logger.info(f"Loading cross-encoder model: {self.model_name}")
                loop = asyncio.get_event_loop()
                self.model = await loop.run_in_executor(
                    None, lambda: CrossEncoder(self.model_name)
                )
                logger.info("Cross-encoder model loaded")
            except Exception as e:
                logger.error(f"Failed to load cross-encoder: {e}")
                raise
    
    async def rerank(self, query: str, documents: List[str], 
                    search_results: List[SearchResult]) -> List[Tuple[SearchResult, float]]:
        """Rerank documents using cross-encoder"""
        try:
            await self._load_model()
            
            if not self.model or not documents:
                return [(result, result.score) for result in search_results]
            
            # Prepare query-document pairs
            pairs = [(query, doc[:self.max_length]) for doc in documents]
            
            # Get reranking scores
            loop = asyncio.get_event_loop()
            scores = await loop.run_in_executor(
                None, lambda: self.model.predict(pairs)
            )
            
            # Combine with search results
            reranked = list(zip(search_results, scores))
            
            # Sort by reranking score
            reranked.sort(key=lambda x: x[1], reverse=True)
            
            logger.debug(f"Reranked {len(reranked)} results")
            return reranked
            
        except Exception as e:
            logger.error(f"Cross-encoder reranking failed: {e}")
            # Fallback to original scores
            return [(result, result.score) for result in search_results]


class HybridRanker:
    """Hybrid ranking system combining multiple signals"""
    
    def __init__(self):
        self.feature_extractor = FeatureExtractor()
        self.cross_encoder = CrossEncoderReranker()
        
        # Ranking weights (can be tuned)
        self.weights = {
            'embedding_score': 0.4,
            'cross_encoder_score': 0.3,
            'term_overlap': 0.1,
            'tfidf_similarity': 0.1,
            'position_penalty': 0.05,
            'quality_bonus': 0.05
        }
    
    async def fit_corpus(self, documents: List[str]):
        """Fit ranker on document corpus"""
        self.feature_extractor.fit(documents)
    
    async def rank_results(self, query: ProcessedQuery, 
                          search_results: List[SearchResult], 
                          use_reranking: bool = True) -> List[RankedResult]:
        """Rank search results using hybrid approach"""
        if not search_results:
            return []
        
        try:
            # Extract documents
            documents = [result.content for result in search_results]
            
            # Get cross-encoder scores if enabled
            if use_reranking and len(search_results) <= settings.search_top_k:
                reranked = await self.cross_encoder.rerank(
                    query.cleaned_query, documents, search_results
                )
                cross_encoder_scores = {i: score for i, (_, score) in enumerate(reranked)}
            else:
                cross_encoder_scores = {i: 0.0 for i in range(len(search_results))}
            
            # Calculate hybrid scores
            ranked_results = []
            
            for i, result in enumerate(search_results):
                # Extract features
                features = self.feature_extractor.extract_features(
                    query.cleaned_query, result.content, result
                )
                
                # Calculate hybrid score
                hybrid_score = self._calculate_hybrid_score(
                    features, cross_encoder_scores.get(i, 0.0), i
                )
                
                ranked_result = RankedResult(
                    search_result=result,
                    relevance_score=hybrid_score,
                    ranking_features=features,
                    final_rank=0  # Will be set after sorting
                )
                
                ranked_results.append(ranked_result)
            
            # Sort by hybrid score
            ranked_results.sort(key=lambda x: x.relevance_score, reverse=True)
            
            # Set final ranks
            for i, result in enumerate(ranked_results):
                result.final_rank = i + 1
            
            logger.debug(f"Ranked {len(ranked_results)} results")
            return ranked_results
            
        except Exception as e:
            logger.error(f"Ranking failed: {e}")
            # Fallback to original order
            return [
                RankedResult(
                    search_result=result,
                    relevance_score=result.score,
                    ranking_features={},
                    final_rank=i + 1
                )
                for i, result in enumerate(search_results)
            ]
    
    def _calculate_hybrid_score(self, features: Dict[str, float], 
                               cross_encoder_score: float, position: int) -> float:
        """Calculate hybrid ranking score"""
        try:
            score = 0.0
            
            # Embedding similarity score
            score += self.weights['embedding_score'] * features.get('embedding_score', 0.0)
            
            # Cross-encoder score
            score += self.weights['cross_encoder_score'] * cross_encoder_score
            
            # Term overlap bonus
            score += self.weights['term_overlap'] * features.get('term_overlap', 0.0)
            
            # TF-IDF similarity
            score += self.weights['tfidf_similarity'] * features.get('tfidf_similarity', 0.0)
            
            # Position penalty (slight preference for earlier results)
            position_penalty = np.exp(-position * 0.1)
            score += self.weights['position_penalty'] * position_penalty
            
            # Quality bonus
            quality_score = self._calculate_quality_score(features)
            score += self.weights['quality_bonus'] * quality_score
            
            return max(0.0, score)
            
        except Exception as e:
            logger.warning(f"Hybrid score calculation failed: {e}")
            return features.get('embedding_score', 0.0)
    
    def _calculate_quality_score(self, features: Dict[str, float]) -> float:
        """Calculate document quality score"""
        try:
            quality = 0.0
            
            # Length bonus (prefer medium-length documents)
            doc_length = features.get('doc_length', 0)
            if 200 <= doc_length <= 1000:
                quality += 0.2
            elif doc_length < 50:
                quality -= 0.1
            
            # Token count bonus
            token_count = features.get('token_count', 0)
            if 30 <= token_count <= 200:
                quality += 0.1
            
            # Content richness bonus
            if features.get('has_numbers', 0) > 0:
                quality += 0.05
            
            # Sentence structure bonus
            sentence_count = features.get('sentence_count', 0)
            if sentence_count > 1:
                quality += 0.05
            
            return max(0.0, min(1.0, quality))
            
        except Exception as e:
            logger.debug(f"Quality score calculation failed: {e}")
            return 0.0


# Global ranker instance
hybrid_ranker = HybridRanker()
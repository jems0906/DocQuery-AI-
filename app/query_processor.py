import re
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from loguru import logger
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from .config import get_settings

settings = get_settings()

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)


class QueryType(Enum):
    FACTUAL = "factual"          # Direct fact extraction
    COMPARATIVE = "comparative"   # Comparing entities
    PROCEDURAL = "procedural"     # How-to questions
    ANALYTICAL = "analytical"     # Analysis/reasoning
    DEFINITIONAL = "definitional" # What is X?
    GENERAL = "general"           # General questions


@dataclass
class ProcessedQuery:
    original_query: str
    cleaned_query: str
    query_type: QueryType
    key_terms: List[str]
    entities: List[str]
    intent_keywords: List[str]
    question_words: List[str]
    expanded_query: str
    confidence: float
    metadata: Dict[str, Any]


class QueryProcessor:
    """Advanced query processing with intent detection and expansion"""
    
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
        # Query type patterns
        self.query_patterns = {
            QueryType.FACTUAL: [
                r'\b(what is|what are|who is|when|where|which)\b',
                r'\b(fact|facts|information about)\b',
                r'\b(tell me about|describe)\b'
            ],
            QueryType.COMPARATIVE: [
                r'\b(compare|comparison|versus|vs|difference|differences)\b',
                r'\b(better|worse|more|less|similar|different)\b',
                r'\b(than|compared to)\b'
            ],
            QueryType.PROCEDURAL: [
                r'\b(how to|how do|how can|steps|process|procedure)\b',
                r'\b(guide|tutorial|instructions|method)\b',
                r'\b(implement|create|build|setup|configure)\b'
            ],
            QueryType.ANALYTICAL: [
                r'\b(why|analyze|analysis|explain|reason|cause)\b',
                r'\b(impact|effect|consequence|result)\b',
                r'\b(evaluate|assess|examine)\b'
            ],
            QueryType.DEFINITIONAL: [
                r'\b(define|definition|meaning|means)\b',
                r'\bwhat is\b.*\?',
                r'\b(concept|term|terminology)\b'
            ]
        }
        
        # Question words for intent detection
        self.question_words = {
            'what', 'who', 'when', 'where', 'why', 'how', 'which', 'whose'
        }
        
        # Entity extraction patterns
        self.entity_patterns = [
            r'\b[A-Z][a-z]+ [A-Z][a-z]+\b',  # Proper nouns
            r'\b\d{4}\b',                     # Years
            r'\b[A-Z]{2,}\b',                # Acronyms
            r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b'  # Title case phrases
        ]
    
    async def process_query(self, query: str) -> ProcessedQuery:
        """Process and analyze user query"""
        try:
            # Basic cleaning
            cleaned_query = self._clean_query(query)
            
            # Extract components
            key_terms = await self._extract_key_terms(cleaned_query)
            entities = self._extract_entities(cleaned_query)
            question_words = self._extract_question_words(cleaned_query)
            query_type = self._classify_query_type(cleaned_query)
            intent_keywords = self._extract_intent_keywords(cleaned_query, query_type)
            
            # Query expansion
            expanded_query = await self._expand_query(cleaned_query, key_terms)
            
            # Calculate confidence
            confidence = self._calculate_confidence(cleaned_query, query_type, key_terms)
            
            # Metadata
            metadata = {
                "query_length": len(cleaned_query),
                "word_count": len(cleaned_query.split()),
                "has_question_mark": '?' in query,
                "is_imperative": self._is_imperative(cleaned_query),
                "complexity_score": self._calculate_complexity(cleaned_query)
            }
            
            return ProcessedQuery(
                original_query=query,
                cleaned_query=cleaned_query,
                query_type=query_type,
                key_terms=key_terms,
                entities=entities,
                intent_keywords=intent_keywords,
                question_words=question_words,
                expanded_query=expanded_query,
                confidence=confidence,
                metadata=metadata
            )
            
        except Exception as e:
            logger.error(f"Query processing failed: {e}")
            # Return basic processed query
            return ProcessedQuery(
                original_query=query,
                cleaned_query=query.strip(),
                query_type=QueryType.GENERAL,
                key_terms=[query.strip()],
                entities=[],
                intent_keywords=[],
                question_words=[],
                expanded_query=query.strip(),
                confidence=0.5,
                metadata={}
            )
    
    def _clean_query(self, query: str) -> str:
        """Clean and normalize query text"""
        # Remove extra whitespace
        query = re.sub(r'\s+', ' ', query.strip())
        
        # Fix common typos and normalize
        query = query.lower()
        
        # Remove excessive punctuation but keep sentence structure
        query = re.sub(r'[!]{2,}', '!', query)
        query = re.sub(r'[?]{2,}', '?', query)
        
        return query
    
    async def _extract_key_terms(self, query: str) -> List[str]:
        """Extract important terms from query"""
        try:
            # Tokenize
            tokens = word_tokenize(query.lower())
            
            # Remove stop words and punctuation
            key_terms = [
                self.lemmatizer.lemmatize(token) 
                for token in tokens 
                if token.isalnum() and token not in self.stop_words and len(token) > 2
            ]
            
            # Add important phrases (2-3 words)
            words = query.split()
            for i in range(len(words) - 1):
                bigram = f"{words[i]} {words[i+1]}"
                if not any(word in self.stop_words for word in words[i:i+2]):
                    key_terms.append(bigram)
            
            # Remove duplicates while preserving order
            seen = set()
            unique_terms = []
            for term in key_terms:
                if term not in seen:
                    seen.add(term)
                    unique_terms.append(term)
            
            return unique_terms[:10]  # Limit to top 10 terms
            
        except Exception as e:
            logger.warning(f"Key term extraction failed: {e}")
            return query.split()[:5]
    
    def _extract_entities(self, query: str) -> List[str]:
        """Extract named entities from query"""
        entities = []
        
        for pattern in self.entity_patterns:
            matches = re.findall(pattern, query)
            entities.extend(matches)
        
        # Remove duplicates
        return list(set(entities))
    
    def _extract_question_words(self, query: str) -> List[str]:
        """Extract question words from query"""
        tokens = word_tokenize(query.lower())
        return [token for token in tokens if token in self.question_words]
    
    def _classify_query_type(self, query: str) -> QueryType:
        """Classify the type of query"""
        query_lower = query.lower()
        
        # Score each query type
        type_scores = {}
        
        for query_type, patterns in self.query_patterns.items():
            score = 0
            for pattern in patterns:
                matches = len(re.findall(pattern, query_lower))
                score += matches
            type_scores[query_type] = score
        
        # Return type with highest score
        if max(type_scores.values()) > 0:
            return max(type_scores, key=type_scores.get)
        
        return QueryType.GENERAL
    
    def _extract_intent_keywords(self, query: str, query_type: QueryType) -> List[str]:
        """Extract keywords that indicate user intent"""
        intent_keywords = []
        
        # Extract based on query type
        if query_type in self.query_patterns:
            for pattern in self.query_patterns[query_type]:
                matches = re.findall(pattern, query.lower())
                intent_keywords.extend(matches)
        
        return list(set(intent_keywords))
    
    async def _expand_query(self, query: str, key_terms: List[str]) -> str:
        """Expand query with synonyms and related terms"""
        # Simple expansion - add important terms
        expanded_parts = [query]
        
        # Add key terms as separate search components
        for term in key_terms[:3]:  # Limit expansion
            if term not in query:
                expanded_parts.append(term)
        
        return " ".join(expanded_parts)
    
    def _calculate_confidence(self, query: str, query_type: QueryType, key_terms: List[str]) -> float:
        """Calculate confidence score for query processing"""
        confidence = 0.5  # Base confidence
        
        # Boost for clear query structure
        if any(qw in query.lower() for qw in self.question_words):
            confidence += 0.2
        
        # Boost for specific query types
        if query_type != QueryType.GENERAL:
            confidence += 0.2
        
        # Boost for good key terms
        if len(key_terms) > 2:
            confidence += 0.1
        
        # Penalty for very short queries
        if len(query.split()) < 3:
            confidence -= 0.2
        
        return max(0.1, min(1.0, confidence))
    
    def _is_imperative(self, query: str) -> bool:
        """Check if query is imperative (command)"""
        imperative_words = ['show', 'tell', 'explain', 'describe', 'list', 'find', 'give']
        first_word = query.split()[0].lower() if query.split() else ""
        return first_word in imperative_words
    
    def _calculate_complexity(self, query: str) -> float:
        """Calculate query complexity score"""
        words = query.split()
        
        # Base complexity on word count
        complexity = len(words) / 20.0  # Normalize to 0-1 range
        
        # Add complexity for question words
        question_count = sum(1 for word in words if word.lower() in self.question_words)
        complexity += question_count * 0.1
        
        # Add complexity for conjunctions
        conjunctions = ['and', 'or', 'but', 'however', 'therefore']
        conjunction_count = sum(1 for word in words if word.lower() in conjunctions)
        complexity += conjunction_count * 0.15
        
        return min(1.0, complexity)
    
    def get_search_terms(self, processed_query: ProcessedQuery) -> List[str]:
        """Get optimized search terms for vector search"""
        search_terms = []
        
        # Always include the cleaned query
        search_terms.append(processed_query.cleaned_query)
        
        # Add key terms
        search_terms.extend(processed_query.key_terms[:5])
        
        # Add entities
        search_terms.extend(processed_query.entities[:3])
        
        # Add expanded query if different
        if processed_query.expanded_query != processed_query.cleaned_query:
            search_terms.append(processed_query.expanded_query)
        
        return search_terms
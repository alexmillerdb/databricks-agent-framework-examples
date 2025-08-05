"""
Comprehensive metrics module for DSPy RAG optimization.

This module provides evaluation metrics for different aspects of RAG performance:
- Retrieval quality metrics
- Answer generation metrics  
- End-to-end system metrics
- Query rewriting effectiveness metrics
"""

import re
import string
from typing import List, Dict, Any, Optional, Union
from collections import Counter
import dspy


class RetrievalRelevanceMetric:
    """Measures how relevant retrieved documents are to the query."""
    
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
    
    def __call__(self, example, pred, trace=None) -> Union[bool, float]:
        """
        Evaluate retrieval relevance.
        
        Args:
            example: DSPy example with question and expected answer
            pred: Prediction with response
            trace: Optional trace information
            
        Returns:
            Boolean or float score for relevance
        """
        # This is a simplified implementation - in practice you'd use 
        # semantic similarity or other relevance measures
        if hasattr(pred, 'context') and pred.context:
            # Check if response uses retrieved context
            response = pred.response.lower()
            context_used = any(
                snippet.lower() in response 
                for snippet in pred.context.split('\n\n')[:3]  # Check first 3 snippets
            )
            return context_used
        return False


class CitationAccuracyMetric:
    """Validates proper citation format and accuracy."""
    
    def __init__(self, require_citations: bool = True):
        self.require_citations = require_citations
        self.citation_pattern = re.compile(r'\[(\d+)\]')
    
    def __call__(self, example, pred, trace=None) -> Union[bool, float]:
        """
        Evaluate citation accuracy.
        
        Args:
            example: DSPy example with question and expected answer
            pred: Prediction with response
            trace: Optional trace information
            
        Returns:
            Boolean or float score for citation accuracy
        """
        response = pred.response
        citations = self.citation_pattern.findall(response)
        
        if not citations:
            return not self.require_citations
        
        # Check that all citations are valid numbers
        try:
            citation_nums = [int(c) for c in citations]
            # Citations should start from 1 and be sequential
            max_citation = max(citation_nums) if citation_nums else 0
            expected_citations = set(range(1, max_citation + 1))
            actual_citations = set(citation_nums)
            
            # All cited numbers should be valid
            return actual_citations.issubset(expected_citations)
        except ValueError:
            return False


class QueryDiversityMetric:
    """Ensures rewritten queries are meaningfully different from originals."""
    
    def __init__(self, min_similarity: float = 0.3, max_similarity: float = 0.8):
        self.min_similarity = min_similarity
        self.max_similarity = max_similarity
    
    def _jaccard_similarity(self, text1: str, text2: str) -> float:
        """Calculate Jaccard similarity between two texts."""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        return len(intersection) / len(union) if union else 0
    
    def __call__(self, example, pred, trace=None) -> Union[bool, float]:
        """
        Evaluate query diversity.
        
        Args:
            example: DSPy example with original question
            pred: Prediction with rewritten query
            trace: Optional trace information
            
        Returns:
            Boolean or float score for query diversity
        """
        if not hasattr(example, 'request') or not hasattr(pred, 'rewritten_query'):
            return True  # Skip if not applicable
        
        similarity = self._jaccard_similarity(example.request, pred.rewritten_query)
        return self.min_similarity <= similarity <= self.max_similarity


class SemanticF1Metric:
    """Comprehensive semantic F1 score for answer quality."""
    
    def __init__(self, use_fuzzy_matching: bool = True):
        self.use_fuzzy_matching = use_fuzzy_matching
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text for comparison."""
        # Convert to lowercase and remove punctuation
        text = text.lower()
        text = text.translate(str.maketrans('', '', string.punctuation))
        return ' '.join(text.split())
    
    def _extract_facts(self, text) -> List[str]:
        """Extract key facts from text or list."""
        # Handle both string and list inputs
        if isinstance(text, list):
            # If it's already a list of facts, normalize each one
            return [self._normalize_text(str(fact)) for fact in text if str(fact).strip()]
        elif isinstance(text, str):
            # Simple fact extraction - split by sentences
            sentences = [s.strip() for s in text.split('.') if s.strip()]
            return [self._normalize_text(s) for s in sentences]
        else:
            # Handle other types by converting to string
            return [self._normalize_text(str(text))] if str(text).strip() else []
    
    def _calculate_f1(self, predicted_facts: List[str], expected_facts: List[str]) -> float:
        """Calculate F1 score between predicted and expected facts."""
        if not expected_facts:
            return 1.0 if not predicted_facts else 0.0
        
        if not predicted_facts:
            return 0.0
        
        # Calculate precision and recall
        predicted_set = set(predicted_facts)
        expected_set = set(expected_facts)
        
        if self.use_fuzzy_matching:
            # Fuzzy matching based on word overlap
            matches = 0
            for pred_fact in predicted_facts:
                pred_words = set(pred_fact.split())
                for exp_fact in expected_facts:
                    exp_words = set(exp_fact.split())
                    overlap = len(pred_words.intersection(exp_words))
                    if overlap >= min(len(pred_words), len(exp_words)) * 0.5:
                        matches += 1
                        break
            
            precision = matches / len(predicted_facts)
            recall = matches / len(expected_facts)
        else:
            # Exact matching
            intersection = predicted_set.intersection(expected_set)
            precision = len(intersection) / len(predicted_set)
            recall = len(intersection) / len(expected_set)
        
        if precision + recall == 0:
            return 0.0
        
        f1 = 2 * (precision * recall) / (precision + recall)
        return f1
    
    def __call__(self, example, pred, trace=None) -> float:
        """
        Evaluate semantic F1 score.
        
        Args:
            example: DSPy example with expected response
            pred: Prediction with actual response
            trace: Optional trace information
            
        Returns:
            Float F1 score between 0 and 1
        """
        if not hasattr(example, 'response') or not hasattr(pred, 'response'):
            return 0.0
        
        predicted_facts = self._extract_facts(pred.response)
        expected_facts = self._extract_facts(example.response)
        
        return self._calculate_f1(predicted_facts, expected_facts)


class CompletenessMetric:
    """Measures if the answer addresses all parts of the question."""
    
    def __init__(self, question_keywords_threshold: float = 0.6):
        self.threshold = question_keywords_threshold
    
    def _extract_question_keywords(self, question: str) -> List[str]:
        """Extract key terms from the question."""
        # Remove question words and extract content words
        question_words = {'what', 'when', 'where', 'who', 'why', 'how', 'is', 'are', 'was', 'were', 'do', 'does', 'did'}
        words = question.lower().split()
        keywords = [w.strip(string.punctuation) for w in words if w not in question_words and len(w) > 2]
        return keywords
    
    def __call__(self, example, pred, trace=None) -> Union[bool, float]:
        """
        Evaluate answer completeness.
        
        Args:
            example: DSPy example with question
            pred: Prediction with response
            trace: Optional trace information
            
        Returns:
            Boolean or float score for completeness
        """
        if not hasattr(example, 'request') or not hasattr(pred, 'response'):
            return False
        
        question_keywords = self._extract_question_keywords(example.request)
        if not question_keywords:
            return True
        
        response_text = pred.response.lower()
        addressed_keywords = sum(1 for keyword in question_keywords if keyword in response_text)
        
        completeness_score = addressed_keywords / len(question_keywords)
        return completeness_score >= self.threshold


class EndToEndRAGMetric:
    """Comprehensive end-to-end metric combining multiple aspects."""
    
    def __init__(self, 
                 citation_weight: float = 0.2,
                 relevance_weight: float = 0.3, 
                 semantic_weight: float = 0.4,
                 completeness_weight: float = 0.1):
        self.citation_metric = CitationAccuracyMetric()
        self.relevance_metric = RetrievalRelevanceMetric()
        self.semantic_metric = SemanticF1Metric()
        self.completeness_metric = CompletenessMetric()
        
        self.weights = {
            'citation': citation_weight,
            'relevance': relevance_weight,
            'semantic': semantic_weight,
            'completeness': completeness_weight
        }
    
    def __call__(self, example, pred, trace=None) -> float:
        """
        Evaluate comprehensive end-to-end performance.
        
        Args:
            example: DSPy example with question and expected answer
            pred: Prediction with response
            trace: Optional trace information
            
        Returns:
            Float score between 0 and 1
        """
        scores = {}
        
        # Citation accuracy
        citation_score = self.citation_metric(example, pred, trace)
        scores['citation'] = float(citation_score) if isinstance(citation_score, bool) else citation_score
        
        # Retrieval relevance
        relevance_score = self.relevance_metric(example, pred, trace)
        scores['relevance'] = float(relevance_score) if isinstance(relevance_score, bool) else relevance_score
        
        # Semantic F1
        scores['semantic'] = self.semantic_metric(example, pred, trace)
        
        # Completeness
        completeness_score = self.completeness_metric(example, pred, trace)
        scores['completeness'] = float(completeness_score) if isinstance(completeness_score, bool) else completeness_score
        
        # Weighted average
        total_score = sum(scores[metric] * weight for metric, weight in self.weights.items())
        return total_score


# Convenience functions for common metric combinations
def get_retrieval_metrics() -> Dict[str, Any]:
    """Get metrics focused on retrieval quality."""
    return {
        'relevance': RetrievalRelevanceMetric(),
        'query_diversity': QueryDiversityMetric()
    }


def get_generation_metrics() -> Dict[str, Any]:
    """Get metrics focused on answer generation quality."""
    return {
        'citation_accuracy': CitationAccuracyMetric(),
        'semantic_f1': SemanticF1Metric(),
        'completeness': CompletenessMetric()
    }


def get_comprehensive_metric() -> EndToEndRAGMetric:
    """Get the comprehensive end-to-end metric."""
    return EndToEndRAGMetric()


# DSPy-compatible metric functions (return bool for DSPy optimizers)
def citation_accuracy_bool(example, pred, trace=None) -> bool:
    """Boolean version of citation accuracy for DSPy optimizers."""
    metric = CitationAccuracyMetric()
    result = metric(example, pred, trace)
    return bool(result) if isinstance(result, (int, float)) else result


def semantic_f1_bool(example, pred, trace=None) -> bool:
    """Boolean version of semantic F1 for DSPy optimizers."""
    metric = SemanticF1Metric()
    result = metric(example, pred, trace)
    return result >= 0.7  # Threshold for boolean conversion


def end_to_end_bool(example, pred, trace=None) -> bool:
    """Boolean version of end-to-end metric for DSPy optimizers."""
    metric = EndToEndRAGMetric()
    result = metric(example, pred, trace)
    return result >= 0.6  # Threshold for boolean conversion
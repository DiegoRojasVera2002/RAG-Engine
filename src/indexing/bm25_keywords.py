"""
BM25-based automatic keyword extraction for document chunks.

Generates automatic metadata:
- keywords: Top-K most relevant terms per chunk
- keyword_scores: TF-IDF scores for each keyword
- global_keywords: Document-level keywords

No manual field definition required - fully automatic.
Uses statistical methods (TF-IDF) without LLM.
"""

import logging
import numpy as np
from typing import List, Dict, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from dataclasses import dataclass
import re

logger = logging.getLogger(__name__)


@dataclass
class KeywordMetadata:
    """Automatic keyword metadata for a chunk."""
    keywords: List[str]
    keyword_scores: List[float]
    top_keyword: str = None
    keyword_diversity: float = 0.0


class BM25KeywordExtractor:
    """
    BM25-based keyword extraction for RAG systems.

    Uses TF-IDF with BM25-inspired normalization to extract
    statistically significant keywords without LLM.
    """

    def __init__(
        self,
        top_k: int = 10,
        max_features: int = 2000,
        ngram_range: Tuple[int, int] = (1, 3),
        min_df: int = 1,
        max_df: float = 0.8
    ):
        """
        Initialize BM25 keyword extractor.

        Args:
            top_k: Number of keywords to extract per chunk (default: 10)
            max_features: Maximum vocabulary size (default: 2000)
            ngram_range: N-gram range (1,3) = unigrams + bigrams + trigrams
            min_df: Minimum document frequency
            max_df: Maximum document frequency (ignore very common words)
        """
        self.top_k = top_k
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.max_df = max_df

        self.vectorizer = None
        self.feature_names = None

        logger.info(
            "Initialized BM25 keyword extractor",
            extra={
                "top_k": top_k,
                "max_features": max_features,
                "ngram_range": ngram_range
            }
        )

    def _preprocess_text(self, text: str) -> str:
        """
        Preprocessing for keyword extraction.

        Removes noise (URLs, emails, phone numbers) while preserving
        technical terms and meaningful numbers.

        Args:
            text: Input text

        Returns:
            Preprocessed text
        """
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        text = re.sub(r'www\.[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', '', text)

        # Remove email addresses
        text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '', text)

        # Remove phone numbers (conservative: only obvious patterns)
        text = re.sub(r'\+?\d{1,3}[-.\s]?\(?\d{3,4}\)?[-.\s]?\d{3,4}[-.\s]?\d{3,4}', '', text)

        # Remove dates (DD/MM/YYYY, YYYY-MM-DD formats only)
        text = re.sub(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b', '', text)
        text = re.sub(r'\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b', '', text)

        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    def fit(self, texts: List[str]) -> None:
        """
        Fit TF-IDF vectorizer on document corpus.

        Args:
            texts: List of chunk texts
        """
        if not texts:
            logger.warning("No texts provided for fitting")
            return

        # Preprocess texts
        processed_texts = [self._preprocess_text(text) for text in texts]

        # Multilingual stopwords (Spanish + English common words)
        # Source: NLTK standard stopwords + domain-agnostic high-frequency terms
        multilingual_stopwords = {
            # Spanish
            'el', 'la', 'de', 'que', 'y', 'a', 'en', 'un', 'ser', 'se', 'no',
            'haber', 'por', 'con', 'su', 'para', 'como', 'estar', 'tener',
            'le', 'lo', 'todo', 'pero', 'más', 'hacer', 'o', 'poder', 'decir',
            'este', 'ir', 'otro', 'ese', 'si', 'me', 'ya', 'ver', 'porque',
            'dar', 'cuando', 'muy', 'sin', 'vez', 'mucho', 'saber', 'sobre',
            'también', 'hasta', 'dos', 'entre', 'así', 'ni', 'nos', 'llegar',
            'bien', 'poco', 'tanto', 'entonces', 'donde', 'ahora', 'después',
            'cada', 'algo', 'estos', 'estas', 'del', 'los', 'las', 'uno', 'una',
            # English
            'the', 'and', 'for', 'are', 'with', 'this', 'that', 'from', 'was',
            'have', 'has', 'had', 'not', 'but', 'can', 'will', 'been', 'were',
            'they', 'their', 'what', 'all', 'would', 'there', 'which', 'she',
            'about', 'who', 'get', 'when', 'make', 'can', 'like', 'time',
            'just', 'him', 'know', 'take', 'into', 'year', 'your', 'some',
            'could', 'them', 'see', 'other', 'than', 'then', 'now', 'look',
            'only', 'come', 'its', 'over', 'also', 'back', 'after', 'use',
        }

        # Initialize TF-IDF vectorizer with improved settings
        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            min_df=self.min_df,
            max_df=self.max_df,
            stop_words=list(multilingual_stopwords),
            lowercase=True,
            sublinear_tf=True,  # BM25-inspired: log(1 + tf)
            norm='l2',
            # Allow alphanumeric terms (preserves Python3, C++, etc.)
            token_pattern=r'(?u)\b[a-záéíóúñü][a-záéíóúñü0-9#+.-]*\b',
            strip_accents=None  # Preserve accents for Spanish
        )

        # Fit vectorizer
        try:
            self.vectorizer.fit(processed_texts)
            self.feature_names = self.vectorizer.get_feature_names_out()

            logger.info(
                "TF-IDF vectorizer fitted",
                extra={
                    "n_documents": len(texts),
                    "vocabulary_size": len(self.feature_names)
                }
            )

        except Exception as e:
            logger.error(
                "Failed to fit vectorizer",
                extra={"error": str(e)}
            )
            raise

    def extract_keywords(
        self,
        texts: List[str],
        fit_corpus: bool = True
    ) -> List[KeywordMetadata]:
        """
        Extract keywords from document chunks.

        Args:
            texts: List of chunk texts
            fit_corpus: Whether to fit vectorizer on this corpus

        Returns:
            List of keyword metadata per chunk
        """
        if not texts:
            logger.warning("No texts provided for keyword extraction")
            return []

        # Fit vectorizer if needed
        if fit_corpus or self.vectorizer is None:
            self.fit(texts)

        # Preprocess texts
        processed_texts = [self._preprocess_text(text) for text in texts]

        # Transform to TF-IDF matrix
        try:
            tfidf_matrix = self.vectorizer.transform(processed_texts)

            logger.info(
                "TF-IDF transformation complete",
                extra={
                    "n_documents": tfidf_matrix.shape[0],
                    "n_features": tfidf_matrix.shape[1]
                }
            )

        except Exception as e:
            logger.error(
                "TF-IDF transformation failed",
                extra={"error": str(e)}
            )
            raise

        # Extract keywords for each chunk
        all_metadata = []

        for idx in range(len(texts)):
            # Get TF-IDF scores for this chunk
            chunk_tfidf = tfidf_matrix[idx].toarray().flatten()

            # Get top-K keyword indices
            top_indices = chunk_tfidf.argsort()[-self.top_k:][::-1]

            # Extract keywords and scores
            keywords = [self.feature_names[i] for i in top_indices]
            scores = [float(chunk_tfidf[i]) for i in top_indices]

            # Filter out zero-score keywords
            valid_pairs = [(kw, sc) for kw, sc in zip(keywords, scores) if sc > 0]

            if valid_pairs:
                keywords, scores = zip(*valid_pairs)
                keywords = list(keywords)
                scores = list(scores)
            else:
                keywords = []
                scores = []

            # Calculate keyword diversity (entropy of scores)
            diversity = self._calculate_diversity(scores)

            metadata = KeywordMetadata(
                keywords=keywords,
                keyword_scores=scores,
                top_keyword=keywords[0] if keywords else None,
                keyword_diversity=diversity
            )

            all_metadata.append(metadata)

        logger.info(
            "Keyword extraction complete",
            extra={
                "n_chunks": len(all_metadata),
                "avg_keywords_per_chunk": np.mean([len(m.keywords) for m in all_metadata])
            }
        )

        return all_metadata

    def _calculate_diversity(self, scores: List[float]) -> float:
        """
        Calculate keyword diversity using normalized entropy.

        Higher diversity = more evenly distributed keyword importance.

        Args:
            scores: TF-IDF scores

        Returns:
            Diversity score [0, 1]
        """
        if not scores or len(scores) == 1:
            return 0.0

        # Normalize scores to probabilities
        scores_array = np.array(scores)
        total = scores_array.sum()

        if total == 0:
            return 0.0

        probs = scores_array / total

        # Calculate entropy
        entropy = -np.sum(probs * np.log2(probs + 1e-10))

        # Normalize by max entropy
        max_entropy = np.log2(len(scores))

        if max_entropy == 0:
            return 0.0

        diversity = entropy / max_entropy

        return float(diversity)

    def extract_global_keywords(
        self,
        texts: List[str],
        top_k: int = 20
    ) -> Tuple[List[str], List[float]]:
        """
        Extract document-level keywords across all chunks.

        Useful for document-level filtering and summarization.

        Args:
            texts: List of chunk texts
            top_k: Number of global keywords

        Returns:
            Tuple of (keywords, scores)
        """
        if not texts:
            return [], []

        # Fit vectorizer if needed
        if self.vectorizer is None:
            self.fit(texts)

        # Combine all texts into single document
        combined_text = " ".join(self._preprocess_text(text) for text in texts)

        # Transform to TF-IDF
        tfidf_vector = self.vectorizer.transform([combined_text])
        tfidf_scores = tfidf_vector.toarray().flatten()

        # Get top-K keywords
        top_indices = tfidf_scores.argsort()[-top_k:][::-1]

        keywords = [self.feature_names[i] for i in top_indices]
        scores = [float(tfidf_scores[i]) for i in top_indices]

        # Filter zero scores
        valid_pairs = [(kw, sc) for kw, sc in zip(keywords, scores) if sc > 0]

        if valid_pairs:
            keywords, scores = zip(*valid_pairs)
            keywords = list(keywords)
            scores = list(scores)
        else:
            keywords = []
            scores = []

        logger.info(
            "Global keywords extracted",
            extra={
                "n_keywords": len(keywords),
                "top_keyword": keywords[0] if keywords else None
            }
        )

        return keywords, scores


def extract_keywords_from_chunks(
    texts: List[str],
    top_k: int = 5,
    ngram_range: Tuple[int, int] = (1, 2)
) -> List[KeywordMetadata]:
    """
    Convenience function for keyword extraction.

    Args:
        texts: List of chunk texts
        top_k: Number of keywords per chunk
        ngram_range: N-gram range

    Returns:
        List of keyword metadata
    """
    extractor = BM25KeywordExtractor(
        top_k=top_k,
        ngram_range=ngram_range
    )

    return extractor.extract_keywords(texts)

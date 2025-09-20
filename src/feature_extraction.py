"""
Feature Extraction Module for Twitter Sentiment Analysis

This module provides various feature extraction techniques to convert
preprocessed text data into numerical features suitable for machine learning.
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline
import pickle
import os

class TextFeatureExtractor:
    """
    A comprehensive feature extractor for text data with multiple vectorization methods.
    """
    
    def __init__(self, method='tfidf', max_features=10000, ngram_range=(1, 2), 
                 min_df=2, max_df=0.95, use_idf=True, sublinear_tf=True):
        """
        Initialize the feature extractor.
        
        Args:
            method (str): Vectorization method ('tfidf', 'count', 'hashing')
            max_features (int): Maximum number of features to extract
            ngram_range (tuple): Range of n-grams to extract
            min_df (int/float): Minimum document frequency
            max_df (float): Maximum document frequency
            use_idf (bool): Whether to use inverse document frequency (TF-IDF only)
            sublinear_tf (bool): Whether to use sublinear term frequency scaling
        """
        self.method = method
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.max_df = max_df
        self.use_idf = use_idf
        self.sublinear_tf = sublinear_tf
        
        self.vectorizer = None
        self.feature_names = None
        self.is_fitted = False
        
        self._initialize_vectorizer()
    
    def _initialize_vectorizer(self):
        """Initialize the vectorizer based on the specified method."""
        if self.method == 'tfidf':
            self.vectorizer = TfidfVectorizer(
                max_features=self.max_features,
                ngram_range=self.ngram_range,
                min_df=self.min_df,
                max_df=self.max_df,
                use_idf=self.use_idf,
                sublinear_tf=self.sublinear_tf,
                stop_words='english',
                lowercase=True,
                token_pattern=r'(?u)\b\w\w+\b'
            )
        elif self.method == 'count':
            self.vectorizer = CountVectorizer(
                max_features=self.max_features,
                ngram_range=self.ngram_range,
                min_df=self.min_df,
                max_df=self.max_df,
                stop_words='english',
                lowercase=True,
                token_pattern=r'(?u)\b\w\w+\b'
            )
        elif self.method == 'hashing':
            self.vectorizer = HashingVectorizer(
                n_features=self.max_features,
                ngram_range=self.ngram_range,
                stop_words='english',
                lowercase=True,
                token_pattern=r'(?u)\b\w\w+\b'
            )
        else:
            raise ValueError(f"Unsupported method: {self.method}")
    
    def fit(self, texts):
        """
        Fit the vectorizer on the provided texts.
        
        Args:
            texts (list): List of text documents
            
        Returns:
            self: Returns the instance for method chaining
        """
        if not isinstance(texts, (list, np.ndarray, pd.Series)):
            raise ValueError("texts must be a list, numpy array, or pandas Series")
        
        # Convert to list if pandas Series
        if isinstance(texts, pd.Series):
            texts = texts.tolist()
        
        # Filter out empty texts
        texts = [text for text in texts if text and text.strip()]
        
        if not texts:
            raise ValueError("No valid texts found for fitting")
        
        # Fit the vectorizer
        self.vectorizer.fit(texts)
        
        # Store feature names (not available for hashing vectorizer)
        if hasattr(self.vectorizer, 'get_feature_names_out'):
            self.feature_names = self.vectorizer.get_feature_names_out()
        elif hasattr(self.vectorizer, 'get_feature_names'):
            self.feature_names = self.vectorizer.get_feature_names()
        
        self.is_fitted = True
        return self
    
    def transform(self, texts):
        """
        Transform texts to feature vectors.
        
        Args:
            texts (list): List of text documents
            
        Returns:
            scipy.sparse matrix: Feature matrix
        """
        if not self.is_fitted:
            raise ValueError("Vectorizer must be fitted before transform")
        
        if not isinstance(texts, (list, np.ndarray, pd.Series)):
            raise ValueError("texts must be a list, numpy array, or pandas Series")
        
        # Convert to list if pandas Series
        if isinstance(texts, pd.Series):
            texts = texts.tolist()
        
        # Handle empty texts
        texts = [text if text and text.strip() else " " for text in texts]
        
        return self.vectorizer.transform(texts)
    
    def fit_transform(self, texts):
        """
        Fit the vectorizer and transform texts in one step.
        
        Args:
            texts (list): List of text documents
            
        Returns:
            scipy.sparse matrix: Feature matrix
        """
        return self.fit(texts).transform(texts)
    
    def get_feature_names(self):
        """
        Get feature names from the fitted vectorizer.
        
        Returns:
            numpy.ndarray: Array of feature names
        """
        if not self.is_fitted:
            raise ValueError("Vectorizer must be fitted before getting feature names")
        
        return self.feature_names
    
    def get_top_features(self, feature_matrix, top_n=20):
        """
        Get top features based on their average TF-IDF scores.
        
        Args:
            feature_matrix (scipy.sparse matrix): Feature matrix
            top_n (int): Number of top features to return
            
        Returns:
            list: List of (feature_name, score) tuples
        """
        if not self.is_fitted or self.feature_names is None:
            raise ValueError("Cannot get top features for this vectorizer type")
        
        # Calculate mean scores for each feature
        mean_scores = np.array(feature_matrix.mean(axis=0)).flatten()
        
        # Get top feature indices
        top_indices = mean_scores.argsort()[-top_n:][::-1]
        
        # Return feature names and scores
        return [(self.feature_names[idx], mean_scores[idx]) for idx in top_indices]
    
    def save(self, filepath):
        """
        Save the fitted vectorizer to a file.
        
        Args:
            filepath (str): Path to save the vectorizer
        """
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted vectorizer")
        
        vectorizer_data = {
            'vectorizer': self.vectorizer,
            'method': self.method,
            'feature_names': self.feature_names,
            'is_fitted': self.is_fitted
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(vectorizer_data, f)
    
    def load(self, filepath):
        """
        Load a fitted vectorizer from a file.
        
        Args:
            filepath (str): Path to load the vectorizer from
        """
        with open(filepath, 'rb') as f:
            vectorizer_data = pickle.load(f)
        
        self.vectorizer = vectorizer_data['vectorizer']
        self.method = vectorizer_data['method']
        self.feature_names = vectorizer_data['feature_names']
        self.is_fitted = vectorizer_data['is_fitted']

class AdvancedFeatureExtractor:
    """
    Advanced feature extractor with dimensionality reduction and multiple feature types.
    """
    
    def __init__(self, base_extractor, use_svd=False, svd_components=100):
        """
        Initialize advanced feature extractor.
        
        Args:
            base_extractor (TextFeatureExtractor): Base feature extractor
            use_svd (bool): Whether to apply SVD for dimensionality reduction
            svd_components (int): Number of SVD components
        """
        self.base_extractor = base_extractor
        self.use_svd = use_svd
        self.svd_components = svd_components
        
        self.svd = None
        if use_svd:
            self.svd = TruncatedSVD(n_components=svd_components, random_state=42)
        
        self.is_fitted = False
    
    def fit(self, texts):
        """Fit the advanced feature extractor."""
        # Fit base extractor
        feature_matrix = self.base_extractor.fit_transform(texts)
        
        # Fit SVD if enabled
        if self.use_svd:
            self.svd.fit(feature_matrix)
        
        self.is_fitted = True
        return self
    
    def transform(self, texts):
        """Transform texts using the advanced feature extractor."""
        if not self.is_fitted:
            raise ValueError("Advanced extractor must be fitted before transform")
        
        # Get base features
        feature_matrix = self.base_extractor.transform(texts)
        
        # Apply SVD if enabled
        if self.use_svd:
            feature_matrix = self.svd.transform(feature_matrix)
        
        return feature_matrix
    
    def fit_transform(self, texts):
        """Fit and transform in one step."""
        return self.fit(texts).transform(texts)

class MultiFeatureExtractor:
    """
    Extract multiple types of features and combine them.
    """
    
    def __init__(self, feature_types=['tfidf', 'count'], **kwargs):
        """
        Initialize multi-feature extractor.
        
        Args:
            feature_types (list): List of feature types to extract
            **kwargs: Arguments passed to individual extractors
        """
        self.feature_types = feature_types
        self.extractors = {}
        self.is_fitted = False
        
        # Initialize extractors
        for feature_type in feature_types:
            self.extractors[feature_type] = TextFeatureExtractor(
                method=feature_type, **kwargs
            )
    
    def fit(self, texts):
        """Fit all extractors."""
        for extractor in self.extractors.values():
            extractor.fit(texts)
        
        self.is_fitted = True
        return self
    
    def transform(self, texts):
        """Transform texts using all extractors and combine features."""
        if not self.is_fitted:
            raise ValueError("Multi-extractor must be fitted before transform")
        
        feature_matrices = []
        for extractor in self.extractors.values():
            feature_matrices.append(extractor.transform(texts))
        
        # Combine features horizontally
        from scipy.sparse import hstack
        return hstack(feature_matrices)
    
    def fit_transform(self, texts):
        """Fit and transform in one step."""
        return self.fit(texts).transform(texts)

def create_feature_pipeline(method='tfidf', use_svd=False, **kwargs):
    """
    Create a complete feature extraction pipeline.
    
    Args:
        method (str): Vectorization method
        use_svd (bool): Whether to use SVD
        **kwargs: Additional arguments for the extractor
        
    Returns:
        Pipeline: Scikit-learn pipeline for feature extraction
    """
    steps = []
    
    # Base vectorizer
    if method == 'tfidf':
        vectorizer = TfidfVectorizer(**kwargs)
    elif method == 'count':
        vectorizer = CountVectorizer(**kwargs)
    else:
        raise ValueError(f"Unsupported method: {method}")
    
    steps.append(('vectorizer', vectorizer))
    
    # Add SVD if requested
    if use_svd:
        svd_components = kwargs.get('svd_components', 100)
        steps.append(('svd', TruncatedSVD(n_components=svd_components)))
    
    return Pipeline(steps)

# Utility functions
def compare_feature_methods(texts, labels=None, methods=['tfidf', 'count']):
    """
    Compare different feature extraction methods.
    
    Args:
        texts (list): List of text documents
        labels (list): Optional labels for evaluation
        methods (list): List of methods to compare
        
    Returns:
        dict: Comparison results
    """
    results = {}
    
    for method in methods:
        extractor = TextFeatureExtractor(method=method)
        feature_matrix = extractor.fit_transform(texts)
        
        results[method] = {
            'shape': feature_matrix.shape,
            'sparsity': 1.0 - (feature_matrix.nnz / (feature_matrix.shape[0] * feature_matrix.shape[1])),
            'memory_usage': feature_matrix.data.nbytes + feature_matrix.indices.nbytes + feature_matrix.indptr.nbytes
        }
        
        if hasattr(extractor, 'get_feature_names') and extractor.feature_names is not None:
            results[method]['top_features'] = extractor.get_top_features(feature_matrix, top_n=10)
    
    return results

if __name__ == "__main__":
    # Example usage
    sample_texts = [
        "i love this movie it amazing",
        "this film terrible waste time",
        "great acting excellent story",
        "boring plot poor direction",
        "fantastic cinematography outstanding performance"
    ]
    
    print("Feature Extraction Demo")
    print("=" * 50)
    
    # Compare different methods
    comparison = compare_feature_methods(sample_texts)
    
    for method, results in comparison.items():
        print(f"\n{method.upper()} Results:")
        print(f"Shape: {results['shape']}")
        print(f"Sparsity: {results['sparsity']:.3f}")
        print(f"Memory Usage: {results['memory_usage']} bytes")
        
        if 'top_features' in results:
            print("Top Features:")
            for feature, score in results['top_features'][:5]:
                print(f"  {feature}: {score:.4f}")
    
    # Demonstrate advanced features
    print(f"\nAdvanced Feature Extraction:")
    print("-" * 30)
    
    base_extractor = TextFeatureExtractor(method='tfidf', max_features=1000)
    advanced_extractor = AdvancedFeatureExtractor(
        base_extractor, use_svd=True, svd_components=50
    )
    
    features = advanced_extractor.fit_transform(sample_texts)
    print(f"Advanced features shape: {features.shape}")
    print(f"Dimensionality reduction: {base_extractor.max_features} -> {features.shape[1]}")
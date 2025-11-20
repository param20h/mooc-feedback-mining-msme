"""
Feature Extraction Module
Handles TF-IDF, word embeddings, and other feature engineering
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import joblib


class FeatureExtractor:
    """
    Extract features from preprocessed text data
    """
    
    def __init__(self, method='tfidf', max_features=5000, ngram_range=(1, 2)):
        """
        Initialize feature extractor
        
        Args:
            method: 'tfidf' or 'count'
            max_features: Maximum number of features to extract
            ngram_range: Range of n-grams (e.g., (1,2) for unigrams and bigrams)
        """
        self.method = method
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.vectorizer = None
        
    def fit_transform(self, texts):
        """
        Fit vectorizer and transform texts to feature matrix
        """
        if self.method == 'tfidf':
            self.vectorizer = TfidfVectorizer(
                max_features=self.max_features,
                ngram_range=self.ngram_range,
                min_df=2,
                max_df=0.8
            )
        elif self.method == 'count':
            self.vectorizer = CountVectorizer(
                max_features=self.max_features,
                ngram_range=self.ngram_range,
                min_df=2,
                max_df=0.8
            )
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        features = self.vectorizer.fit_transform(texts)
        print(f"✅ Extracted {features.shape[1]} features using {self.method.upper()}")
        return features
    
    def transform(self, texts):
        """
        Transform new texts using fitted vectorizer
        """
        if self.vectorizer is None:
            raise ValueError("Vectorizer not fitted. Call fit_transform first.")
        
        return self.vectorizer.transform(texts)
    
    def get_feature_names(self):
        """
        Get feature names from vectorizer
        """
        if self.vectorizer is None:
            return []
        return self.vectorizer.get_feature_names_out()
    
    def save_vectorizer(self, filepath):
        """
        Save fitted vectorizer to disk
        """
        if self.vectorizer is None:
            raise ValueError("No vectorizer to save")
        
        joblib.dump(self.vectorizer, filepath)
        print(f"✅ Vectorizer saved to {filepath}")
    
    def load_vectorizer(self, filepath):
        """
        Load vectorizer from disk
        """
        self.vectorizer = joblib.load(filepath)
        print(f"✅ Vectorizer loaded from {filepath}")
        return self.vectorizer


class TopicModeler:
    """
    Extract topics from text using LDA (Latent Dirichlet Allocation)
    """
    
    def __init__(self, n_topics=5, max_features=1000):
        self.n_topics = n_topics
        self.max_features = max_features
        self.vectorizer = CountVectorizer(max_features=max_features, max_df=0.8, min_df=2)
        self.lda_model = None
        
    def fit_transform(self, texts):
        """
        Fit LDA model and extract topics
        """
        # Create document-term matrix
        doc_term_matrix = self.vectorizer.fit_transform(texts)
        
        # Fit LDA model
        self.lda_model = LatentDirichletAllocation(
            n_components=self.n_topics,
            random_state=42,
            max_iter=20
        )
        
        doc_topics = self.lda_model.fit_transform(doc_term_matrix)
        print(f"✅ Extracted {self.n_topics} topics")
        
        return doc_topics
    
    def get_top_words_per_topic(self, n_words=10):
        """
        Get top words for each topic
        """
        if self.lda_model is None:
            raise ValueError("Model not fitted")
        
        feature_names = self.vectorizer.get_feature_names_out()
        topics = []
        
        for idx, topic in enumerate(self.lda_model.components_):
            top_words_idx = topic.argsort()[-n_words:][::-1]
            top_words = [feature_names[i] for i in top_words_idx]
            topics.append((f"Topic {idx+1}", top_words))
        
        return topics
    
    def print_topics(self, n_words=10):
        """
        Print top words for each topic
        """
        topics = self.get_top_words_per_topic(n_words)
        
        print("\n🔍 Discovered Topics:\n")
        for topic_name, words in topics:
            print(f"{topic_name}: {', '.join(words)}")


if __name__ == "__main__":
    print("Feature Extraction Module Loaded!")
    
    # Example usage
    sample_texts = [
        "great course learned lot",
        "instructor excellent material helpful",
        "poor quality boring lectures",
        "amazing content highly recommend"
    ]
    
    # TF-IDF extraction
    extractor = FeatureExtractor(method='tfidf', max_features=50)
    features = extractor.fit_transform(sample_texts)
    print(f"\nFeature matrix shape: {features.shape}")

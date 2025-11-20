"""
Model Training Module
Train sentiment analysis models using various algorithms
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import joblib
from datetime import datetime


class SentimentModel:
    """
    Train and manage sentiment analysis models
    """
    
    def __init__(self, model_type='logistic', random_state=42):
        """
        Initialize model
        
        Args:
            model_type: 'logistic', 'naive_bayes', 'svm', or 'random_forest'
            random_state: Random seed for reproducibility
        """
        self.model_type = model_type
        self.random_state = random_state
        self.model = self._initialize_model()
        
    def _initialize_model(self):
        """
        Initialize the appropriate model based on model_type
        """
        if self.model_type == 'logistic':
            return LogisticRegression(
                max_iter=1000,
                random_state=self.random_state,
                solver='lbfgs'
            )
        elif self.model_type == 'naive_bayes':
            return MultinomialNB()
        elif self.model_type == 'svm':
            return SVC(
                kernel='linear',
                random_state=self.random_state,
                probability=True
            )
        elif self.model_type == 'random_forest':
            return RandomForestClassifier(
                n_estimators=100,
                random_state=self.random_state
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def train(self, X_train, y_train):
        """
        Train the model
        """
        print(f"\n🚀 Training {self.model_type.upper()} model...")
        self.model.fit(X_train, y_train)
        print("✅ Training completed!")
        
    def predict(self, X):
        """
        Make predictions
        """
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """
        Get prediction probabilities
        """
        return self.model.predict_proba(X)
    
    def cross_validate(self, X, y, cv=5):
        """
        Perform cross-validation
        """
        print(f"\n🔄 Performing {cv}-fold cross-validation...")
        scores = cross_val_score(self.model, X, y, cv=cv, scoring='accuracy')
        
        print(f"Cross-validation scores: {scores}")
        print(f"Mean accuracy: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
        
        return scores
    
    def save_model(self, filepath):
        """
        Save trained model to disk
        """
        model_data = {
            'model': self.model,
            'model_type': self.model_type,
            'timestamp': datetime.now().isoformat()
        }
        
        joblib.dump(model_data, filepath)
        print(f"✅ Model saved to {filepath}")
    
    def load_model(self, filepath):
        """
        Load model from disk
        """
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.model_type = model_data['model_type']
        print(f"✅ Model loaded from {filepath}")
        print(f"   Model type: {self.model_type}")
        print(f"   Saved on: {model_data.get('timestamp', 'Unknown')}")


def prepare_train_test_split(X, y, test_size=0.2, random_state=42):
    """
    Split data into training and testing sets
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
    
    print(f"\n📊 Data Split:")
    print(f"   Training samples: {X_train.shape[0]}")
    print(f"   Testing samples: {X_test.shape[0]}")
    print(f"   Features: {X_train.shape[1]}")
    
    return X_train, X_test, y_train, y_test


def train_multiple_models(X_train, y_train, X_test, y_test):
    """
    Train and compare multiple models
    """
    models = {
        'Logistic Regression': 'logistic',
        'Naive Bayes': 'naive_bayes',
        'Random Forest': 'random_forest'
    }
    
    results = {}
    
    for name, model_type in models.items():
        print(f"\n{'='*50}")
        print(f"Training {name}...")
        print('='*50)
        
        # Train model
        model = SentimentModel(model_type=model_type)
        model.train(X_train, y_train)
        
        # Evaluate
        train_score = model.model.score(X_train, y_train)
        test_score = model.model.score(X_test, y_test)
        
        results[name] = {
            'model': model,
            'train_accuracy': train_score,
            'test_accuracy': test_score
        }
        
        print(f"✅ {name} Results:")
        print(f"   Training Accuracy: {train_score:.4f}")
        print(f"   Testing Accuracy: {test_score:.4f}")
    
    return results


if __name__ == "__main__":
    print("Model Training Module Loaded!")
    
    # Example usage
    print("\nExample: Training a Logistic Regression model")
    print("This module provides tools for training various ML models for sentiment analysis")

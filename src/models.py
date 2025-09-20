"""
Machine Learning Models Module for Twitter Sentiment Analysis

This module implements multiple machine learning algorithms for sentiment
classification using Scikit-Learn.
"""

import numpy as np
import pandas as pd
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, ClassifierMixin
import pickle
import os
from datetime import datetime

class SentimentClassifier(BaseEstimator, ClassifierMixin):
    """
    Base sentiment classifier with common functionality.
    """
    
    def __init__(self, model_type='naive_bayes', **kwargs):
        """
        Initialize the sentiment classifier.
        
        Args:
            model_type (str): Type of model to use
            **kwargs: Additional parameters for the specific model
        """
        self.model_type = model_type
        self.model_params = kwargs
        self.model = None
        self.classes_ = None
        self.is_fitted = False
        
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the specific model based on model_type."""
        if self.model_type == 'naive_bayes':
            self.model = MultinomialNB(**self.model_params)
        elif self.model_type == 'complement_nb':
            self.model = ComplementNB(**self.model_params)
        elif self.model_type == 'svm':
            self.model = SVC(probability=True, **self.model_params)
        elif self.model_type == 'linear_svm':
            self.model = LinearSVC(**self.model_params)
        elif self.model_type == 'logistic_regression':
            self.model = LogisticRegression(**self.model_params)
        elif self.model_type == 'random_forest':
            self.model = RandomForestClassifier(**self.model_params)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
    
    def fit(self, X, y):
        """
        Fit the model to training data.
        
        Args:
            X: Feature matrix
            y: Target labels
            
        Returns:
            self: Returns the instance for method chaining
        """
        self.model.fit(X, y)
        self.classes_ = self.model.classes_
        self.is_fitted = True
        return self
    
    def predict(self, X):
        """
        Predict sentiment labels.
        
        Args:
            X: Feature matrix
            
        Returns:
            numpy.ndarray: Predicted labels
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """
        Predict sentiment probabilities.
        
        Args:
            X: Feature matrix
            
        Returns:
            numpy.ndarray: Prediction probabilities
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        elif hasattr(self.model, 'decision_function'):
            # Convert decision function to probabilities for SVM
            decision = self.model.decision_function(X)
            if decision.ndim == 1:
                # Binary classification
                proba = np.exp(decision) / (1 + np.exp(decision))
                return np.column_stack([1 - proba, proba])
            else:
                # Multi-class classification
                exp_decision = np.exp(decision)
                return exp_decision / exp_decision.sum(axis=1, keepdims=True)
        else:
            raise ValueError("Model does not support probability prediction")
    
    def get_feature_importance(self, feature_names=None):
        """
        Get feature importance if available.
        
        Args:
            feature_names (list): List of feature names
            
        Returns:
            dict or None: Feature importance scores
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting feature importance")
        
        if hasattr(self.model, 'feature_importances_'):
            # Random Forest
            importances = self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            # Linear models
            if self.model.coef_.ndim == 1:
                importances = np.abs(self.model.coef_)
            else:
                importances = np.abs(self.model.coef_).mean(axis=0)
        else:
            return None
        
        if feature_names is not None:
            return dict(zip(feature_names, importances))
        else:
            return importances

class MultiModelClassifier:
    """
    Classifier that trains and compares multiple models.
    """
    
    def __init__(self, models=None):
        """
        Initialize multi-model classifier.
        
        Args:
            models (dict): Dictionary of model configurations
        """
        if models is None:
            self.models = self._get_default_models()
        else:
            self.models = models
        
        self.fitted_models = {}
        self.best_model = None
        self.best_score = -1
        self.is_fitted = False
    
    def _get_default_models(self):
        """Get default model configurations."""
        return {
            'naive_bayes': {
                'model_type': 'naive_bayes',
                'alpha': 1.0
            },
            'complement_nb': {
                'model_type': 'complement_nb',
                'alpha': 1.0
            },
            'logistic_regression': {
                'model_type': 'logistic_regression',
                'C': 1.0,
                'max_iter': 1000,
                'random_state': 42
            },
            'linear_svm': {
                'model_type': 'linear_svm',
                'C': 1.0,
                'random_state': 42
            },
            'random_forest': {
                'model_type': 'random_forest',
                'n_estimators': 100,
                'random_state': 42
            }
        }
    
    def fit(self, X, y, validation_split=0.2):
        """
        Fit all models and find the best one.
        
        Args:
            X: Feature matrix
            y: Target labels
            validation_split (float): Proportion of data for validation
            
        Returns:
            self: Returns the instance for method chaining
        """
        # Split data for validation
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, random_state=42, stratify=y
        )
        
        model_scores = {}
        
        # Train each model
        for model_name, model_config in self.models.items():
            print(f"Training {model_name}...")
            
            # Create and train model
            classifier = SentimentClassifier(**model_config)
            classifier.fit(X_train, y_train)
            
            # Evaluate on validation set
            val_score = classifier.model.score(X_val, y_val)
            model_scores[model_name] = val_score
            
            # Store fitted model
            self.fitted_models[model_name] = classifier
            
            print(f"{model_name} validation score: {val_score:.4f}")
        
        # Find best model
        self.best_model = max(model_scores, key=model_scores.get)
        self.best_score = model_scores[self.best_model]
        self.is_fitted = True
        
        print(f"\nBest model: {self.best_model} (score: {self.best_score:.4f})")
        return self
    
    def predict(self, X, use_best=True, model_name=None):
        """
        Make predictions using the best model or a specific model.
        
        Args:
            X: Feature matrix
            use_best (bool): Whether to use the best model
            model_name (str): Specific model to use (if use_best is False)
            
        Returns:
            numpy.ndarray: Predicted labels
        """
        if not self.is_fitted:
            raise ValueError("Models must be fitted before prediction")
        
        if use_best:
            return self.fitted_models[self.best_model].predict(X)
        elif model_name in self.fitted_models:
            return self.fitted_models[model_name].predict(X)
        else:
            raise ValueError(f"Model {model_name} not found")
    
    def predict_proba(self, X, use_best=True, model_name=None):
        """
        Predict probabilities using the best model or a specific model.
        
        Args:
            X: Feature matrix
            use_best (bool): Whether to use the best model
            model_name (str): Specific model to use (if use_best is False)
            
        Returns:
            numpy.ndarray: Prediction probabilities
        """
        if not self.is_fitted:
            raise ValueError("Models must be fitted before prediction")
        
        if use_best:
            return self.fitted_models[self.best_model].predict_proba(X)
        elif model_name in self.fitted_models:
            return self.fitted_models[model_name].predict_proba(X)
        else:
            raise ValueError(f"Model {model_name} not found")
    
    def get_model_comparison(self):
        """
        Get comparison of all trained models.
        
        Returns:
            pandas.DataFrame: Model comparison results
        """
        if not self.is_fitted:
            raise ValueError("Models must be fitted before comparison")
        
        results = []
        for model_name, model in self.fitted_models.items():
            results.append({
                'model': model_name,
                'model_type': model.model_type,
                'is_best': model_name == self.best_model,
                'validation_score': self.best_score if model_name == self.best_model else None
            })
        
        return pd.DataFrame(results)

class EnsembleClassifier:
    """
    Ensemble classifier combining multiple models using voting.
    """
    
    def __init__(self, base_models=None, voting='soft'):
        """
        Initialize ensemble classifier.
        
        Args:
            base_models (list): List of base model configurations
            voting (str): Voting strategy ('hard' or 'soft')
        """
        if base_models is None:
            self.base_models = self._get_default_base_models()
        else:
            self.base_models = base_models
        
        self.voting = voting
        self.ensemble = None
        self.is_fitted = False
    
    def _get_default_base_models(self):
        """Get default base models for ensemble."""
        return [
            ('nb', MultinomialNB(alpha=1.0)),
            ('lr', LogisticRegression(C=1.0, max_iter=1000, random_state=42)),
            ('svm', SVC(C=1.0, probability=True, random_state=42))
        ]
    
    def fit(self, X, y):
        """
        Fit the ensemble classifier.
        
        Args:
            X: Feature matrix
            y: Target labels
            
        Returns:
            self: Returns the instance for method chaining
        """
        self.ensemble = VotingClassifier(
            estimators=self.base_models,
            voting=self.voting
        )
        self.ensemble.fit(X, y)
        self.is_fitted = True
        return self
    
    def predict(self, X):
        """Make predictions using the ensemble."""
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before prediction")
        return self.ensemble.predict(X)
    
    def predict_proba(self, X):
        """Predict probabilities using the ensemble."""
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before prediction")
        return self.ensemble.predict_proba(X)

class HyperparameterTuner:
    """
    Hyperparameter tuning for sentiment classifiers.
    """
    
    def __init__(self, model_type='logistic_regression', cv=5):
        """
        Initialize hyperparameter tuner.
        
        Args:
            model_type (str): Type of model to tune
            cv (int): Number of cross-validation folds
        """
        self.model_type = model_type
        self.cv = cv
        self.best_params = None
        self.best_score = None
        self.best_model = None
        
        self.param_grids = self._get_param_grids()
    
    def _get_param_grids(self):
        """Get parameter grids for different models."""
        return {
            'naive_bayes': {
                'alpha': [0.1, 0.5, 1.0, 2.0, 5.0]
            },
            'logistic_regression': {
                'C': [0.1, 1.0, 10.0, 100.0],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear']
            },
            'linear_svm': {
                'C': [0.1, 1.0, 10.0, 100.0]
            },
            'random_forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5, 10]
            }
        }
    
    def tune(self, X, y, scoring='accuracy'):
        """
        Perform hyperparameter tuning.
        
        Args:
            X: Feature matrix
            y: Target labels
            scoring (str): Scoring metric
            
        Returns:
            dict: Best parameters found
        """
        # Get base model
        if self.model_type == 'naive_bayes':
            base_model = MultinomialNB()
        elif self.model_type == 'logistic_regression':
            base_model = LogisticRegression(max_iter=1000, random_state=42)
        elif self.model_type == 'linear_svm':
            base_model = LinearSVC(random_state=42)
        elif self.model_type == 'random_forest':
            base_model = RandomForestClassifier(random_state=42)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        
        # Perform grid search
        param_grid = self.param_grids[self.model_type]
        grid_search = GridSearchCV(
            base_model, param_grid, cv=self.cv, scoring=scoring, n_jobs=-1
        )
        
        grid_search.fit(X, y)
        
        self.best_params = grid_search.best_params_
        self.best_score = grid_search.best_score_
        self.best_model = grid_search.best_estimator_
        
        return self.best_params

def save_model(model, filepath):
    """
    Save a trained model to disk.
    
    Args:
        model: Trained model object
        filepath (str): Path to save the model
    """
    model_data = {
        'model': model,
        'timestamp': datetime.now().isoformat(),
        'model_type': getattr(model, 'model_type', 'unknown')
    }
    
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'wb') as f:
        pickle.dump(model_data, f)

def load_model(filepath):
    """
    Load a trained model from disk.
    
    Args:
        filepath (str): Path to load the model from
        
    Returns:
        Trained model object
    """
    with open(filepath, 'rb') as f:
        model_data = pickle.load(f)
    
    return model_data['model']

if __name__ == "__main__":
    # Example usage
    print("Sentiment Classification Models Demo")
    print("=" * 50)
    
    # Generate sample data
    from sklearn.datasets import make_classification
    
    X, y = make_classification(
        n_samples=1000, n_features=100, n_classes=3,
        n_informative=50, random_state=42
    )
    
    # Map to sentiment labels
    label_map = {0: 'negative', 1: 'neutral', 2: 'positive'}
    y_labels = [label_map[label] for label in y]
    
    print(f"Sample data shape: {X.shape}")
    print(f"Classes: {set(y_labels)}")
    
    # Train multiple models
    multi_classifier = MultiModelClassifier()
    multi_classifier.fit(X, y_labels)
    
    # Make predictions
    predictions = multi_classifier.predict(X[:10])
    probabilities = multi_classifier.predict_proba(X[:10])
    
    print(f"\nSample predictions: {predictions}")
    print(f"Prediction probabilities shape: {probabilities.shape}")
    
    # Model comparison
    comparison = multi_classifier.get_model_comparison()
    print(f"\nModel Comparison:")
    print(comparison)
    
    # Hyperparameter tuning example
    print(f"\nHyperparameter Tuning:")
    tuner = HyperparameterTuner('logistic_regression')
    best_params = tuner.tune(X, y_labels)
    print(f"Best parameters: {best_params}")
    print(f"Best score: {tuner.best_score:.4f}")
"""
Main Pipeline for Twitter Sentiment Analysis

This module provides the complete end-to-end pipeline for sentiment analysis,
from data loading and preprocessing to model training and evaluation.
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import joblib
from datetime import datetime
import json
import logging

# Import our custom modules
from preprocessing import TwitterPreprocessor, quick_preprocess
from feature_extraction import TextFeatureExtractor, AdvancedFeatureExtractor
from models import SentimentClassifier, MultiModelClassifier, save_model, load_model
from evaluation import SentimentModelEvaluator, ModelComparator

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('sentiment_analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class SentimentAnalysisPipeline:
    """
    Complete pipeline for Twitter sentiment analysis.
    """
    
    def __init__(self, config=None):
        """
        Initialize the sentiment analysis pipeline.
        
        Args:
            config (dict): Configuration parameters
        """
        self.config = config or self._get_default_config()
        
        # Initialize components
        self.preprocessor = None
        self.feature_extractor = None
        self.model = None
        self.evaluator = None
        
        # Data storage
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_matrix = None
        
        # Results storage
        self.results = {}
        
        # Initialize components based on config
        self._initialize_components()
    
    def _get_default_config(self):
        """Get default configuration parameters."""
        return {
            'preprocessing': {
                'remove_stopwords': True,
                'use_stemming': False,
                'use_lemmatization': True
            },
            'feature_extraction': {
                'method': 'tfidf',
                'max_features': 10000,
                'ngram_range': (1, 2),
                'min_df': 2,
                'max_df': 0.95
            },
            'model': {
                'type': 'multi_model',  # or specific model type
                'models': ['naive_bayes', 'logistic_regression', 'linear_svm']
            },
            'evaluation': {
                'test_size': 0.2,
                'random_state': 42,
                'cross_validation': True,
                'cv_folds': 5
            },
            'output': {
                'save_models': True,
                'save_results': True,
                'models_dir': 'models',
                'results_dir': 'results'
            }
        }
    
    def _initialize_components(self):
        """Initialize pipeline components based on configuration."""
        # Initialize preprocessor
        preproc_config = self.config['preprocessing']
        self.preprocessor = TwitterPreprocessor(**preproc_config)
        
        # Initialize feature extractor
        feat_config = self.config['feature_extraction']
        self.feature_extractor = TextFeatureExtractor(**feat_config)
        
        logger.info("Pipeline components initialized")
    
    def load_data(self, data_source, text_column='text', label_column='sentiment'):
        """
        Load data from various sources.
        
        Args:
            data_source (str or pd.DataFrame): Path to CSV file or DataFrame
            text_column (str): Name of text column
            label_column (str): Name of label column
            
        Returns:
            pd.DataFrame: Loaded data
        """
        if isinstance(data_source, str):
            if data_source.endswith('.csv'):
                self.data = pd.read_csv(data_source)
            elif data_source.endswith('.json'):
                self.data = pd.read_json(data_source)
            else:
                raise ValueError("Unsupported file format")
        elif isinstance(data_source, pd.DataFrame):
            self.data = data_source.copy()
        else:
            raise ValueError("Data source must be file path or DataFrame")
        
        # Validate required columns
        if text_column not in self.data.columns:
            raise ValueError(f"Text column '{text_column}' not found in data")
        if label_column not in self.data.columns:
            raise ValueError(f"Label column '{label_column}' not found in data")
        
        # Standardize column names
        self.data = self.data.rename(columns={
            text_column: 'text',
            label_column: 'sentiment'
        })
        
        # Remove rows with missing data
        initial_count = len(self.data)
        self.data = self.data.dropna(subset=['text', 'sentiment'])
        final_count = len(self.data)
        
        if initial_count != final_count:
            logger.info(f"Removed {initial_count - final_count} rows with missing data")
        
        logger.info(f"Loaded {len(self.data)} samples")
        logger.info(f"Sentiment distribution: {self.data['sentiment'].value_counts().to_dict()}")
        
        return self.data
    
    def preprocess_data(self):
        """
        Preprocess the loaded data.
        
        Returns:
            pd.DataFrame: Preprocessed data
        """
        if self.data is None:
            raise ValueError("Data must be loaded first")
        
        logger.info("Starting data preprocessing...")
        
        # Apply preprocessing
        self.data = self.preprocessor.preprocess_dataframe(
            self.data, text_column='text', target_column='sentiment'
        )
        
        # Remove empty texts after preprocessing
        initial_count = len(self.data)
        self.data = self.data[self.data['text_processed'].str.len() > 0]
        final_count = len(self.data)
        
        if initial_count != final_count:
            logger.info(f"Removed {initial_count - final_count} empty texts after preprocessing")
        
        logger.info(f"Preprocessing completed. {len(self.data)} samples remaining")
        return self.data
    
    def extract_features(self):
        """
        Extract features from preprocessed text.
        
        Returns:
            scipy.sparse matrix: Feature matrix
        """
        if self.data is None:
            raise ValueError("Data must be preprocessed first")
        
        logger.info("Starting feature extraction...")
        
        # Extract features
        texts = self.data['text_processed'].tolist()
        self.feature_matrix = self.feature_extractor.fit_transform(texts)
        
        logger.info(f"Feature extraction completed. Shape: {self.feature_matrix.shape}")
        
        # Get top features for analysis
        if hasattr(self.feature_extractor, 'get_top_features'):
            try:
                top_features = self.feature_extractor.get_top_features(self.feature_matrix, top_n=10)
                logger.info(f"Top features: {[f[0] for f in top_features[:5]]}")
            except:
                pass
        
        return self.feature_matrix
    
    def split_data(self):
        """
        Split data into training and testing sets.
        
        Returns:
            tuple: (X_train, X_test, y_train, y_test)
        """
        if self.feature_matrix is None:
            raise ValueError("Features must be extracted first")
        
        eval_config = self.config['evaluation']
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.feature_matrix,
            self.data['sentiment'].values,
            test_size=eval_config['test_size'],
            random_state=eval_config['random_state'],
            stratify=self.data['sentiment'].values
        )
        
        logger.info(f"Data split completed:")
        logger.info(f"Training set: {self.X_train.shape[0]} samples")
        logger.info(f"Test set: {self.X_test.shape[0]} samples")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def train_model(self):
        """
        Train the sentiment classification model.
        
        Returns:
            Trained model
        """
        if self.X_train is None:
            raise ValueError("Data must be split first")
        
        model_config = self.config['model']
        
        logger.info("Starting model training...")
        
        if model_config['type'] == 'multi_model':
            # Train multiple models and select the best
            self.model = MultiModelClassifier()
            self.model.fit(self.X_train, self.y_train)
            logger.info(f"Best model: {self.model.best_model}")
        else:
            # Train specific model type
            self.model = SentimentClassifier(model_type=model_config['type'])
            self.model.fit(self.X_train, self.y_train)
        
        logger.info("Model training completed")
        return self.model
    
    def evaluate_model(self):
        """
        Evaluate the trained model.
        
        Returns:
            dict: Evaluation results
        """
        if self.model is None:
            raise ValueError("Model must be trained first")
        
        logger.info("Starting model evaluation...")
        
        # Initialize evaluator
        self.evaluator = SentimentModelEvaluator(self.model)
        
        # Perform evaluation
        evaluation_results = self.evaluator.evaluate(
            self.X_test, self.y_test, self.X_train, self.y_train
        )
        
        # Store results
        self.results['evaluation'] = evaluation_results
        
        # Log key metrics
        logger.info(f"Test Accuracy: {evaluation_results['accuracy']:.4f}")
        logger.info(f"Test F1-Score (Macro): {evaluation_results['f1_macro']:.4f}")
        
        if 'train_metrics' in evaluation_results:
            train_acc = evaluation_results['train_metrics']['accuracy']
            test_acc = evaluation_results['accuracy']
            logger.info(f"Training Accuracy: {train_acc:.4f}")
            logger.info(f"Overfitting Check: {train_acc - test_acc:.4f}")
        
        return evaluation_results
    
    def predict(self, texts, return_probabilities=False):
        """
        Make predictions on new texts.
        
        Args:
            texts (list): List of text strings
            return_probabilities (bool): Whether to return probabilities
            
        Returns:
            numpy.ndarray: Predictions or probabilities
        """
        if self.model is None:
            raise ValueError("Model must be trained first")
        
        # Preprocess texts
        if isinstance(texts, str):
            texts = [texts]
        
        processed_texts = [self.preprocessor.preprocess_text(text) for text in texts]
        
        # Extract features
        features = self.feature_extractor.transform(processed_texts)
        
        # Make predictions
        if return_probabilities:
            return self.model.predict_proba(features)
        else:
            return self.model.predict(features)
    
    def save_pipeline(self, save_dir='models'):
        """
        Save the complete pipeline.
        
        Args:
            save_dir (str): Directory to save the pipeline
        """
        if self.model is None:
            raise ValueError("Pipeline must be trained first")
        
        os.makedirs(save_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save individual components
        model_path = os.path.join(save_dir, f'sentiment_model_{timestamp}.pkl')
        feature_extractor_path = os.path.join(save_dir, f'feature_extractor_{timestamp}.pkl')
        config_path = os.path.join(save_dir, f'config_{timestamp}.json')
        
        # Save model
        save_model(self.model, model_path)
        
        # Save feature extractor
        self.feature_extractor.save(feature_extractor_path)
        
        # Save configuration
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
        
        # Save results if available
        if self.results:
            results_path = os.path.join(save_dir, f'results_{timestamp}.json')
            # Convert numpy arrays to lists for JSON serialization
            serializable_results = self._make_json_serializable(self.results)
            with open(results_path, 'w') as f:
                json.dump(serializable_results, f, indent=2)
        
        logger.info(f"Pipeline saved to {save_dir}")
        
        return {
            'model_path': model_path,
            'feature_extractor_path': feature_extractor_path,
            'config_path': config_path
        }
    
    def _make_json_serializable(self, obj):
        """Convert numpy arrays and other non-serializable objects to JSON-compatible format."""
        if isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        else:
            return obj
    
    def run_complete_pipeline(self, data_source, text_column='text', label_column='sentiment'):
        """
        Run the complete sentiment analysis pipeline.
        
        Args:
            data_source: Data source (file path or DataFrame)
            text_column (str): Name of text column
            label_column (str): Name of label column
            
        Returns:
            dict: Complete pipeline results
        """
        logger.info("Starting complete sentiment analysis pipeline...")
        
        try:
            # Step 1: Load data
            self.load_data(data_source, text_column, label_column)
            
            # Step 2: Preprocess data
            self.preprocess_data()
            
            # Step 3: Extract features
            self.extract_features()
            
            # Step 4: Split data
            self.split_data()
            
            # Step 5: Train model
            self.train_model()
            
            # Step 6: Evaluate model
            self.evaluate_model()
            
            # Step 7: Save pipeline if configured
            if self.config['output']['save_models']:
                save_paths = self.save_pipeline(self.config['output']['models_dir'])
                self.results['save_paths'] = save_paths
            
            logger.info("Pipeline completed successfully!")
            return self.results
            
        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}")
            raise

class SentimentAnalyzer:
    """
    Simple interface for sentiment analysis predictions.
    """
    
    def __init__(self, model_path=None, feature_extractor_path=None, config_path=None):
        """
        Initialize sentiment analyzer with saved components.
        
        Args:
            model_path (str): Path to saved model
            feature_extractor_path (str): Path to saved feature extractor
            config_path (str): Path to saved configuration
        """
        self.model = None
        self.feature_extractor = None
        self.preprocessor = None
        self.config = None
        
        if model_path and feature_extractor_path:
            self.load_components(model_path, feature_extractor_path, config_path)
    
    def load_components(self, model_path, feature_extractor_path, config_path=None):
        """Load saved model components."""
        # Load model
        self.model = load_model(model_path)
        
        # Load feature extractor
        self.feature_extractor = TextFeatureExtractor(method='tfidf')
        self.feature_extractor.load(feature_extractor_path)
        
        # Load configuration
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self.config = json.load(f)
            
            # Initialize preprocessor with config
            preproc_config = self.config.get('preprocessing', {})
            self.preprocessor = TwitterPreprocessor(**preproc_config)
        else:
            # Use default preprocessor
            self.preprocessor = TwitterPreprocessor()
        
        logger.info("Components loaded successfully")
    
    def predict(self, text, return_probabilities=False):
        """
        Predict sentiment for a single text or list of texts.
        
        Args:
            text (str or list): Text(s) to analyze
            return_probabilities (bool): Whether to return probabilities
            
        Returns:
            str/list or numpy.ndarray: Predictions or probabilities
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_components() first.")
        
        # Handle single text
        single_text = isinstance(text, str)
        if single_text:
            texts = [text]
        else:
            texts = text
        
        # Preprocess
        processed_texts = [self.preprocessor.preprocess_text(t) for t in texts]
        
        # Extract features
        features = self.feature_extractor.transform(processed_texts)
        
        # Predict
        if return_probabilities:
            predictions = self.model.predict_proba(features)
        else:
            predictions = self.model.predict(features)
        
        # Return single prediction for single input
        if single_text:
            return predictions[0] if not return_probabilities else predictions[0]
        else:
            return predictions

if __name__ == "__main__":
    # Example usage
    print("Sentiment Analysis Pipeline Demo")
    print("=" * 40)
    
    # Create sample data
    sample_data = pd.DataFrame({
        'text': [
            "I love this movie! It's absolutely amazing! 😍",
            "This film is terrible. Waste of time and money.",
            "The movie is okay, nothing special but not bad either.",
            "Fantastic acting and great storyline! Highly recommended!",
            "Boring plot, poor direction. Very disappointed.",
            "It's an average movie. Some good parts, some bad.",
            "Absolutely loved it! Best movie I've seen this year!",
            "Not worth watching. Very predictable and dull.",
            "The movie has its moments but overall mediocre.",
            "Outstanding performance! Brilliant cinematography!"
        ],
        'sentiment': [
            'positive', 'negative', 'neutral', 'positive', 'negative',
            'neutral', 'positive', 'negative', 'neutral', 'positive'
        ]
    })
    
    print(f"Sample data created with {len(sample_data)} samples")
    print(f"Sentiment distribution: {sample_data['sentiment'].value_counts().to_dict()}")
    
    # Initialize and run pipeline
    pipeline = SentimentAnalysisPipeline()
    
    try:
        results = pipeline.run_complete_pipeline(sample_data)
        
        print("\nPipeline Results:")
        print(f"Test Accuracy: {results['evaluation']['accuracy']:.4f}")
        print(f"F1-Score (Macro): {results['evaluation']['f1_macro']:.4f}")
        
        # Test prediction on new text
        test_text = "This movie is absolutely fantastic! I loved every minute of it!"
        prediction = pipeline.predict(test_text)
        probabilities = pipeline.predict(test_text, return_probabilities=True)
        
        print(f"\nTest Prediction:")
        print(f"Text: {test_text}")
        print(f"Predicted Sentiment: {prediction[0]}")
        print(f"Probabilities: {probabilities[0]}")
        
    except Exception as e:
        print(f"Error running pipeline: {e}")
        logger.error(f"Pipeline error: {e}")
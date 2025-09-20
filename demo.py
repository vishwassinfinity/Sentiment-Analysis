#!/usr/bin/env python3
"""
Sentiment Analysis Demo Script

This script demonstrates the complete Twitter sentiment analysis system
with interactive examples and real-time predictions.
"""

import sys
import os
import pandas as pd
import numpy as np
import json
from datetime import datetime

# Add src directory to path to import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    from pipeline import SentimentAnalysisPipeline, SentimentAnalyzer
    from preprocessing import quick_preprocess
    from evaluation import SentimentModelEvaluator
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Make sure you're running this script from the project root directory")
    sys.exit(1)

class SentimentAnalysisDemo:
    """
    Interactive demo for the sentiment analysis system.
    """
    
    def __init__(self):
        """Initialize the demo."""
        self.pipeline = None
        self.analyzer = None
        self.sample_data = None
        self.test_cases = None
        
    def print_header(self, title):
        """Print a formatted header."""
        print("\n" + "=" * 60)
        print(f" {title}")
        print("=" * 60)
    
    def print_section(self, title):
        """Print a formatted section header."""
        print(f"\n{'-' * 40}")
        print(f" {title}")
        print(f"{'-' * 40}")
    
    def load_sample_data(self):
        """Load or create sample data for demonstration."""
        data_path = "data/sample_twitter_data.csv"
        
        if os.path.exists(data_path):
            print(f"Loading sample data from {data_path}")
            self.sample_data = pd.read_csv(data_path)
        else:
            print("Creating sample data...")
            # Create sample data if file doesn't exist
            self.sample_data = pd.DataFrame({
                'text': [
                    "I absolutely love this new movie! Best film ever! 🎬✨",
                    "Terrible service! Worst experience ever! Very disappointed 😡",
                    "The product is okay. Nothing special but works fine.",
                    "Amazing customer support! Very helpful and friendly! 👍",
                    "This restaurant is awful! Food is cold and tasteless 🤮",
                    "Average weather today. Neither sunny nor rainy.",
                    "Best vacation ever! Beautiful beaches and amazing sunsets! 🏖️",
                    "Computer keeps crashing! So frustrated with this! 💻😤",
                    "The book is decent. Some good parts, some boring ones.",
                    "Outstanding performance! Brilliant acting and directing! 🎭"
                ],
                'sentiment': [
                    'positive', 'negative', 'neutral', 'positive', 'negative',
                    'neutral', 'positive', 'negative', 'neutral', 'positive'
                ]
            })
        
        print(f"Loaded {len(self.sample_data)} samples")
        print(f"Sentiment distribution: {self.sample_data['sentiment'].value_counts().to_dict()}")
    
    def load_test_cases(self):
        """Load test cases for demonstration."""
        test_cases_path = "data/test_cases.json"
        
        if os.path.exists(test_cases_path):
            with open(test_cases_path, 'r') as f:
                self.test_cases = json.load(f)
        else:
            # Create sample test cases
            self.test_cases = [
                {
                    'text': "@customer_service Thanks for the amazing help! You guys rock! 🙌",
                    'expected': 'positive',
                    'category': 'customer_service'
                },
                {
                    'text': "Can't believe how terrible this delay is... #frustrated ✈️😡",
                    'expected': 'negative',
                    'category': 'travel'
                },
                {
                    'text': "Just finished the new episode. It was alright, nothing special.",
                    'expected': 'neutral',
                    'category': 'entertainment'
                }
            ]
        
        print(f"Loaded {len(self.test_cases)} test cases")
    
    def train_model_demo(self):
        """Demonstrate model training process."""
        self.print_section("Training Sentiment Analysis Model")
        
        print("Initializing pipeline...")
        self.pipeline = SentimentAnalysisPipeline()
        
        print("Running complete training pipeline...")
        try:
            results = self.pipeline.run_complete_pipeline(self.sample_data)
            
            print("\nTraining Results:")
            eval_results = results['evaluation']
            print(f"✓ Test Accuracy: {eval_results['accuracy']:.4f}")
            print(f"✓ Precision (Macro): {eval_results['precision_macro']:.4f}")
            print(f"✓ Recall (Macro): {eval_results['recall_macro']:.4f}")
            print(f"✓ F1-Score (Macro): {eval_results['f1_macro']:.4f}")
            
            # Show training vs test comparison if available
            if 'train_metrics' in eval_results:
                train_acc = eval_results['train_metrics']['accuracy']
                test_acc = eval_results['accuracy']
                print(f"✓ Training Accuracy: {train_acc:.4f}")
                print(f"✓ Overfitting Check: {abs(train_acc - test_acc):.4f}")
                
                if abs(train_acc - test_acc) < 0.1:
                    print("  → Good generalization!")
                else:
                    print("  → Potential overfitting detected")
            
            return True
            
        except Exception as e:
            print(f"❌ Training failed: {e}")
            return False
    
    def interactive_prediction_demo(self):
        """Interactive prediction demonstration."""
        self.print_section("Interactive Sentiment Prediction")
        
        if self.pipeline is None:
            print("❌ Model not trained. Please run training demo first.")
            return
        
        print("Enter text to analyze sentiment (or 'quit' to exit):")
        print("Example: 'I love this amazing product!'")
        
        while True:
            try:
                user_input = input("\n📝 Your text: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    break
                
                if not user_input:
                    print("Please enter some text to analyze.")
                    continue
                
                # Make prediction
                prediction = self.pipeline.predict(user_input)
                probabilities = self.pipeline.predict(user_input, return_probabilities=True)
                
                # Show results
                print(f"\n📊 Analysis Results:")
                print(f"   Text: {user_input}")
                print(f"   Predicted Sentiment: {prediction[0].upper()}")
                
                # Show probabilities
                if hasattr(self.pipeline.model, 'classes_'):
                    classes = self.pipeline.model.classes_
                elif hasattr(self.pipeline.model, 'fitted_models'):
                    # Multi-model classifier
                    best_model_name = self.pipeline.model.best_model
                    best_model = self.pipeline.model.fitted_models[best_model_name]
                    classes = best_model.classes_
                else:
                    classes = ['negative', 'neutral', 'positive']
                
                print(f"   Confidence Scores:")
                for class_name, prob in zip(classes, probabilities[0]):
                    emoji = "😞" if class_name == 'negative' else "😐" if class_name == 'neutral' else "😊"
                    print(f"     {emoji} {class_name.capitalize()}: {prob:.3f}")
                
                # Show preprocessing
                processed = quick_preprocess(user_input)
                print(f"   Processed Text: {processed}")
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"❌ Error: {e}")
        
        print("\nThank you for using the sentiment analyzer!")
    
    def test_cases_demo(self):
        """Demonstrate predictions on test cases."""
        self.print_section("Test Cases Evaluation")
        
        if self.pipeline is None:
            print("❌ Model not trained. Please run training demo first.")
            return
        
        correct_predictions = 0
        total_predictions = len(self.test_cases)
        
        print(f"Testing {total_predictions} real-world examples...")
        
        for i, test_case in enumerate(self.test_cases, 1):
            text = test_case['text']
            expected = test_case['expected']
            category = test_case.get('category', 'general')
            
            # Make prediction
            prediction = self.pipeline.predict(text)[0]
            probabilities = self.pipeline.predict(text, return_probabilities=True)[0]
            
            # Check if correct
            is_correct = prediction == expected
            correct_predictions += is_correct
            
            # Display result
            result_emoji = "✅" if is_correct else "❌"
            print(f"\n{result_emoji} Test Case {i} ({category}):")
            print(f"   Text: {text}")
            print(f"   Expected: {expected.upper()}")
            print(f"   Predicted: {prediction.upper()}")
            print(f"   Confidence: {max(probabilities):.3f}")
        
        # Show overall accuracy
        accuracy = correct_predictions / total_predictions
        print(f"\n🎯 Test Cases Accuracy: {correct_predictions}/{total_predictions} ({accuracy:.1%})")
        
        if accuracy >= 0.7:
            print("🎉 Great performance on test cases!")
        elif accuracy >= 0.5:
            print("👍 Decent performance on test cases.")
        else:
            print("⚠️ Performance could be improved.")
    
    def preprocessing_demo(self):
        """Demonstrate text preprocessing."""
        self.print_section("Text Preprocessing Demo")
        
        sample_tweets = [
            "OMG I LOVE this new movie!!! 😍😍😍 #amazing #movienight https://example.com @username",
            "This product is soooooo bad... I want my money back!!! 😡💸",
            "Can't believe how good this restaurant is! Best food ever! 🍕👌 #foodie"
        ]
        
        print("Demonstrating text preprocessing pipeline...")
        
        for i, tweet in enumerate(sample_tweets, 1):
            processed = quick_preprocess(tweet)
            
            print(f"\nExample {i}:")
            print(f"  Original:  {tweet}")
            print(f"  Processed: {processed}")
            print(f"  Changes:")
            print(f"    - Removed URLs, mentions, hashtags")
            print(f"    - Converted to lowercase")
            print(f"    - Removed punctuation and emojis")
            print(f"    - Reduced repeated characters")
            print(f"    - Removed stopwords")
    
    def model_comparison_demo(self):
        """Demonstrate model comparison."""
        self.print_section("Model Comparison Demo")
        
        if self.pipeline is None or not hasattr(self.pipeline.model, 'fitted_models'):
            print("❌ Multi-model pipeline not available.")
            return
        
        print("Comparing different machine learning models...")
        
        # Get model comparison results
        comparison = self.pipeline.model.get_model_comparison()
        print(f"\nTrained Models:")
        
        for _, row in comparison.iterrows():
            model_name = row['model']
            model_type = row['model_type']
            is_best = row['is_best']
            
            status = "🏆 BEST" if is_best else "   "
            print(f"  {status} {model_name} ({model_type})")
        
        print(f"\nBest Model: {self.pipeline.model.best_model}")
        print(f"Best Score: {self.pipeline.model.best_score:.4f}")
    
    def run_demo(self):
        """Run the complete demonstration."""
        self.print_header("Twitter Sentiment Analysis - Interactive Demo")
        
        print("Welcome to the Twitter Sentiment Analysis Demo!")
        print("This demo will show you how to:")
        print("1. Train sentiment analysis models")
        print("2. Make predictions on new text")
        print("3. Evaluate model performance")
        print("4. Compare different algorithms")
        
        # Load data
        print("\n🔄 Loading sample data...")
        self.load_sample_data()
        self.load_test_cases()
        
        # Run demos
        demos = [
            ("Text Preprocessing", self.preprocessing_demo),
            ("Model Training", self.train_model_demo),
            ("Model Comparison", self.model_comparison_demo),
            ("Test Cases Evaluation", self.test_cases_demo),
            ("Interactive Prediction", self.interactive_prediction_demo)
        ]
        
        for demo_name, demo_func in demos:
            try:
                demo_func()
            except KeyboardInterrupt:
                print(f"\n⚠️ {demo_name} demo interrupted by user.")
                break
            except Exception as e:
                print(f"❌ Error in {demo_name} demo: {e}")
                continue
        
        self.print_header("Demo Complete")
        print("Thank you for trying the Twitter Sentiment Analysis system!")
        print("\nTo use this system in your own projects:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Import the pipeline: from src.pipeline import SentimentAnalysisPipeline")
        print("3. Train on your data: pipeline.run_complete_pipeline(your_data)")
        print("4. Make predictions: pipeline.predict('your text here')")

def main():
    """Main function to run the demo."""
    try:
        demo = SentimentAnalysisDemo()
        demo.run_demo()
    except KeyboardInterrupt:
        print("\n\n👋 Demo interrupted. Goodbye!")
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        print("Please check your setup and try again.")

if __name__ == "__main__":
    main()
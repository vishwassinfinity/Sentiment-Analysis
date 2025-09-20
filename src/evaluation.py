"""
Model Evaluation Module for Twitter Sentiment Analysis

This module provides comprehensive evaluation metrics, visualizations, and
analysis tools for sentiment classification models.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, precision_recall_curve, average_precision_score
)
from sklearn.model_selection import learning_curve, validation_curve
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

class SentimentModelEvaluator:
    """
    Comprehensive evaluator for sentiment classification models.
    """
    
    def __init__(self, model, class_names=['negative', 'neutral', 'positive']):
        """
        Initialize the evaluator.
        
        Args:
            model: Trained sentiment classification model
            class_names (list): List of class names
        """
        self.model = model
        self.class_names = class_names
        self.evaluation_results = {}
    
    def evaluate(self, X_test, y_test, X_train=None, y_train=None):
        """
        Perform comprehensive evaluation of the model.
        
        Args:
            X_test: Test feature matrix
            y_test: Test target labels
            X_train: Training feature matrix (optional)
            y_train: Training target labels (optional)
            
        Returns:
            dict: Comprehensive evaluation results
        """
        # Make predictions
        y_pred = self.model.predict(X_test)
        
        # Get prediction probabilities if available
        try:
            y_pred_proba = self.model.predict_proba(X_test)
        except:
            y_pred_proba = None
        
        # Basic metrics
        results = self._calculate_basic_metrics(y_test, y_pred)
        
        # Classification report
        results['classification_report'] = classification_report(
            y_test, y_pred, target_names=self.class_names, output_dict=True
        )
        
        # Confusion matrix
        results['confusion_matrix'] = confusion_matrix(y_test, y_pred)
        
        # ROC and PR curves (for binary and multiclass)
        if y_pred_proba is not None:
            results.update(self._calculate_curve_metrics(y_test, y_pred_proba))
        
        # Training vs test performance (if training data provided)
        if X_train is not None and y_train is not None:
            y_train_pred = self.model.predict(X_train)
            results['train_metrics'] = self._calculate_basic_metrics(y_train, y_train_pred)
        
        self.evaluation_results = results
        return results
    
    def _calculate_basic_metrics(self, y_true, y_pred):
        """Calculate basic classification metrics."""
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision_macro': precision_score(y_true, y_pred, average='macro'),
            'precision_micro': precision_score(y_true, y_pred, average='micro'),
            'precision_weighted': precision_score(y_true, y_pred, average='weighted'),
            'recall_macro': recall_score(y_true, y_pred, average='macro'),
            'recall_micro': recall_score(y_true, y_pred, average='micro'),
            'recall_weighted': recall_score(y_true, y_pred, average='weighted'),
            'f1_macro': f1_score(y_true, y_pred, average='macro'),
            'f1_micro': f1_score(y_true, y_pred, average='micro'),
            'f1_weighted': f1_score(y_true, y_pred, average='weighted')
        }
    
    def _calculate_curve_metrics(self, y_true, y_pred_proba):
        """Calculate ROC and PR curve metrics."""
        results = {}
        
        # Convert labels to binary format for multiclass ROC
        from sklearn.preprocessing import label_binarize
        
        if len(self.class_names) == 2:
            # Binary classification
            if y_pred_proba.shape[1] == 2:
                y_scores = y_pred_proba[:, 1]
            else:
                y_scores = y_pred_proba
            
            # ROC curve
            fpr, tpr, _ = roc_curve(y_true, y_scores, pos_label=self.class_names[1])
            results['roc_auc'] = roc_auc_score(y_true, y_scores)
            results['roc_curve'] = {'fpr': fpr, 'tpr': tpr}
            
            # Precision-Recall curve
            precision, recall, _ = precision_recall_curve(y_true, y_scores, pos_label=self.class_names[1])
            results['pr_auc'] = average_precision_score(y_true, y_scores)
            results['pr_curve'] = {'precision': precision, 'recall': recall}
        
        else:
            # Multiclass classification
            y_true_bin = label_binarize(y_true, classes=self.class_names)
            
            # ROC for each class
            roc_auc_scores = {}
            for i, class_name in enumerate(self.class_names):
                fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_proba[:, i])
                roc_auc_scores[class_name] = roc_auc_score(y_true_bin[:, i], y_pred_proba[:, i])
            
            results['roc_auc_multiclass'] = roc_auc_scores
            results['roc_auc_macro'] = np.mean(list(roc_auc_scores.values()))
        
        return results
    
    def plot_confusion_matrix(self, normalize=False, figsize=(8, 6), save_path=None):
        """
        Plot confusion matrix.
        
        Args:
            normalize (bool): Whether to normalize the confusion matrix
            figsize (tuple): Figure size
            save_path (str): Path to save the plot
            
        Returns:
            matplotlib.figure.Figure: Confusion matrix plot
        """
        if 'confusion_matrix' not in self.evaluation_results:
            raise ValueError("Evaluation must be run first")
        
        cm = self.evaluation_results['confusion_matrix']
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            title = 'Normalized Confusion Matrix'
            fmt = '.2f'
        else:
            title = 'Confusion Matrix'
            fmt = 'd'
        
        plt.figure(figsize=figsize)
        sns.heatmap(
            cm, annot=True, fmt=fmt, cmap='Blues',
            xticklabels=self.class_names, yticklabels=self.class_names
        )
        plt.title(title)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return plt.gcf()
    
    def plot_classification_report(self, figsize=(10, 6), save_path=None):
        """
        Plot classification report as a heatmap.
        
        Args:
            figsize (tuple): Figure size
            save_path (str): Path to save the plot
            
        Returns:
            matplotlib.figure.Figure: Classification report plot
        """
        if 'classification_report' not in self.evaluation_results:
            raise ValueError("Evaluation must be run first")
        
        report = self.evaluation_results['classification_report']
        
        # Extract metrics for each class
        metrics_data = []
        for class_name in self.class_names:
            if class_name in report:
                metrics_data.append([
                    report[class_name]['precision'],
                    report[class_name]['recall'],
                    report[class_name]['f1-score']
                ])
        
        # Add overall metrics
        metrics_data.append([
            report['macro avg']['precision'],
            report['macro avg']['recall'],
            report['macro avg']['f1-score']
        ])
        
        metrics_df = pd.DataFrame(
            metrics_data,
            index=self.class_names + ['Macro Avg'],
            columns=['Precision', 'Recall', 'F1-Score']
        )
        
        plt.figure(figsize=figsize)
        sns.heatmap(
            metrics_df, annot=True, fmt='.3f', cmap='YlOrRd',
            cbar_kws={'label': 'Score'}
        )
        plt.title('Classification Report')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return plt.gcf()
    
    def plot_roc_curves(self, figsize=(10, 8), save_path=None):
        """
        Plot ROC curves for multiclass classification.
        
        Args:
            figsize (tuple): Figure size
            save_path (str): Path to save the plot
            
        Returns:
            matplotlib.figure.Figure: ROC curves plot
        """
        if 'roc_auc_multiclass' not in self.evaluation_results:
            if 'roc_curve' in self.evaluation_results:
                # Binary classification
                plt.figure(figsize=figsize)
                fpr = self.evaluation_results['roc_curve']['fpr']
                tpr = self.evaluation_results['roc_curve']['tpr']
                auc = self.evaluation_results['roc_auc']
                
                plt.plot(fpr, tpr, linewidth=2, label=f'ROC Curve (AUC = {auc:.3f})')
                plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title('Receiver Operating Characteristic (ROC) Curve')
                plt.legend(loc="lower right")
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                
                if save_path:
                    plt.savefig(save_path, dpi=300, bbox_inches='tight')
                
                return plt.gcf()
            else:
                raise ValueError("ROC curve data not available")
        
        # Multiclass ROC curves would require original prediction probabilities
        # This is a placeholder for the multiclass case
        print("Multiclass ROC curves require original prediction data")
        return None
    
    def plot_learning_curves(self, X, y, cv=5, figsize=(12, 5), save_path=None):
        """
        Plot learning curves to analyze model performance vs training size.
        
        Args:
            X: Feature matrix
            y: Target labels
            cv (int): Number of cross-validation folds
            figsize (tuple): Figure size
            save_path (str): Path to save the plot
            
        Returns:
            matplotlib.figure.Figure: Learning curves plot
        """
        train_sizes, train_scores, val_scores = learning_curve(
            self.model, X, y, cv=cv, n_jobs=-1,
            train_sizes=np.linspace(0.1, 1.0, 10),
            scoring='accuracy'
        )
        
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)
        
        plt.figure(figsize=figsize)
        
        plt.subplot(1, 2, 1)
        plt.plot(train_sizes, train_mean, 'o-', color='blue', label='Training Score')
        plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
        plt.plot(train_sizes, val_mean, 'o-', color='red', label='Validation Score')
        plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1, color='red')
        plt.xlabel('Training Set Size')
        plt.ylabel('Accuracy Score')
        plt.title('Learning Curves')
        plt.legend(loc='lower right')
        plt.grid(True, alpha=0.3)
        
        # Plot the gap between training and validation scores
        plt.subplot(1, 2, 2)
        gap = train_mean - val_mean
        plt.plot(train_sizes, gap, 'o-', color='green', label='Training-Validation Gap')
        plt.fill_between(train_sizes, gap - (train_std + val_std), gap + (train_std + val_std), alpha=0.1, color='green')
        plt.xlabel('Training Set Size')
        plt.ylabel('Score Difference')
        plt.title('Overfitting Analysis')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return plt.gcf()
    
    def generate_evaluation_report(self):
        """
        Generate a comprehensive text report of the evaluation results.
        
        Returns:
            str: Formatted evaluation report
        """
        if not self.evaluation_results:
            raise ValueError("Evaluation must be run first")
        
        report = []
        report.append("SENTIMENT ANALYSIS MODEL EVALUATION REPORT")
        report.append("=" * 50)
        
        # Basic metrics
        results = self.evaluation_results
        report.append(f"\nOVERALL PERFORMANCE:")
        report.append(f"Accuracy: {results['accuracy']:.4f}")
        report.append(f"Precision (Macro): {results['precision_macro']:.4f}")
        report.append(f"Recall (Macro): {results['recall_macro']:.4f}")
        report.append(f"F1-Score (Macro): {results['f1_macro']:.4f}")
        
        # Per-class performance
        report.append(f"\nPER-CLASS PERFORMANCE:")
        class_report = results['classification_report']
        for class_name in self.class_names:
            if class_name in class_report:
                metrics = class_report[class_name]
                report.append(f"\n{class_name.capitalize()}:")
                report.append(f"  Precision: {metrics['precision']:.4f}")
                report.append(f"  Recall: {metrics['recall']:.4f}")
                report.append(f"  F1-Score: {metrics['f1-score']:.4f}")
                report.append(f"  Support: {metrics['support']}")
        
        # ROC AUC if available
        if 'roc_auc' in results:
            report.append(f"\nROC AUC: {results['roc_auc']:.4f}")
        elif 'roc_auc_macro' in results:
            report.append(f"\nROC AUC (Macro): {results['roc_auc_macro']:.4f}")
        
        # Training vs test comparison if available
        if 'train_metrics' in results:
            report.append(f"\nTRAINING vs TEST COMPARISON:")
            train_acc = results['train_metrics']['accuracy']
            test_acc = results['accuracy']
            report.append(f"Training Accuracy: {train_acc:.4f}")
            report.append(f"Test Accuracy: {test_acc:.4f}")
            report.append(f"Difference: {train_acc - test_acc:.4f}")
            
            if train_acc - test_acc > 0.1:
                report.append("⚠️  Warning: Potential overfitting detected!")
            elif test_acc > train_acc:
                report.append("ℹ️  Note: Test performance exceeds training (good generalization)")
        
        return "\n".join(report)

class ModelComparator:
    """
    Compare multiple sentiment analysis models.
    """
    
    def __init__(self, models, model_names=None):
        """
        Initialize model comparator.
        
        Args:
            models (list): List of trained models
            model_names (list): List of model names
        """
        self.models = models
        self.model_names = model_names or [f"Model_{i+1}" for i in range(len(models))]
        self.comparison_results = {}
    
    def compare(self, X_test, y_test):
        """
        Compare all models on the test set.
        
        Args:
            X_test: Test feature matrix
            y_test: Test target labels
            
        Returns:
            pandas.DataFrame: Comparison results
        """
        results = []
        
        for model, name in zip(self.models, self.model_names):
            evaluator = SentimentModelEvaluator(model)
            eval_results = evaluator.evaluate(X_test, y_test)
            
            results.append({
                'Model': name,
                'Accuracy': eval_results['accuracy'],
                'Precision (Macro)': eval_results['precision_macro'],
                'Recall (Macro)': eval_results['recall_macro'],
                'F1-Score (Macro)': eval_results['f1_macro'],
                'F1-Score (Weighted)': eval_results['f1_weighted']
            })
        
        comparison_df = pd.DataFrame(results)
        self.comparison_results = comparison_df
        return comparison_df
    
    def plot_comparison(self, metric='Accuracy', figsize=(10, 6), save_path=None):
        """
        Plot model comparison.
        
        Args:
            metric (str): Metric to compare
            figsize (tuple): Figure size
            save_path (str): Path to save the plot
            
        Returns:
            matplotlib.figure.Figure: Comparison plot
        """
        if self.comparison_results.empty:
            raise ValueError("Comparison must be run first")
        
        plt.figure(figsize=figsize)
        
        x = range(len(self.model_names))
        values = self.comparison_results[metric].values
        colors = plt.cm.viridis(np.linspace(0, 1, len(self.model_names)))
        
        bars = plt.bar(x, values, color=colors, alpha=0.7, edgecolor='black', linewidth=1)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.xlabel('Models')
        plt.ylabel(metric)
        plt.title(f'Model Comparison - {metric}')
        plt.xticks(x, self.model_names, rotation=45)
        plt.ylim(0, 1.1)
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return plt.gcf()

if __name__ == "__main__":
    # Example usage
    print("Model Evaluation Demo")
    print("=" * 30)
    
    # Generate sample data and predictions
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.naive_bayes import MultinomialNB
    
    # Create sample data
    X, y = make_classification(
        n_samples=1000, n_features=20, n_classes=3,
        n_informative=15, random_state=42
    )
    
    # Map to sentiment labels
    label_map = {0: 'negative', 1: 'neutral', 2: 'positive'}
    y_labels = [label_map[label] for label in y]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_labels, test_size=0.3, random_state=42, stratify=y_labels
    )
    
    # Train a simple model
    model = MultinomialNB()
    model.fit(X_train, y_train)
    
    # Evaluate the model
    evaluator = SentimentModelEvaluator(model)
    results = evaluator.evaluate(X_test, y_test, X_train, y_train)
    
    # Print evaluation report
    print(evaluator.generate_evaluation_report())
    
    # Plot confusion matrix
    evaluator.plot_confusion_matrix()
    plt.show()
    
    # Plot classification report
    evaluator.plot_classification_report()
    plt.show()
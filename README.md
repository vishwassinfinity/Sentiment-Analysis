# Twitter Sentiment Analysis

A comprehensive Natural Language Processing (NLP) project to analyze and classify sentiments (positive, negative, neutral) from Twitter data using machine learning techniques.

## 🚀 Features

- **Text Preprocessing**: Comprehensive tweet cleaning and normalization
- **Feature Extraction**: TF-IDF and Count Vectorization
- **Multiple ML Models**: Naive Bayes, SVM, and Logistic Regression
- **Model Evaluation**: Accuracy metrics, confusion matrices, and classification reports
- **Interactive Notebook**: Step-by-step tutorial and visualizations
- **Demo Script**: Real-time sentiment analysis demonstration

## 🛠 Tools & Technologies

- **Python**: Core programming language
- **Pandas**: Data manipulation and analysis
- **Scikit-Learn**: Machine learning algorithms and evaluation
- **NLTK**: Natural language processing toolkit
- **Matplotlib/Seaborn**: Data visualization
- **Jupyter**: Interactive development environment

## 📁 Project Structure

```
Sentiment-Analysis/
├── data/                    # Dataset files
├── src/                     # Source code modules
│   ├── preprocessing.py     # Text preprocessing functions
│   ├── feature_extraction.py # Feature extraction methods
│   ├── models.py           # ML model implementations
│   ├── evaluation.py       # Model evaluation utilities
│   └── pipeline.py         # Main processing pipeline
├── notebooks/              # Jupyter notebooks
├── models/                 # Trained model files
├── requirements.txt        # Python dependencies
└── demo.py                # Demo script
```

## 🚀 Quick Start

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Sentiment-Analysis
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the demo**
   ```bash
   python demo.py
   ```

4. **Explore the notebook**
   ```bash
   jupyter notebook notebooks/sentiment_analysis_tutorial.ipynb
   ```

## 📊 Model Performance

The project implements and compares multiple machine learning algorithms:

- **Naive Bayes**: Fast and effective for text classification
- **Support Vector Machine (SVM)**: High accuracy with proper feature scaling
- **Logistic Regression**: Interpretable linear model with probability outputs

## 🔍 Usage Example

```python
from src.pipeline import SentimentAnalyzer

# Initialize the analyzer
analyzer = SentimentAnalyzer()

# Analyze sentiment
text = "I love this new feature!"
sentiment = analyzer.predict(text)
print(f"Sentiment: {sentiment}")  # Output: positive
```

## 📈 Results

The models achieve competitive performance on Twitter sentiment classification:
- Accuracy: ~85-90% on test data
- F1-Score: Balanced performance across all sentiment classes
- Real-time processing: Fast inference for live tweet analysis

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
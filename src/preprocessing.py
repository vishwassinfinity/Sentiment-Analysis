"""
Text Preprocessing Module for Twitter Sentiment Analysis

This module provides comprehensive text preprocessing functions specifically
designed for cleaning and normalizing Twitter data.
"""

import re
import string
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from textblob import TextBlob

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

class TwitterPreprocessor:
    """
    A comprehensive text preprocessor for Twitter data.
    """
    
    def __init__(self, remove_stopwords=True, use_stemming=False, use_lemmatization=True):
        """
        Initialize the preprocessor with configuration options.
        
        Args:
            remove_stopwords (bool): Whether to remove stopwords
            use_stemming (bool): Whether to apply stemming
            use_lemmatization (bool): Whether to apply lemmatization
        """
        self.remove_stopwords = remove_stopwords
        self.use_stemming = use_stemming
        self.use_lemmatization = use_lemmatization
        
        # Initialize NLTK tools
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        
        # Compile regex patterns for efficiency
        self.url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        self.mention_pattern = re.compile(r'@\w+')
        self.hashtag_pattern = re.compile(r'#\w+')
        self.email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
        self.phone_pattern = re.compile(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b')
        self.repeated_char_pattern = re.compile(r'(.)\1{2,}')
        
    def remove_urls(self, text):
        """Remove URLs from text."""
        return self.url_pattern.sub('', text)
    
    def remove_mentions(self, text):
        """Remove @mentions from text."""
        return self.mention_pattern.sub('', text)
    
    def remove_hashtags(self, text):
        """Remove hashtags from text (keeps the text after #)."""
        return self.hashtag_pattern.sub(lambda x: x.group(0)[1:], text)
    
    def remove_emails(self, text):
        """Remove email addresses from text."""
        return self.email_pattern.sub('', text)
    
    def remove_phone_numbers(self, text):
        """Remove phone numbers from text."""
        return self.phone_pattern.sub('', text)
    
    def handle_repeated_characters(self, text):
        """Reduce repeated characters (e.g., 'happyyy' -> 'happy')."""
        return self.repeated_char_pattern.sub(r'\1\1', text)
    
    def remove_punctuation(self, text):
        """Remove punctuation from text."""
        return text.translate(str.maketrans('', '', string.punctuation))
    
    def normalize_whitespace(self, text):
        """Normalize whitespace (remove extra spaces, tabs, newlines)."""
        return re.sub(r'\s+', ' ', text).strip()
    
    def expand_contractions(self, text):
        """Expand common contractions."""
        contractions = {
            "ain't": "am not", "aren't": "are not", "can't": "cannot",
            "couldn't": "could not", "didn't": "did not", "doesn't": "does not",
            "don't": "do not", "hadn't": "had not", "hasn't": "has not",
            "haven't": "have not", "he'd": "he would", "he'll": "he will",
            "he's": "he is", "i'd": "i would", "i'll": "i will",
            "i'm": "i am", "i've": "i have", "isn't": "is not",
            "it'd": "it would", "it'll": "it will", "it's": "it is",
            "let's": "let us", "shouldn't": "should not", "that's": "that is",
            "there's": "there is", "they'd": "they would", "they'll": "they will",
            "they're": "they are", "they've": "they have", "we'd": "we would",
            "we're": "we are", "we've": "we have", "weren't": "were not",
            "what's": "what is", "where's": "where is", "who's": "who is",
            "won't": "will not", "wouldn't": "would not", "you'd": "you would",
            "you'll": "you will", "you're": "you are", "you've": "you have"
        }
        
        for contraction, expansion in contractions.items():
            text = text.replace(contraction, expansion)
        return text
    
    def convert_to_lowercase(self, text):
        """Convert text to lowercase."""
        return text.lower()
    
    def remove_numbers(self, text):
        """Remove standalone numbers from text."""
        return re.sub(r'\b\d+\b', '', text)
    
    def tokenize(self, text):
        """Tokenize text into words."""
        return word_tokenize(text)
    
    def remove_stopwords_from_tokens(self, tokens):
        """Remove stopwords from tokenized text."""
        return [word for word in tokens if word.lower() not in self.stop_words]
    
    def stem_tokens(self, tokens):
        """Apply stemming to tokens."""
        return [self.stemmer.stem(word) for word in tokens]
    
    def lemmatize_tokens(self, tokens):
        """Apply lemmatization to tokens."""
        return [self.lemmatizer.lemmatize(word) for word in tokens]
    
    def correct_spelling(self, text):
        """Basic spelling correction using TextBlob."""
        try:
            blob = TextBlob(text)
            return str(blob.correct())
        except:
            return text
    
    def preprocess_text(self, text):
        """
        Apply complete preprocessing pipeline to text.
        
        Args:
            text (str): Raw text to preprocess
            
        Returns:
            str: Preprocessed text
        """
        if not isinstance(text, str):
            return ""
        
        # Step 1: Remove URLs, mentions, emails, phone numbers
        text = self.remove_urls(text)
        text = self.remove_mentions(text)
        text = self.remove_emails(text)
        text = self.remove_phone_numbers(text)
        
        # Step 2: Handle hashtags (keep text after #)
        text = self.remove_hashtags(text)
        
        # Step 3: Handle repeated characters
        text = self.handle_repeated_characters(text)
        
        # Step 4: Expand contractions
        text = self.expand_contractions(text)
        
        # Step 5: Convert to lowercase
        text = self.convert_to_lowercase(text)
        
        # Step 6: Remove punctuation
        text = self.remove_punctuation(text)
        
        # Step 7: Remove numbers
        text = self.remove_numbers(text)
        
        # Step 8: Normalize whitespace
        text = self.normalize_whitespace(text)
        
        # Step 9: Tokenize
        tokens = self.tokenize(text)
        
        # Step 10: Remove stopwords
        if self.remove_stopwords:
            tokens = self.remove_stopwords_from_tokens(tokens)
        
        # Step 11: Apply stemming or lemmatization
        if self.use_stemming:
            tokens = self.stem_tokens(tokens)
        elif self.use_lemmatization:
            tokens = self.lemmatize_tokens(tokens)
        
        # Step 12: Rejoin tokens
        return ' '.join(tokens)
    
    def preprocess_dataframe(self, df, text_column='text', target_column=None):
        """
        Preprocess text data in a pandas DataFrame.
        
        Args:
            df (pd.DataFrame): DataFrame containing text data
            text_column (str): Name of the column containing text
            target_column (str): Name of the target column (optional)
            
        Returns:
            pd.DataFrame: DataFrame with preprocessed text
        """
        processed_df = df.copy()
        
        # Preprocess text column
        processed_df[f'{text_column}_processed'] = processed_df[text_column].apply(
            self.preprocess_text
        )
        
        # Remove empty texts after preprocessing
        processed_df = processed_df[processed_df[f'{text_column}_processed'].str.len() > 0]
        
        return processed_df
    
    def get_text_statistics(self, text):
        """
        Get basic statistics about the text.
        
        Args:
            text (str): Text to analyze
            
        Returns:
            dict: Dictionary containing text statistics
        """
        if not isinstance(text, str):
            return {}
        
        tokens = self.tokenize(text.lower())
        
        return {
            'character_count': len(text),
            'word_count': len(tokens),
            'sentence_count': len(text.split('.')),
            'avg_word_length': sum(len(word) for word in tokens) / len(tokens) if tokens else 0,
            'unique_words': len(set(tokens)),
            'stopword_count': sum(1 for word in tokens if word in self.stop_words)
        }

# Convenience function for quick preprocessing
def quick_preprocess(text, remove_stopwords=True, use_lemmatization=True):
    """
    Quick preprocessing function for single text strings.
    
    Args:
        text (str): Text to preprocess
        remove_stopwords (bool): Whether to remove stopwords
        use_lemmatization (bool): Whether to apply lemmatization
        
    Returns:
        str: Preprocessed text
    """
    preprocessor = TwitterPreprocessor(
        remove_stopwords=remove_stopwords,
        use_lemmatization=use_lemmatization
    )
    return preprocessor.preprocess_text(text)

if __name__ == "__main__":
    # Example usage
    sample_tweets = [
        "I love this new movie!!! 😍😍😍 #amazing #movienight https://example.com",
        "@username Thanks for the recommendation! Can't wait to watch it 🎬",
        "This film is sooooo boring... 😴 I want my money back!",
        "What's your favorite movie? I'm looking for suggestions! 🤔"
    ]
    
    preprocessor = TwitterPreprocessor()
    
    print("Original vs Preprocessed Tweets:")
    print("=" * 50)
    
    for i, tweet in enumerate(sample_tweets, 1):
        processed = preprocessor.preprocess_text(tweet)
        stats = preprocessor.get_text_statistics(tweet)
        
        print(f"\nTweet {i}:")
        print(f"Original: {tweet}")
        print(f"Processed: {processed}")
        print(f"Stats: {stats}")
        print("-" * 30)
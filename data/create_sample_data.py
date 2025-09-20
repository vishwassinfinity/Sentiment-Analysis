"""
Sample Twitter Dataset for Sentiment Analysis

This file creates a comprehensive sample dataset that mimics real Twitter data
for sentiment analysis training and testing.
"""

import pandas as pd
import numpy as np
import json
import os

def create_sample_twitter_dataset(size=1000, save_path=None):
    """
    Create a sample Twitter-like dataset for sentiment analysis.
    
    Args:
        size (int): Number of samples to generate
        save_path (str): Path to save the dataset
        
    Returns:
        pd.DataFrame: Sample dataset
    """
    
    # Sample tweets for each sentiment category
    positive_tweets = [
        "I absolutely love this new movie! Best film of the year! 🎬✨",
        "Amazing product! Exceeded all my expectations. Highly recommend! 👍",
        "Such a beautiful day today! Perfect weather for a walk in the park 🌞",
        "Just got promoted at work! So excited and grateful! 🎉",
        "This restaurant has the most delicious food ever! Will definitely come back!",
        "Great customer service! The staff was so helpful and friendly 😊",
        "Successfully completed my marathon today! Feeling accomplished! 🏃‍♂️",
        "Love spending time with family and friends. Best weekend ever! ❤️",
        "This book is incredible! Can't put it down. Highly recommended! 📚",
        "Perfect vacation! Beautiful beaches and amazing sunsets 🏖️",
        "Got my dream job today! So happy and excited for this new chapter!",
        "This music is fantastic! Love the new album from my favorite artist 🎵",
        "Wonderful shopping experience! Found everything I was looking for",
        "Great workout today! Feeling energized and motivated 💪",
        "Delicious coffee and friendly barista! Love this café ☕",
        "Amazing concert last night! The performance was outstanding 🎤",
        "Beautiful weather for outdoor activities! Perfect day for hiking 🏔️",
        "Excellent movie! Great plot and outstanding acting 🎭",
        "Love my new phone! Great features and amazing camera 📱",
        "Fantastic dinner with friends! Great food and wonderful company 🍽️"
    ]
    
    negative_tweets = [
        "Terrible movie! Complete waste of time and money. Don't watch it! 😡",
        "Worst customer service ever! Rude staff and long waiting times 😤",
        "This product is broken and useless. Total disappointment!",
        "Horrible weather today! Rain ruined all my plans ☔",
        "Got stuck in traffic for 2 hours! So frustrating and annoying 🚗",
        "This restaurant has terrible food! Overpriced and tasteless",
        "Feeling sick and exhausted. Worst day ever! 😷",
        "Computer crashed and lost all my work! So angry right now 💻",
        "Cancelled flight ruined my vacation plans. Terrible airline service ✈️",
        "This book is boring and poorly written. Couldn't finish it 📖",
        "Awful shopping experience! Rude salespeople and poor quality products",
        "Horrible traffic jam! Going to be late for my important meeting",
        "This coffee tastes terrible! Cold and bitter ☕",
        "Disappointed with my purchase! Poor quality and doesn't work properly",
        "Terrible concert! Bad sound quality and overpriced tickets 🎵",
        "Worst vacation ever! Bad weather and terrible accommodation 🏨",
        "This app keeps crashing! So buggy and unreliable 📱",
        "Horrible experience at the gym! Broken equipment and crowded 🏋️",
        "This pizza is disgusting! Cold and tasteless 🍕",
        "Terrible service at this store! Unhelpful and unprofessional staff"
    ]
    
    neutral_tweets = [
        "The movie was okay. Not great but not terrible either",
        "Weather is average today. Neither sunny nor rainy ⛅",
        "This product works as expected. Nothing special about it",
        "Had lunch at a new restaurant. Food was decent, nothing more",
        "Regular day at work. Nothing exciting happened today",
        "The book is fine. Some interesting parts but overall average 📚",
        "Traffic was normal today. No major delays or issues",
        "This coffee shop is alright. Standard coffee and service ☕",
        "Average shopping experience. Found some items I needed",
        "The concert was okay. Not the best but not the worst either 🎵",
        "Standard gym session today. Normal workout routine 💪",
        "This app works fine. Does what it's supposed to do 📱",
        "Regular meeting at work. Discussed usual business topics",
        "Weather forecast shows mixed conditions for the week 🌦️",
        "This pizza place is decent. Standard quality and taste 🍕",
        "Average commute today. Same route as always 🚇",
        "This hotel is okay for the price. Basic amenities provided 🏨",
        "Regular day at school. Attended classes and studied 📖",
        "This store has standard selection. Found what I was looking for",
        "Normal evening at home. Watched some TV and relaxed 📺"
    ]
    
    # Create more varied tweets by combining base tweets with modifications
    def create_variations(base_tweets, sentiment, count):
        variations = []
        prefixes = ["", "Just ", "Today ", "Yesterday ", "Finally ", "Really "]
        suffixes = ["", "!", ".", "...", " #life", " #mood", " #today"]
        
        for i in range(count):
            base_tweet = np.random.choice(base_tweets)
            prefix = np.random.choice(prefixes)
            suffix = np.random.choice(suffixes)
            
            # Add some randomness
            if np.random.random() < 0.3:  # 30% chance to add prefix
                tweet = prefix + base_tweet.lower()
                tweet = tweet[0].upper() + tweet[1:]  # Capitalize first letter
            else:
                tweet = base_tweet
            
            if np.random.random() < 0.4:  # 40% chance to add suffix
                tweet += suffix
            
            variations.append({
                'text': tweet,
                'sentiment': sentiment,
                'length': len(tweet),
                'word_count': len(tweet.split())
            })
        
        return variations
    
    # Generate samples for each sentiment
    samples_per_sentiment = size // 3
    remaining_samples = size % 3
    
    # Create balanced dataset
    positive_samples = create_variations(positive_tweets, 'positive', samples_per_sentiment + (1 if remaining_samples > 0 else 0))
    negative_samples = create_variations(negative_tweets, 'negative', samples_per_sentiment + (1 if remaining_samples > 1 else 0))
    neutral_samples = create_variations(neutral_tweets, 'neutral', samples_per_sentiment)
    
    # Combine all samples
    all_samples = positive_samples + negative_samples + neutral_samples
    
    # Create DataFrame
    df = pd.DataFrame(all_samples)
    
    # Shuffle the dataset
    df = df.sample(frac=1).reset_index(drop=True)
    
    # Add additional metadata
    df['tweet_id'] = range(1, len(df) + 1)
    df['timestamp'] = pd.date_range(start='2023-01-01', periods=len(df), freq='H')
    df['user_id'] = np.random.randint(1, 1000, size=len(df))
    
    # Reorder columns
    df = df[['tweet_id', 'user_id', 'timestamp', 'text', 'sentiment', 'length', 'word_count']]
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df.to_csv(save_path, index=False)
        print(f"Dataset saved to {save_path}")
    
    return df

def create_real_world_test_cases():
    """
    Create realistic test cases that might appear in real Twitter data.
    
    Returns:
        list: List of test cases with expected sentiments
    """
    
    test_cases = [
        {
            'text': "@username Thanks for the amazing customer service! You guys rock! 🙌",
            'expected': 'positive',
            'category': 'customer_service'
        },
        {
            'text': "Can't believe how terrible this flight delay is... #frustrated ✈️😡",
            'expected': 'negative',
            'category': 'travel'
        },
        {
            'text': "Just finished watching the new episode. It was alright, nothing special.",
            'expected': 'neutral',
            'category': 'entertainment'
        },
        {
            'text': "OMG this concert is AMAZING!!! Best night ever!!! 🎵🎤🔥 #concert #music",
            'expected': 'positive',
            'category': 'entertainment'
        },
        {
            'text': "Stuck in traffic again... why does this always happen? 🚗😤 #traffic",
            'expected': 'negative',
            'category': 'daily_life'
        },
        {
            'text': "Weather today is cloudy with some sun. Pretty standard for this time of year.",
            'expected': 'neutral',
            'category': 'weather'
        },
        {
            'text': "Just got accepted to my dream university!!! So excited!!! 🎓✨ #college",
            'expected': 'positive',
            'category': 'achievement'
        },
        {
            'text': "This new phone is trash! Battery dies in 2 hours! #disappointed 📱",
            'expected': 'negative',
            'category': 'technology'
        },
        {
            'text': "Had lunch at that new place downtown. Food was decent, service was okay.",
            'expected': 'neutral',
            'category': 'food'
        },
        {
            'text': "Feeling grateful for my amazing friends and family! Love you all! ❤️ #blessed",
            'expected': 'positive',
            'category': 'personal'
        }
    ]
    
    return test_cases

def analyze_dataset_statistics(df):
    """
    Analyze and print statistics about the dataset.
    
    Args:
        df (pd.DataFrame): Dataset to analyze
    """
    print("Dataset Statistics:")
    print("=" * 30)
    print(f"Total samples: {len(df)}")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"Unique users: {df['user_id'].nunique()}")
    
    print("\nSentiment Distribution:")
    sentiment_counts = df['sentiment'].value_counts()
    for sentiment, count in sentiment_counts.items():
        percentage = (count / len(df)) * 100
        print(f"{sentiment.capitalize()}: {count} ({percentage:.1f}%)")
    
    print("\nText Length Statistics:")
    print(f"Average character length: {df['length'].mean():.1f}")
    print(f"Average word count: {df['word_count'].mean():.1f}")
    print(f"Min length: {df['length'].min()}")
    print(f"Max length: {df['length'].max()}")
    
    print("\nSample tweets by sentiment:")
    for sentiment in ['positive', 'negative', 'neutral']:
        sample = df[df['sentiment'] == sentiment]['text'].iloc[0]
        print(f"{sentiment.capitalize()}: {sample}")

if __name__ == "__main__":
    # Create sample dataset
    print("Creating sample Twitter dataset...")
    
    # Create dataset
    dataset = create_sample_twitter_dataset(size=1000)
    
    # Save to data directory
    data_dir = "../data"
    os.makedirs(data_dir, exist_ok=True)
    
    dataset_path = os.path.join(data_dir, "sample_twitter_data.csv")
    dataset.to_csv(dataset_path, index=False)
    
    # Analyze dataset
    analyze_dataset_statistics(dataset)
    
    # Create test cases
    test_cases = create_real_world_test_cases()
    
    # Save test cases
    test_cases_path = os.path.join(data_dir, "test_cases.json")
    with open(test_cases_path, 'w') as f:
        json.dump(test_cases, f, indent=2)
    
    print(f"\nDataset saved to: {dataset_path}")
    print(f"Test cases saved to: {test_cases_path}")
    print(f"\nCreated {len(test_cases)} real-world test cases")
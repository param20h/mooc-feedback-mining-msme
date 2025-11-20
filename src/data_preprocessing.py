"""
Data Preprocessing Module
Handles text cleaning, tokenization, and preparation for NLP tasks
"""

import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import contractions

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


class TextPreprocessor:
    """
    Text preprocessing pipeline for MOOC feedback data
    """
    
    def __init__(self, remove_stopwords=True, lemmatize=True):
        self.remove_stopwords = remove_stopwords
        self.lemmatize = lemmatize
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        
    def clean_text(self, text):
        """
        Clean raw text: lowercase, remove URLs, special chars, numbers
        """
        if pd.isna(text):
            return ""
        
        # Convert to string and lowercase
        text = str(text).lower()
        
        # Expand contractions (don't -> do not)
        text = contractions.fix(text)
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove mentions and hashtags
        text = re.sub(r'@\w+|#\w+', '', text)
        
        # Remove special characters but keep spaces
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def tokenize(self, text):
        """
        Tokenize text into words
        """
        return word_tokenize(text)
    
    def remove_stop_words(self, tokens):
        """
        Remove common stopwords
        """
        return [word for word in tokens if word not in self.stop_words]
    
    def lemmatize_tokens(self, tokens):
        """
        Lemmatize tokens to their base form
        """
        return [self.lemmatizer.lemmatize(word) for word in tokens]
    
    def preprocess(self, text):
        """
        Complete preprocessing pipeline
        """
        # Clean text
        cleaned = self.clean_text(text)
        
        # Tokenize
        tokens = self.tokenize(cleaned)
        
        # Remove stopwords
        if self.remove_stopwords:
            tokens = self.remove_stop_words(tokens)
        
        # Lemmatize
        if self.lemmatize:
            tokens = self.lemmatize_tokens(tokens)
        
        # Join back to string
        return ' '.join(tokens)
    
    def preprocess_dataframe(self, df, text_column='review', output_column='cleaned_review'):
        """
        Apply preprocessing to entire dataframe
        """
        df[output_column] = df[text_column].apply(self.preprocess)
        return df


def load_data(filepath, encoding='utf-8'):
    """
    Load dataset from CSV file
    """
    try:
        df = pd.read_csv(filepath, encoding=encoding)
        print(f"✅ Loaded {len(df)} reviews from {filepath}")
        return df
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None


def save_processed_data(df, output_path):
    """
    Save preprocessed data to CSV
    """
    try:
        df.to_csv(output_path, index=False, encoding='utf-8')
        print(f"✅ Saved processed data to {output_path}")
    except Exception as e:
        print(f"❌ Error saving data: {e}")


if __name__ == "__main__":
    # Example usage
    print("Text Preprocessing Module Loaded!")
    
    # Test preprocessing
    preprocessor = TextPreprocessor()
    sample_text = "This course is AMAZING!!! I learned so much. Check it out at www.example.com"
    cleaned = preprocessor.preprocess(sample_text)
    print(f"\nOriginal: {sample_text}")
    print(f"Cleaned: {cleaned}")

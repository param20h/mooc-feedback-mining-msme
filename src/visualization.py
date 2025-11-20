"""
Visualization Module
Create insightful visualizations for MOOC feedback analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)


def plot_sentiment_distribution(df, sentiment_column='sentiment', save_path=None):
    """
    Plot distribution of sentiments in the dataset
    """
    sentiment_counts = df[sentiment_column].value_counts()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Bar plot
    sentiment_counts.plot(kind='bar', ax=ax1, color=['#e74c3c', '#95a5a6', '#2ecc71'])
    ax1.set_title('Sentiment Distribution (Bar Chart)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Sentiment', fontsize=12)
    ax1.set_ylabel('Count', fontsize=12)
    ax1.tick_params(axis='x', rotation=0)
    
    # Pie chart
    colors = ['#e74c3c', '#95a5a6', '#2ecc71']
    ax2.pie(sentiment_counts.values, labels=sentiment_counts.index, autopct='%1.1f%%',
            startangle=90, colors=colors)
    ax2.set_title('Sentiment Distribution (Pie Chart)', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Sentiment distribution saved to {save_path}")
    
    plt.show()


def generate_wordcloud(text_data, title='Word Cloud', save_path=None, figsize=(12, 6)):
    """
    Generate word cloud from text data
    """
    # Combine all text
    if isinstance(text_data, pd.Series):
        text = ' '.join(text_data.astype(str))
    else:
        text = ' '.join(text_data)
    
    # Generate word cloud
    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color='white',
        colormap='viridis',
        max_words=100,
        relative_scaling=0.5
    ).generate(text)
    
    # Plot
    plt.figure(figsize=figsize)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Word cloud saved to {save_path}")
    
    plt.show()


def plot_top_words(text_data, n=20, title='Top Words', save_path=None):
    """
    Plot most frequent words
    """
    # Combine and tokenize
    if isinstance(text_data, pd.Series):
        all_words = ' '.join(text_data.astype(str)).split()
    else:
        all_words = ' '.join(text_data).split()
    
    # Count words
    word_freq = Counter(all_words)
    top_words = word_freq.most_common(n)
    
    # Create dataframe
    words_df = pd.DataFrame(top_words, columns=['Word', 'Frequency'])
    
    # Plot
    plt.figure(figsize=(12, 6))
    sns.barplot(data=words_df, x='Frequency', y='Word', palette='viridis')
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Frequency', fontsize=12)
    plt.ylabel('Word', fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Top words plot saved to {save_path}")
    
    plt.show()
    
    return words_df


def plot_review_length_distribution(df, text_column='review', save_path=None):
    """
    Plot distribution of review lengths
    """
    df['review_length'] = df[text_column].astype(str).apply(lambda x: len(x.split()))
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram
    ax1.hist(df['review_length'], bins=50, color='skyblue', edgecolor='black')
    ax1.set_title('Review Length Distribution', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Number of Words', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.axvline(df['review_length'].mean(), color='red', linestyle='--', 
                label=f'Mean: {df["review_length"].mean():.1f}')
    ax1.legend()
    
    # Box plot
    ax2.boxplot(df['review_length'], vert=True)
    ax2.set_title('Review Length Box Plot', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Number of Words', fontsize=12)
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Review length distribution saved to {save_path}")
    
    plt.show()
    
    print(f"\n📊 Review Length Statistics:")
    print(f"   Mean: {df['review_length'].mean():.2f} words")
    print(f"   Median: {df['review_length'].median():.2f} words")
    print(f"   Min: {df['review_length'].min()} words")
    print(f"   Max: {df['review_length'].max()} words")


def plot_sentiment_by_category(df, category_column, sentiment_column='sentiment', save_path=None):
    """
    Plot sentiment distribution across different categories (e.g., courses)
    """
    plt.figure(figsize=(12, 6))
    
    # Create crosstab
    ct = pd.crosstab(df[category_column], df[sentiment_column], normalize='index') * 100
    
    ct.plot(kind='bar', stacked=False, ax=plt.gca(), 
            color=['#e74c3c', '#95a5a6', '#2ecc71'])
    
    plt.title(f'Sentiment Distribution by {category_column.title()}', 
              fontsize=14, fontweight='bold')
    plt.xlabel(category_column.title(), fontsize=12)
    plt.ylabel('Percentage (%)', fontsize=12)
    plt.legend(title='Sentiment', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Sentiment by category plot saved to {save_path}")
    
    plt.show()


def create_dashboard(df, text_column='review', sentiment_column='sentiment', save_path=None):
    """
    Create a comprehensive visualization dashboard
    """
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # 1. Sentiment Distribution
    ax1 = fig.add_subplot(gs[0, 0])
    sentiment_counts = df[sentiment_column].value_counts()
    sentiment_counts.plot(kind='bar', ax=ax1, color=['#e74c3c', '#95a5a6', '#2ecc71'])
    ax1.set_title('Sentiment Distribution', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Sentiment')
    ax1.tick_params(axis='x', rotation=0)
    
    # 2. Review Length Distribution
    ax2 = fig.add_subplot(gs[0, 1])
    review_lengths = df[text_column].astype(str).apply(lambda x: len(x.split()))
    ax2.hist(review_lengths, bins=30, color='skyblue', edgecolor='black')
    ax2.set_title('Review Length Distribution', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Number of Words')
    ax2.axvline(review_lengths.mean(), color='red', linestyle='--', alpha=0.7)
    
    # 3. Sentiment Pie Chart
    ax3 = fig.add_subplot(gs[1, 0])
    colors = ['#e74c3c', '#95a5a6', '#2ecc71']
    ax3.pie(sentiment_counts.values, labels=sentiment_counts.index, 
            autopct='%1.1f%%', startangle=90, colors=colors)
    ax3.set_title('Sentiment Percentage', fontsize=12, fontweight='bold')
    
    # 4. Top Words
    ax4 = fig.add_subplot(gs[1, 1])
    all_words = ' '.join(df[text_column].astype(str)).split()
    word_freq = Counter(all_words)
    top_10 = pd.DataFrame(word_freq.most_common(10), columns=['Word', 'Frequency'])
    sns.barplot(data=top_10, x='Frequency', y='Word', ax=ax4, palette='viridis')
    ax4.set_title('Top 10 Words', fontsize=12, fontweight='bold')
    
    # 5. Word Cloud (spanning bottom row)
    ax5 = fig.add_subplot(gs[2, :])
    text = ' '.join(df[text_column].astype(str))
    wordcloud = WordCloud(width=800, height=300, background_color='white', 
                          colormap='viridis', max_words=80).generate(text)
    ax5.imshow(wordcloud, interpolation='bilinear')
    ax5.axis('off')
    ax5.set_title('Word Cloud - All Reviews', fontsize=12, fontweight='bold')
    
    fig.suptitle('📊 MOOC Feedback Analysis Dashboard', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Dashboard saved to {save_path}")
    
    plt.show()


if __name__ == "__main__":
    print("Visualization Module Loaded!")
    print("\nAvailable visualization functions:")
    print("  - plot_sentiment_distribution()")
    print("  - generate_wordcloud()")
    print("  - plot_top_words()")
    print("  - plot_review_length_distribution()")
    print("  - plot_sentiment_by_category()")
    print("  - create_dashboard()")

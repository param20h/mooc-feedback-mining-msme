# 📚 MOOC Feedback Dataset Sources

This document provides information on where to obtain datasets for MOOC feedback analysis.

---

## 🎯 Recommended Data Sources

### 1. **Kaggle Datasets**

#### 📌 Coursera Course Reviews
- **URL**: https://www.kaggle.com/datasets/septa97/100k-courseras-course-reviews-dataset
- **Description**: 100K+ Coursera course reviews with ratings
- **Format**: CSV
- **Columns**: Course name, review text, rating, date
- **Size**: ~100,000 reviews
- **License**: Open source

#### 📌 Udemy Courses Dataset
- **URL**: https://www.kaggle.com/datasets/andrewmvd/udemy-courses
- **Description**: Udemy courses with reviews and ratings
- **Format**: CSV
- **Contains**: Course information, subscriber count, reviews

#### 📌 edX Course Reviews
- **URL**: https://www.kaggle.com/search?q=edx+reviews
- **Description**: Various edX platform reviews
- **Format**: CSV/JSON

### 2. **Academic Datasets**

#### 📌 MOOC Dropout Prediction Dataset
- **Source**: UCI Machine Learning Repository
- **URL**: https://archive.ics.uci.edu/ml/datasets/MOOC+data
- **Description**: Student interaction data with feedback

#### 📌 Stanford MOOC Posts Dataset
- **Source**: Stanford Network Analysis Project (SNAP)
- **URL**: https://snap.stanford.edu/data/
- **Description**: Forum posts and discussions from MOOCs

### 3. **API-Based Collection**

#### 📌 Coursera API (Unofficial)
- **Method**: Web scraping with BeautifulSoup/Selenium
- **Target**: Course review pages
- **Note**: Respect robots.txt and rate limiting

#### 📌 Udemy API
- **URL**: https://www.udemy.com/developers/
- **Access**: Requires API key (for affiliates)
- **Data**: Course reviews, ratings, enrollment data

#### 📌 edX API
- **Documentation**: Limited public API
- **Alternative**: Web scraping course catalogs

---

## 🔧 Sample Datasets for Testing

### Quick Start: Synthetic Data
If you need to test your pipeline immediately, you can create synthetic MOOC feedback data:

```python
import pandas as pd
import numpy as np

# Sample data generator
reviews = [
    "This course was excellent! Learned so much about data science.",
    "Terrible instructor, couldn't understand anything.",
    "Good content but too fast paced for beginners.",
    "Amazing course! Highly recommend for MSMEs.",
    "Waste of time and money, very disappointed."
]

sentiments = ['positive', 'negative', 'neutral', 'positive', 'negative']

# Create sample dataset
df = pd.DataFrame({
    'course_name': ['Data Science 101'] * len(reviews),
    'review': reviews,
    'sentiment': sentiments,
    'rating': [5, 1, 3, 5, 1]
})

# Save to CSV
df.to_csv('data/raw/sample_reviews.csv', index=False)
```

---

## 📥 How to Download and Prepare Data

### Option 1: From Kaggle

1. **Install Kaggle API**:
   ```bash
   pip install kaggle
   ```

2. **Set up API credentials**:
   - Go to https://www.kaggle.com/settings
   - Click "Create New API Token"
   - Place `kaggle.json` in `~/.kaggle/` (Linux/Mac) or `C:\Users\<username>\.kaggle\` (Windows)

3. **Download dataset**:
   ```bash
   kaggle datasets download -d septa97/100k-courseras-course-reviews-dataset
   unzip 100k-courseras-course-reviews-dataset.zip -d data/raw/
   ```

### Option 2: Manual Download

1. Visit the dataset URL
2. Click "Download" button
3. Extract files to `data/raw/` folder
4. Rename file to `coursera_reviews.csv` (or adjust code accordingly)

### Option 3: Web Scraping

Create a scraper script (example provided below):

```python
# src/scrape_reviews.py
import requests
from bs4 import BeautifulSoup
import pandas as pd
import time

def scrape_coursera_reviews(course_url, num_pages=5):
    """
    Scrape Coursera course reviews
    Note: Always check robots.txt and terms of service
    """
    reviews = []
    
    for page in range(1, num_pages + 1):
        # Add respectful delay
        time.sleep(2)
        
        # Make request (adjust based on actual website structure)
        # ... scraping logic here ...
        
    return pd.DataFrame(reviews)
```

---

## 📊 Expected Data Format

Your dataset should have these columns (minimum):

| Column | Type | Description |
|--------|------|-------------|
| `review` or `text` | string | The actual feedback text |
| `sentiment` or `label` | string/int | Sentiment label (positive/negative/neutral or 0/1/2) |
| `rating` (optional) | float | Numeric rating (e.g., 1-5 stars) |
| `course_name` (optional) | string | Course identifier |
| `date` (optional) | datetime | Review date |

### Example:
```csv
review,sentiment,rating,course_name
"Great course! Learned a lot",positive,5,Machine Learning
"Too difficult for beginners",negative,2,Advanced Python
"Decent content, okay instructor",neutral,3,Data Science 101
```

---

## 🏷️ Creating Sentiment Labels

If your dataset only has **ratings** but no sentiment labels:

```python
def rating_to_sentiment(rating):
    """Convert numeric rating to sentiment label"""
    if rating >= 4:
        return 'positive'
    elif rating <= 2:
        return 'negative'
    else:
        return 'neutral'

df['sentiment'] = df['rating'].apply(rating_to_sentiment)
```

---

## ✅ Data Preparation Checklist

- [ ] Downloaded dataset from source
- [ ] Placed in `data/raw/` folder
- [ ] Verified file format (CSV/JSON)
- [ ] Checked for required columns (`review`, `sentiment`)
- [ ] Handled missing values
- [ ] Created sentiment labels (if needed from ratings)
- [ ] Split into train/test sets (done automatically in notebooks)

---

## 📖 Additional Resources

### Research Papers with Datasets:
1. **"A Large-Scale Study of MOOC Learner Behaviors"** - Contains forum posts
2. **"Sentiment Analysis of MOOC Reviews"** - Labeled sentiment datasets

### Open Data Repositories:
- **Google Dataset Search**: https://datasetsearch.research.google.com/
- **Papers with Code**: https://paperswithcode.com/datasets
- **Zenodo**: https://zenodo.org/ (Academic datasets)

---

## ⚖️ Legal & Ethical Considerations

⚠️ **Important Notes**:
- Always respect website terms of service
- Check data licensing before use
- Anonymize personal information
- Cite data sources in your research
- Follow rate limits when scraping
- Consider privacy implications

---

## 🆘 Troubleshooting

### Can't find a dataset?
1. Start with synthetic data for testing
2. Use the Kaggle Coursera dataset (most reliable)
3. Combine multiple smaller datasets

### Dataset format different?
- Adjust column names in notebooks (search for `'review'` and `'sentiment'`)
- Use the preprocessing module to clean data

### No sentiment labels?
- Create labels from ratings (code provided above)
- Use pre-trained sentiment models to auto-label
- Manually label a small sample for training

---

## 📞 Support

If you need help with data acquisition:
1. Check existing issues in the repository
2. Create a new issue with the tag `data-help`
3. Reach out to the maintainer

---

**Happy Data Mining! 🚀**

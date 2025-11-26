# 🚀 MOOC Feedback Sentiment Analysis API

REST API for analyzing sentiment in MOOC course reviews using machine learning.

## 📋 Features

- **Single Review Analysis**: Predict sentiment for individual reviews
- **Batch Processing**: Analyze multiple reviews at once
- **CSV Upload**: Upload CSV files for bulk analysis
- **Real-time Predictions**: Fast inference with pre-trained models
- **Interactive Documentation**: Auto-generated API docs with Swagger UI

## 🛠️ Installation

### 1. Install Dependencies

```bash
pip install -r requirements_api.txt
```

### 2. Ensure Models are Trained

Make sure you have these files in the `models/` directory:
- `optimized_best_model.pkl` (trained model)
- `tfidf_vectorizer.pkl` (TF-IDF vectorizer)

Run notebook `03_model_optimization.ipynb` if needed.

## 🚀 Running the API

### Start the Server

```bash
python api.py
```

Or using uvicorn directly:

```bash
uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at: **http://localhost:8000**

### Access Documentation

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## 📡 API Endpoints

### 1. Health Check

**GET** `/health`

Check if API is running and models are loaded.

```bash
curl http://localhost:8000/health
```

**Response:**
```json
{
  "status": "healthy",
  "models_loaded": true,
  "version": "1.0.0"
}
```

---

### 2. Single Review Prediction

**POST** `/predict`

Analyze sentiment for a single review.

**Request:**
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"text": "This course is amazing! I learned so much."}'
```

**Response:**
```json
{
  "text": "This course is amazing! I learned so much.",
  "sentiment": "Positive",
  "confidence": 0.95,
  "probabilities": {
    "Negative": 0.02,
    "Neutral": 0.03,
    "Positive": 0.95
  }
}
```

---

### 3. Batch Prediction

**POST** `/predict/batch`

Analyze multiple reviews at once.

**Request:**
```bash
curl -X POST "http://localhost:8000/predict/batch" \
  -H "Content-Type: application/json" \
  -d '{
    "reviews": [
      "Excellent course!",
      "Waste of time",
      "It was okay"
    ]
  }'
```

**Response:**
```json
{
  "total_reviews": 3,
  "results": [
    {
      "text": "Excellent course!",
      "sentiment": "Positive",
      "confidence": 0.92,
      "probabilities": {"Negative": 0.03, "Neutral": 0.05, "Positive": 0.92}
    }
  ],
  "summary": {
    "positive_count": 1,
    "neutral_count": 1,
    "negative_count": 1,
    "positive_percentage": 33.33,
    "neutral_percentage": 33.33,
    "negative_percentage": 33.33
  }
}
```

---

### 4. CSV Upload

**POST** `/predict/csv`

Upload CSV file for bulk analysis.

**Request:**
```bash
curl -X POST "http://localhost:8000/predict/csv" \
  -F "file=@data/test_reviews.csv"
```

**Response:**
```json
{
  "filename": "test_reviews.csv",
  "total_reviews": 10,
  "csv_data": "Review,Rating,Sentiment,Confidence\n..."
}
```

---

### 5. Model Information

**GET** `/models/info`

Get information about loaded models.

```bash
curl http://localhost:8000/models/info
```

**Response:**
```json
{
  "model_type": "RandomForestClassifier",
  "vectorizer_type": "TfidfVectorizer",
  "features": ["excellent", "great", "good"],
  "sentiment_classes": ["Negative", "Neutral", "Positive"]
}
```

## 🐍 Python Client Example

```python
import requests

# API base URL
BASE_URL = "http://localhost:8000"

# Single prediction
response = requests.post(
    f"{BASE_URL}/predict",
    json={"text": "This course is fantastic!"}
)
print(response.json())

# Batch prediction
response = requests.post(
    f"{BASE_URL}/predict/batch",
    json={
        "reviews": [
            "Great content!",
            "Poor quality",
            "Average course"
        ]
    }
)
print(response.json())

# CSV upload
with open("data/test_reviews.csv", "rb") as f:
    response = requests.post(
        f"{BASE_URL}/predict/csv",
        files={"file": f}
    )
print(response.json())
```

## 🌐 JavaScript/Node.js Example

```javascript
// Single prediction
fetch('http://localhost:8000/predict', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({
    text: 'This course is amazing!'
  })
})
.then(response => response.json())
.then(data => console.log(data));

// Batch prediction
fetch('http://localhost:8000/predict/batch', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({
    reviews: [
      'Excellent!',
      'Terrible',
      'Okay'
    ]
  })
})
.then(response => response.json())
.then(data => console.log(data));
```

## 🔒 Production Deployment

### Using Docker

Create `Dockerfile`:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements_api.txt .
RUN pip install --no-cache-dir -r requirements_api.txt

COPY . .

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
```

Build and run:

```bash
docker build -t mooc-sentiment-api .
docker run -p 8000:8000 mooc-sentiment-api
```

### Using Cloud Platforms

- **Heroku**: `heroku create` + `git push heroku main`
- **AWS Lambda**: Use AWS SAM or Serverless Framework
- **Google Cloud Run**: Deploy container directly
- **Azure App Service**: Deploy as web app

## 🧪 Testing

Test the API using the interactive documentation at http://localhost:8000/docs

Or use curl:

```bash
# Health check
curl http://localhost:8000/health

# Test prediction
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"text": "Amazing course!"}'
```

## 📝 License

MIT License - See LICENSE file

## 🤝 Contributing

Contributions welcome! Please open an issue or submit a PR.

## 📧 Contact

For questions or support, please open an issue on GitHub.

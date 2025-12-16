"""
REST API for MOOC Feedback Sentiment Analysis
FastAPI implementation for model deployment
"""

from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import List, Optional
from contextlib import asynccontextmanager
import joblib
import pandas as pd
import numpy as np
from io import StringIO
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    from data_preprocessing import DataPreprocessor
except ImportError:
    DataPreprocessor = None

# Global variables for models
model = None
vectorizer = None
preprocessor = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on startup and cleanup on shutdown"""
    global model, vectorizer, preprocessor
    try:
        model = joblib.load('models/optimized_best_model.pkl')
        vectorizer = joblib.load('models/tfidf_vectorizer.pkl')
        
        if DataPreprocessor is not None:
            preprocessor = DataPreprocessor()
        
        print("✅ Models loaded successfully!")
    except Exception as e:
        print(f"⚠️ Error loading models: {str(e)}")
    
    yield  # API runs here
    
    # Cleanup (if needed)
    print("🔄 Shutting down...")

# Initialize FastAPI app with lifespan
app = FastAPI(
    title="MOOC Feedback Sentiment Analysis API",
    description="API for analyzing sentiment in MOOC course reviews",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response models
class ReviewRequest(BaseModel):
    """Single review request"""
    text: str = Field(..., min_length=1, example="This course is amazing! I learned so much.")

class ReviewResponse(BaseModel):
    """Single review response"""
    text: str
    sentiment: str
    confidence: float
    probabilities: dict

class BatchReviewRequest(BaseModel):
    """Batch review request"""
    reviews: List[str] = Field(..., min_items=1, max_items=1000)

class BatchReviewResponse(BaseModel):
    """Batch review response"""
    total_reviews: int
    results: List[ReviewResponse]
    summary: dict

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    models_loaded: bool
    version: str

# Helper functions
def preprocess_text(text: str) -> str:
    """Preprocess review text"""
    if preprocessor is not None:
        return preprocessor.preprocess(text)
    else:
        # Simple fallback preprocessing
        return text.lower().strip()

def predict_sentiment(text: str) -> dict:
    """Predict sentiment for a single review"""
    if model is None or vectorizer is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    # Preprocess
    cleaned_text = preprocess_text(text)
    
    # Vectorize
    features = vectorizer.transform([cleaned_text])
    
    # Predict
    prediction = model.predict(features)[0]
    proba = model.predict_proba(features)[0]
    
    # Map prediction
    sentiment_map = {0: "Negative", 1: "Neutral", 2: "Positive",
                    "Negative": "Negative", "Neutral": "Neutral", "Positive": "Positive"}
    sentiment = sentiment_map.get(prediction, str(prediction))
    
    # Get confidence
    if isinstance(prediction, (int, np.integer)):
        confidence = float(proba[prediction])
    else:
        sentiment_to_idx = {"Negative": 0, "Neutral": 1, "Positive": 2}
        idx = sentiment_to_idx.get(prediction, 0)
        confidence = float(proba[idx])
    
    return {
        "sentiment": sentiment,
        "confidence": confidence,
        "probabilities": {
            "Negative": float(proba[0]),
            "Neutral": float(proba[1]),
            "Positive": float(proba[2])
        }
    }

# API Endpoints

@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint - API info"""
    return {
        "status": "running",
        "models_loaded": model is not None and vectorizer is not None,
        "version": "1.0.0"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy" if model is not None else "unhealthy",
        "models_loaded": model is not None and vectorizer is not None,
        "version": "1.0.0"
    }

@app.post("/predict", response_model=ReviewResponse)
async def predict(review: ReviewRequest):
    """
    Predict sentiment for a single review
    
    - **text**: The review text to analyze
    
    Returns sentiment (Negative/Neutral/Positive), confidence score, and probabilities
    """
    try:
        result = predict_sentiment(review.text)
        
        return {
            "text": review.text,
            "sentiment": result["sentiment"],
            "confidence": result["confidence"],
            "probabilities": result["probabilities"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/batch", response_model=BatchReviewResponse)
async def predict_batch(request: BatchReviewRequest):
    """
    Predict sentiment for multiple reviews
    
    - **reviews**: List of review texts (max 1000)
    
    Returns results for all reviews plus summary statistics
    """
    try:
        results = []
        sentiments = {"Positive": 0, "Neutral": 0, "Negative": 0}
        
        for text in request.reviews:
            result = predict_sentiment(text)
            results.append({
                "text": text,
                "sentiment": result["sentiment"],
                "confidence": result["confidence"],
                "probabilities": result["probabilities"]
            })
            sentiments[result["sentiment"]] += 1
        
        return {
            "total_reviews": len(request.reviews),
            "results": results,
            "summary": {
                "positive_count": sentiments["Positive"],
                "neutral_count": sentiments["Neutral"],
                "negative_count": sentiments["Negative"],
                "positive_percentage": round(sentiments["Positive"] / len(request.reviews) * 100, 2),
                "neutral_percentage": round(sentiments["Neutral"] / len(request.reviews) * 100, 2),
                "negative_percentage": round(sentiments["Negative"] / len(request.reviews) * 100, 2)
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/csv")
async def predict_csv(file: UploadFile = File(...)):
    """
    Upload CSV file and get sentiment predictions
    
    - **file**: CSV file with a 'Review' column
    
    Returns CSV with predictions
    """
    try:
        # Read CSV
        contents = await file.read()
        df = pd.read_csv(StringIO(contents.decode('utf-8')))
        
        if 'Review' not in df.columns:
            raise HTTPException(status_code=400, detail="CSV must contain 'Review' column")
        
        # Predict for all reviews
        predictions = []
        confidences = []
        
        for text in df['Review']:
            if pd.notna(text):
                result = predict_sentiment(str(text))
                predictions.append(result["sentiment"])
                confidences.append(result["confidence"])
            else:
                predictions.append("Unknown")
                confidences.append(0.0)
        
        # Add results to dataframe
        df['Sentiment'] = predictions
        df['Confidence'] = confidences
        
        # Convert to CSV
        csv_output = df.to_csv(index=False)
        
        return {
            "filename": file.filename,
            "total_reviews": len(df),
            "csv_data": csv_output
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/models/info")
async def model_info():
    """Get information about loaded models"""
    if model is None or vectorizer is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    return {
        "model_type": type(model).__name__,
        "vectorizer_type": type(vectorizer).__name__,
        "features": vectorizer.get_feature_names_out()[:20].tolist() if hasattr(vectorizer, 'get_feature_names_out') else [],
        "sentiment_classes": ["Negative", "Neutral", "Positive"]
    }

@app.get("/paper/download")
async def download_paper():
    """
    Download the research paper PDF
    
    Returns the IEEE conference paper as a downloadable PDF file
    """
    paper_path = os.path.join(os.path.dirname(__file__), 'paper', 'reseachpaper.pdf')
    
    if not os.path.exists(paper_path):
        raise HTTPException(status_code=404, detail="Research paper not found")
    
    return FileResponse(
        path=paper_path,
        media_type='application/pdf',
        filename='MOOC_Sentiment_Analysis_IEEE_Paper.pdf',
        headers={
            "Content-Disposition": "attachment; filename=MOOC_Sentiment_Analysis_IEEE_Paper.pdf"
        }
    )

@app.get("/paper/info")
async def paper_info():
    """Get information about the research paper"""
    paper_path = os.path.join(os.path.dirname(__file__), 'paper', 'reseachpaper.pdf')
    
    return {
        "title": "Intelligent Sentiment Analysis of MOOC Reviews: A Deep Learning Approach for Educational Feedback Mining",
        "format": "IEEE Conference Paper",
        "pages": 8,
        "available": os.path.exists(paper_path),
        "download_url": "/paper/download",
        "authors": "Research Team",
        "conference": "IEEE Conference",
        "year": 2025,
        "keywords": ["sentiment analysis", "MOOC", "educational data mining", "BERT", "machine learning", "NLP"],
        "abstract": "This paper presents a comprehensive sentiment analysis framework for automated processing of MOOC reviews. We evaluated four ML approaches on 140,322 Coursera reviews, achieving 87.2% accuracy with BERT.",
        "dataset_size": "140,322 reviews",
        "models_compared": ["Logistic Regression", "Naive Bayes", "Random Forest", "BERT"],
        "best_accuracy": "87.2% (BERT)"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)

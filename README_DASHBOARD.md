# 📚 MOOC Feedback Analyzer - Streamlit Dashboard

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install streamlit
```

(Other dependencies should already be installed from requirements.txt)

### 2. Run the Dashboard
```bash
streamlit run app.py
```

The dashboard will open automatically in your browser at `http://localhost:8501`

## 📱 Features

### 🏠 Home Page
- Project overview and statistics
- Dataset insights
- Rating distribution visualization

### 🔍 Single Review Analysis
- Enter any course review
- Get instant sentiment prediction (Positive/Neutral/Negative)
- View confidence scores
- See probability distribution
- View preprocessed text

### 📁 Batch Analysis
- Upload CSV files with multiple reviews
- Analyze hundreds of reviews at once
- Download results as CSV
- View sentiment distribution
- Analyze confidence scores

### 🔬 Model Insights
- Compare all models (Logistic Regression, Naive Bayes, Random Forest, BERT)
- View performance metrics
- See feature importance
- Understand model decisions

### ℹ️ About
- Project information
- Technical details
- Team information

## 📊 Usage Examples

### Single Review Analysis
1. Go to "Single Review Analysis" page
2. Type or paste a course review
3. Click "Analyze Sentiment"
4. View results with confidence scores

### Batch Analysis
1. Prepare a CSV file with a column containing reviews
2. Go to "Batch Analysis" page
3. Upload your CSV file
4. Select the review column
5. Click "Analyze All Reviews"
6. Download results

## 🎨 Customization

You can customize the dashboard by editing `app.py`:
- Change colors in the CSS section
- Modify model comparison data
- Add new visualizations
- Update feature importance data

## 📝 Notes

- The dashboard uses the trained Random Forest model by default (`models/best_model.pkl`)
- Make sure you've trained the models using the notebooks before running the dashboard
- For BERT predictions, you'll need to load the BERT model separately (currently using traditional ML models)

## 🐛 Troubleshooting

**Models not found error:**
- Make sure you've run the training notebooks (02 and 03)
- Check that `models/best_model.pkl` and `models/vectorizer.pkl` exist

**Import errors:**
- Verify all dependencies are installed: `pip install -r requirements.txt`

**Port already in use:**
- Run with a different port: `streamlit run app.py --server.port 8502`

## 🔗 Links

- GitHub: https://github.com/param20h/mooc-feedback-mining-msme
- Dataset: Coursera Course Reviews (Kaggle)
- Documentation: See main README.md

---

Enjoy analyzing course feedback! 🎓

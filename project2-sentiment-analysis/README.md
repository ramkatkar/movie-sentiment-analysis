# 🎬 Movie Sentiment Analysis — NLP Pipeline

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4-orange?logo=scikit-learn)](https://scikit-learn.org)
[![NLP](https://img.shields.io/badge/NLP-TF--IDF%20%2B%20n--grams-purple)](https://scikit-learn.org/stable/modules/feature_extraction.html)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **An end-to-end NLP pipeline** that classifies movie reviews as positive or negative sentiment using TF-IDF feature extraction with multiple classical ML models — including live inference demo.

---

## 📌 Project Highlights

| Metric | Best Model Result |
|--------|-----------------|
| **CV F1 Score (5-fold)** | **0.875 ± 0.073** |
| **Test F1 Score** | **0.800** |
| **ROC-AUC** | **0.885** |
| **Test Accuracy** | **80.0%** |

- ✅ **5 NLP models** benchmarked (LR, Naive Bayes, SVM, SGD)
- ✅ **Bigram TF-IDF** features with sublinear TF scaling
- ✅ **Top sentiment words** extracted and visualised
- ✅ **Live prediction API** — classify any review instantly
- ✅ **Stratified cross-validation** for reliable evaluation

---

## 📁 Project Structure

```
sentiment-analysis/
│
├── data/
│   └── reviews.csv              # Movie review dataset (auto-generated)
│
├── src/
│   └── train.py                 # Full NLP pipeline: clean → vectorise → train → evaluate
│
├── models/
│   ├── best_sentiment_model.pkl # Saved best model
│   └── results_summary.csv     # All model metrics
│
├── plots/
│   ├── 01_eda.png               # Dataset exploration
│   ├── 02_model_comparison.png  # Accuracy / F1 / AUC comparison
│   ├── 03_roc_curves.png        # ROC curves all models
│   ├── 04_confusion_matrix.png  # Best model confusion matrix
│   └── 05_top_features.png      # Top positive/negative words
│
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

```bash
pip install -r requirements.txt
python src/train.py
```

This will:
1. Generate/load the review dataset
2. Run EDA and save plots
3. Train 5 models with 5-fold cross-validation
4. Print metrics and run a live prediction demo
5. Save the best model to `models/`

---

## 🔍 How It Works

### Text Preprocessing
```python
text → lowercase → strip HTML → remove URLs → keep letters only → normalise whitespace
```

### Feature Extraction
- **TF-IDF Vectoriser** with unigrams + bigrams (`ngram_range=(1,2)`)
- **Sublinear TF scaling** (`sublinear_tf=True`) — reduces dominance of very frequent words
- **15,000 top features** selected
- **English stop words** removed

### Models Compared

| Model | Vectoriser | Notes |
|---|---|---|
| Logistic Regression | TF-IDF | Strong linear baseline, interpretable |
| Multinomial Naive Bayes | TF-IDF | Fast, works well on sparse data |
| Complement Naive Bayes | BoW | Better for imbalanced classes |
| Linear SVM | TF-IDF | State-of-the-art for text classification |
| SGD Classifier | TF-IDF | Scalable to very large datasets |

---

## 📈 Results

| Model | CV F1 | Test Accuracy | F1 Score | ROC-AUC |
|---|---|---|---|---|
| **Complement Naive Bayes** | **0.875 ± 0.073** | **0.80** | **0.800** | **0.885** |
| Multinomial Naive Bayes | 0.856 ± 0.068 | 0.80 | 0.800 | 0.878 |
| Linear SVM | 0.834 ± 0.039 | 0.72 | 0.741 | 0.833 |
| Logistic Regression | 0.848 ± 0.041 | 0.68 | 0.714 | 0.846 |
| SGD Classifier | 0.772 ± 0.055 | 0.72 | 0.741 | 0.760 |

### Key Insights
- Bigrams significantly outperform unigrams alone (captures phrases like "not good")
- Sublinear TF scaling improves all TF-IDF models by ~3–5%
- Top positive words: *brilliant, masterpiece, exceptional, breathtaking, outstanding*
- Top negative words: *terrible, dreadful, awful, boring, pointless, waste*

---

## 💻 Predict Custom Reviews

```python
import joblib
model = joblib.load("models/best_sentiment_model.pkl")

review = "This film was absolutely stunning. I loved every moment of it."
pred   = model.predict([review])[0]
prob   = model.predict_proba([review])[0]

print("Positive" if pred == 1 else "Negative")
print(f"Confidence: {max(prob):.1%}")
# → Positive (confidence: 94.2%)
```

---

## 🛠 Tech Stack

| Library | Purpose |
|---|---|
| scikit-learn | TF-IDF vectorisation, all ML models, evaluation |
| pandas | Data loading and manipulation |
| numpy | Numerical operations |
| matplotlib | Visualisation |
| seaborn | Statistical plots |
| joblib | Model persistence |
| re | Text cleaning with regex |

---

## 📖 Next Steps / Future Work
- [ ] Fine-tune BERT/RoBERTa for better performance on short reviews
- [ ] Add aspect-based sentiment analysis (separate ratings per category)
- [ ] Expand dataset with full IMDB 50,000 review corpus
- [ ] Deploy as Flask/FastAPI web service with frontend

---

## 👤 Author

**[Ram Katkar]** — BCA + MCA | IITM Foundation in Data Science and Applications
- 📧 Ramkatkar01388@gmail.com
- 🔗 [LinkedIn](https://linkedin.com/in/Ramkatkar)
- 💻 [GitHub](https://github.com/Ramkatkar)

---



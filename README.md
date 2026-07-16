# 🐦 Twitter Sentiment Analysis using PySpark

> Distributed large-scale sentiment classification of tweets using Apache Spark's MLlib — classifying public opinion into **Positive**, **Negative**, and **Neutral** categories.

---

## 📌 Overview

Social media platforms like Twitter generate over **500 million tweets daily**, making manual sentiment analysis impractical. This project leverages **PySpark** — a distributed big data framework — to efficiently process and classify large volumes of tweets using Natural Language Processing (NLP) and Machine Learning.

Organizations, governments, and businesses rely on sentiment analysis to:
- 📊 Track public opinion in real time
- 🏷️ Monitor brand reputation
- 💡 Improve customer satisfaction
- 📈 Drive data-driven decisions

---

## 🎯 Objectives

- Preprocess raw Twitter data at scale using PySpark DataFrames
- Extract meaningful features using **TF-IDF** vectorization
- Train and evaluate multiple ML models via **PySpark MLlib**
- Compare models using **Accuracy** and **F1-Score** metrics
- Lay groundwork for real-time analysis using **Spark Structured Streaming**

---

## 📂 Dataset

**Sentiment140** — A benchmark Twitter sentiment dataset containing **1.6 million tweets**, labeled as positive or negative.

- Source: [Sentiment140](http://help.sentiment140.com/for-students)
- Columns: `sentiment`, `id`, `date`, `query`, `user`, `text`
- Subset used: **~10% (~128K tweets)** for computational feasibility

---

## 🏗️ Methodology

```
Raw Tweets (Sentiment140)
        │
        ▼
  Data Loading (PySpark DataFrames)
        │
        ▼
  Data Cleaning
  (Remove URLs, hashtags, mentions, emojis, special chars)
        │
        ▼
  Text Preprocessing
  (Lowercase → Tokenization → Stopword Removal)
        │
        ▼
  Feature Extraction
  (HashingTF + IDF → Bigrams)
        │
        ▼
  Model Training (MLlib)
  ┌─────────────┬──────────────┬───────────────┐
  │  Logistic   │  Naive Bayes │ Random Forest │
  │ Regression  │              │               │
  └─────────────┴──────────────┴───────────────┘
        │
        ▼
  Cross-Validation & Hyperparameter Tuning
  (3-fold & 5-fold CV)
        │
        ▼
  Model Evaluation (Accuracy, F1-Score)
```

---

## 📊 Results

| Model               | Accuracy (3-fold CV) | Accuracy (5-fold CV) | Training Time |
|---------------------|----------------------|----------------------|---------------|
| Logistic Regression | 0.692                | 0.692                | ~300 sec      |
| Naive Bayes         | 0.509                | 0.508                | ~1500 sec     |
| **Random Forest**   | **0.718**            | **0.720**            | ~2220 sec     |

### 🏆 Key Takeaways
- **Random Forest** with 5-fold CV (**72.0% accuracy**) outperformed all other models
- Logistic Regression was fastest but less accurate (69.2%)
- Naive Bayes struggled due to sparse text assumptions (50.9%)
- Text cleaning + bigrams significantly boosted Random Forest performance
- Using the full 1.6M dataset is expected to yield higher accuracy

---

## ⚙️ Tech Stack

| Tool | Purpose |
|------|---------|
| **Apache Spark / PySpark** | Distributed data processing |
| **PySpark MLlib** | Machine learning pipeline |
| **Google Colab** | Cloud runtime environment |
| **Google Drive** | Dataset storage |
| **Python 3** | Core programming language |

---

## 🚀 How to Run

### Prerequisites
- Google Colab (recommended) or a local Spark cluster
- PySpark installed: `pip install pyspark`
- Sentiment140 dataset loaded into Google Drive

### Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/<your-username>/Twitter-Sentiment-Analysis-PySpark.git
   cd Twitter-Sentiment-Analysis-PySpark
   ```

2. **Open the notebook**
   - Upload `Sentiment_Analysis_on_Tweets.ipynb` to Google Colab

3. **Mount Google Drive** *(first cell)*
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```

4. **Place the dataset** at:
   ```
   /content/drive/MyDrive/Colab Notebooks/Pyspark/Datasets/tweets/tweets.csv
   ```

5. **Run all cells** sequentially

---

## 🔮 Future Enhancements

- [ ] Test advanced embeddings: **Word2Vec**, **GloVe**, **BERT**
- [ ] Train on full **1.6M tweet** dataset using a Spark cluster
- [ ] Implement **real-time analysis** via Spark Structured Streaming
- [ ] Add multi-class sentiment (beyond binary positive/negative)
- [ ] Deploy as a web dashboard for live tweet monitoring

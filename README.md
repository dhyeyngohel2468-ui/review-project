# 🛡️ Hybrid Review Authenticity Detection System

An explainable, ML-powered application that analyzes Amazon product reviews using **Behavioral Anomaly Detection, Natural Language Processing (NLP), and a Hybrid Decision Engine** to assess review authenticity.

The application combines reviewer activity patterns with textual characteristics to identify potentially suspicious reviews and provide interpretable results through an interactive Streamlit dashboard.

---

## 📌 Project Overview

Online product reviews play an important role in purchasing decisions, but manipulated and misleading reviews can affect consumer trust.

The **Hybrid Review Authenticity Detection System** addresses this problem by evaluating two independent aspects:

1. **Reviewer Behavior Analysis:** Examines historical reviewing patterns to identify unusual or potentially suspicious activity.
2. **Review Text Analysis:** Uses NLP and machine learning to identify linguistic patterns associated with suspicious or genuine reviews.

Both signals are combined through a Hybrid Decision Engine to generate an overall authenticity score.

Unlike systems that rely on a single prediction, this application provides separate behavioral and textual assessments, along with explanations of the factors influencing the final result.

---

## ✨ Key Features

- **Interactive Web Application:** Complete Streamlit frontend with a dark-themed dashboard and multiple analytical sections.
- **Behavioral Anomaly Detection:** Uses Isolation Forest to identify unusual reviewer activity.
- **NLP-Based Classification:** Uses TF-IDF vectorization and Logistic Regression to analyze review text.
- **Hybrid Decision Engine:** Combines behavioral and textual scores using configurable weights.
- **Review Analysis:** Select and analyze reviews from the integrated Amazon review dataset.
- **Reviewer Lookup:** Search, inspect, compare, and rank modeled reviewers.
- **Cold-Start Handling:** Uses text-only analysis when a reviewer has fewer than three historical reviews.
- **Explainable Results:** Displays behavioral indicators, linguistic signals, score breakdowns, and decision explanations.
- **EDA & Insights:** Includes visualizations of reviewer activity and rating distributions.
- **Dataset Exploration:** Browse historical review records and analyze patterns across the dataset.

---

## 🏗️ System Architecture

The application follows a modular architecture consisting of three major components.

### 1. Behavioral Analysis Module

This module analyzes reviewer-level historical activity rather than judging individual review text.

**Algorithm:** Isolation Forest (Unsupervised Learning)

The model uses engineered behavioral features, including:

| Feature | Description |
|---|---|
| Rating Variance | Measures variation in a reviewer's ratings. |
| Rating Entropy | Measures the distribution of ratings across different star values. |
| Reviews Per Day | Measures the review submission rate. |
| Review Burst (7 Days) | Maximum number of reviews submitted within any seven-day window. |
| Burst Ratio (7 Days) | Measures the concentration of reviewing activity within a burst. |
| Average Days Between Reviews | Measures the average time interval between consecutive reviews. |
| Burst Intensity | Combines review burst frequency and burst ratio. |

The Isolation Forest identifies unusual combinations of these behavioral patterns.

The module produces a **Behavior Score (0–100)**, where:

- Higher scores indicate more normal behavioral patterns.
- Lower scores indicate potentially anomalous behavior.

This score is a behavioral assessment, not a probability of authenticity.

### 2. NLP Module

This module evaluates the textual characteristics of individual product reviews.

**Technologies:**
- TF-IDF (Term Frequency–Inverse Document Frequency)
- Logistic Regression
- Scikit-learn

The text processing pipeline converts review content into numerical representations using TF-IDF and passes these representations to the trained classification model.

The model evaluates linguistic patterns such as:
- Excessive promotional language.
- Repetitive or generic expressions.
- Overly enthusiastic wording.
- Potentially incentivized review language.

The module produces a **Text Authenticity Score (0–100)**, where higher scores indicate text classified as more likely to be genuine.

### 3. Hybrid Decision Engine

The decision engine combines the outputs of the Behavioral and NLP modules.

When both scores are available, the final score is calculated using:

`Final Score = (wB × Behavior Score) + (wT × Text Score)`

Where:

- `wB` = Behavioral weight.
- `wT` = Text weight.
- `wB + wT = 1`.

The behavioral weight can be adjusted through the application interface.

**Decision interpretation:**

| Final Score | Interpretation |
|---|---|
| 75–100 | Most probably genuine |
| 55–74 | Needs to be reviewed manually |
| 30–54 | Suspicious |
| 0–29 | Highly suspicious |

**Cold-start handling:**

When a reviewer has fewer than three historical reviews, behavioral analysis is skipped because the available history is insufficient for meaningful behavioral assessment.

In such cases, the Text Authenticity Score is used directly without applying hybrid weighting.

If only one of the two signals is available, the available score is used independently.

---

## 🖥️ Application Modules

The Streamlit application provides the following sections:

| Module | Functionality |
|---|---|
| Overview | Project introduction, system statistics, and detection summary. |
| Architecture | Explains the overall system architecture and processing pipeline. |
| EDA & Insights | Visualizes reviewer statistics and behavioral patterns. |
| Behavioral Model | Explores anomaly detection, reviewer scores, and feature-level insights. |
| NLP Module | Demonstrates text classification and linguistic analysis. |
| Analyze Review | Selects an actual Amazon review and runs the hybrid analysis pipeline. |
| Reviewer Lookup | Searches and explores modeled reviewers and their behavioral scores. |

---

## 🛠️ Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Frontend / Application | Streamlit |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn |
| Behavioral Algorithm | Isolation Forest |
| NLP | TF-IDF, Logistic Regression |
| Model Persistence | Joblib, Pickle |
| Data Visualization | Matplotlib |
| Development Environment | Jupyter Notebook, VS Code |
| Dataset | Amazon Product Reviews |

---

## 📂 Project Structure

```text
Hybrid-Review-Authenticity-Detection/
│
├── app/
│   ├── main.py
│   └── explaination.pdf
│
├── Dataset/
│   ├── reviewer_dataset.csv
│   ├── Amazon_Reviews_Combined_2000-2023.csv
│   ├── Reviewer Dataset(2000-2014).csv
│   ├── Reviewer Dataset(2015-2023).csv
│   └── Other source datasets...
│
├── models/
│   ├── isolation_forest.pkl
│   ├── scaler.pkl
│   ├── score_scaler.pkl
│   ├── nlp_model.pkl
│   └── tfidf_vectorizer.pkl
│
├── Outputs/
│   ├── Behaviour Score.csv
│   ├── anomaly.csv
│   └── normal.csv
│
├── EDA_Plots/
│   └── Behavioral analysis visualizations...
│
├── notebooks/
│   ├── EDA.ipynb
│   ├── behavorial_model_01.ipynb
│   ├── behavorial_model_02.ipynb
│   ├── behavorial_model_03.ipynb
│   ├── behavorial_model_final.ipynb
│   └── Feature engineering notebooks...
│
├── archive/
│   └── Previous experiments and intermediate files
│
├── .streamlit/
│   └── config.toml
│
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation and Setup

### Prerequisites

- Python 3.10–3.13
- pip (Python package manager)
- Git (optional, for cloning the repository)

### 1. Clone the Repository

```bash
git clone https://github.com/dhyeyngohel2468-ui/review-project.git
```

Navigate into the project directory:

```bash
cd review_project
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

Activate the environment.

**Windows:**

```bash
venv\Scripts\activate
```

**Linux / macOS:**

```bash
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Verify Project Files

Ensure the following directories and files are present:

- `Dataset/` – Required datasets.
- `models/` – Pre-trained machine learning models and preprocessing objects.
- `app/main.py` – Main Streamlit application.
- `requirements.txt` – Project dependencies.

The application expects these files to be available in their respective directories.

### 5. Run the Application

From the root project directory, execute:

```bash
streamlit run app/main.py
```

The application will start locally and can be accessed through the URL displayed in your terminal, typically:

```text
http://localhost:8501
```

---

## 📊 Dataset Information

The project uses historical Amazon product review data spanning **2000–2023**.

The application primarily depends on two datasets:

| Dataset | Purpose |
|---|---|
| `reviewer_dataset.csv` | Aggregated reviewer-level features used for behavioral analysis. |
| `Amazon_Reviews_Combined_2000-2023.csv` | Individual review records used for review selection and text analysis. |

The behavioral dataset contains reviewer statistics and engineered features, while the combined review dataset contains individual review information, including reviewer IDs, product IDs, ratings, titles, and review text.

Additional intermediate datasets and notebooks are retained for feature engineering, experimentation, and reproducibility.

---

## 💾 Trained Models

The application uses pre-trained models and preprocessing objects stored in the `models/` directory.

| File | Purpose |
|---|---|
| `isolation_forest.pkl` | Trained behavioral anomaly detection model. |
| `scaler.pkl` | Feature scaling for behavioral inputs. |
| `score_scaler.pkl` | Transforms behavioral anomaly outputs into a normalized score. |
| `nlp_model.pkl` | Trained Logistic Regression text classification model. |
| `tfidf_vectorizer.pkl` | Converts review text into TF-IDF features. |

These files are loaded when the application starts, avoiding the need to retrain the models for every execution.

---

## ⚠️ Important Notes and Limitations

- The system provides an authenticity assessment, not a definitive fake-or-genuine verdict.
- Behavioral anomaly detection identifies unusual reviewing patterns, which do not necessarily indicate fraudulent activity.
- NLP predictions depend on the training data and learned linguistic patterns.
- Reviewers with fewer than three historical reviews are handled through text-only analysis.
- The application operates using the included historical datasets and saved machine learning models.
- Results should be interpreted as decision-support signals rather than absolute proof of review manipulation.

---

## 🚀 Future Scope

Potential improvements include:

- Integrating live review collection through supported data sources.
- Expanding the NLP pipeline with transformer-based language models.
- Exploring more advanced behavioral anomaly detection techniques.
- Adding a database layer for persistent review and prediction storage.
- Deploying the application on a cloud platform.
- Improving model generalization using larger and more diverse labeled datasets.

---

## 👥 Project Team

**Project:** Hybrid Review Authenticity Detection System

**Institution:** Indus University

**Department:** Computer Engineering

**Semester:** 7th Semester

**Project Type:** Software Group Project

**Team Members:**
- Dhyey Gohel
- Shrey Chauhan
- Nimesh Panchal
- Kirtisinh Rajput

---

## 📄 License

This project was developed for academic purposes as part of the Software Group Project curriculum at Indus University.

Dataset ownership and usage remain subject to the original dataset sources and their respective licenses and terms of use.

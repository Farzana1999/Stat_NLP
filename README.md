IMDB Sentiment Analysis - README

Project Overview

This project performs sentiment analysis on the IMDB movie review dataset. It explores various machine learning models (Logistic Regression, Support Vector Machines, Multinomial Naive Bayes) using Bag-of-Words and TF-IDF features. It also includes text preprocessing, feature extraction, model evaluation, and visualization via word clouds.

Dataset

File: IMDB Dataset.csv

Location: C:\Users\ual-laptop\Documents\stat nlp

Attributes:

review: Textual content of the movie review

sentiment: Sentiment label (positive/negative)

Installation & Requirements

Make sure the following Python libraries are installed:

pip install numpy pandas seaborn matplotlib nltk scikit-learn wordcloud bs4 spacy textblob

Also, download the necessary NLTK resources:

import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

Data Preprocessing

HTML Removal

Square Bracket Text Removal

Special Character Removal

Stemming (Porter Stemmer)

Stopword Removal (using NLTK's stopwords)

Feature Engineering

Bag-of-Words (BoW): CountVectorizer(ngram_range=(1,3))

TF-IDF: TfidfVectorizer(ngram_range=(1,3))

Model Training

Models trained and evaluated:

Logistic Regression

Support Vector Machine (SGDClassifier with hinge loss)

Multinomial Naive Bayes

Each model is trained and evaluated using both BoW and TF-IDF features.

Evaluation Metrics

Accuracy Score

Classification Report (Precision, Recall, F1-score)

Confusion Matrix

Visualization

WordClouds generated for a sample positive and negative review.

Output Example

# Logistic Regression Accuracy:
lr_bow_score : 0.89
lr_tfidf_score : 0.91

# Naive Bayes Model Representation:
MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)

How to Run

Place IMDB Dataset.csv in the specified directory.

Run the script in a Jupyter notebook or Python environment.

Evaluate the model scores and confusion matrices.

Notes

Uncomment the spelling correction and textblob tokenization if you want deeper preprocessing (it's commented for performance reasons).

Ensure the models are trained on normalized text (norm_train_reviews).

You can customize the vectorizer ngram_range, min_df, and max_df as needed.

© 2025, Sentiment Analysis Project by Farzana Dudekula


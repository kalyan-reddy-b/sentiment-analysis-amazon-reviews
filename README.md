# sentiment-analysis-amazon-reviews

This project applies natural language processing (NLP) techniques to perform sentiment analysis on Amazon Fine Food Reviews. It classifies customer reviews into positive, neutral, or negative sentiments using a logistic regression model trained on TF-IDF features.

## 📝 Files in This Project
- Amazon Fine Food Reviews.csv — Raw dataset containing Amazon food product reviews.

- sentiment_analysis.py — Main script for data preprocessing, model training, evaluation, and saving the model and vectorizer.

## 🧠 What the Script Does
- Loads and cleans the dataset.

- Converts review scores to sentiment labels:

   - 1-2 → Negative

   - 3 → Neutral

   - 4-5 → Positive

- Balances the dataset by sampling an equal number of reviews from each sentiment.

- Extracts features using TF-IDF vectorization.

- Trains a logistic regression model on the processed data.

- Evaluates the model using accuracy and classification report.

- Saves the trained model and vectorizer using joblib for future use.

## 🛠️ Technologies Used
- Python

- Pandas

- Scikit-learn

- Joblib

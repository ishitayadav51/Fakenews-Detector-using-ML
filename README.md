# Fakenews-Detector-using-ML
🧠 Project Summary
This machine learning project focuses on classifying news articles as fake or real. It leverages a labeled dataset consisting of both real and fake news items. The pipeline involves text cleaning, TF-IDF feature extraction, and building a Logistic Regression classifier to make accurate predictions.

📂 Dataset Details

Two CSV files are used as the data source:
Fake.csv – includes fabricated news content
True.csv – contains verified, factual news articles
These files are combined into a single dataset, where each article is tagged accordingly to indicate whether it's real or fake.

📌 Source: Kaggle – Fake and Real News Dataset

💻 Tools & Libraries

The following technologies were used in the implementation:
Python 🐍
Scikit-learn
Pandas
NumPy
TF-IDF Vectorizer
Logistic Regression

⚙️ Implementation Workflow

Here's a step-by-step breakdown of the system:
Data Importing: Load both Fake.csv and True.csv files.
Labeling: Tag fake news with 1 and real news with 0.
Data Merging & Shuffling: Combine both datasets and shuffle them to avoid any ordering bias.
Text Vectorization: Apply TF-IDF to convert text data into numerical format.
Splitting the Data: Split the dataset into training and testing sets.
Model Training: Use Logistic Regression to train the classifier.
Model Evaluation: Assess the model’s performance using metrics like accuracy, precision, and recall.
User Prediction: Allow input of custom news sentences for prediction.

✅ Performance

Accuracy: Achieves around 92% on the test set (may vary slightly depending on the random seed and data split)
Works well for articles of varying lengths
Can be enhanced further by integrating models like SVM, Naive Bayes, or advanced techniques like BERT


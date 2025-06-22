import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score

# Load both datasets
df_fake = pd.read_csv("Fake.csv")
df_true = pd.read_csv("True.csv")

 # Add a label column: 0 for true news, 1 for fake news
df_fake['label'] = 1
df_true['label'] = 0

# Combine the data
df = pd.concat([df_fake, df_true], axis=0)

# Shuffle the combined dataset
df = df.sample(frac=1).reset_index(drop=True)

# Show few rows
df.head()

# Use 'text' column as feature and 'label' as target
X = df['text']
y = df['label']


vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_vectorized = vectorizer.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)


model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


sample_news = ["The government passed a new policy to hazard school standards."]
sample_vector = vectorizer.transform(sample_news)
prediction = model.predict(sample_vector)

print("Prediction:", "Fake" if prediction[0] == 1 else "Real")












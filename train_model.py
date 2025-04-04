import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

# Step 1: Load the dataset (Ensure it has 'message' and 'label' columns)
df = pd.read_csv("spam_dataset.csv")  # Replace with your actual dataset file

# Convert 'label' column to numerical values (Spam: 1, Ham: 0)
df['label'] = df['label'].map({'spam': 1, 'ham': 0})

# Step 2: Define features and labels
X = df["message"]  # Text messages
y = df["label"]    # Spam (1) or Ham (0)

# Step 3: Split dataset into training & testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train TfidfVectorizer on training data
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)

# Save the trained TfidfVectorizer
with open("TfidfVectorizer.pkl", "wb") as file:
    pickle.dump(vectorizer, file)
print("✅ TfidfVectorizer.pkl saved successfully!")

# Step 5: Train a Naïve Bayes classifier
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# Save the trained model
with open("mymodel.pkl", "wb") as file:
    pickle.dump(model, file)
print("✅ mymodel.pkl saved successfully!")

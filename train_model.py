import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle

# Load dataset
data = pd.read_csv("spam.csv", encoding='latin-1')

# Clean unwanted columns (the Kaggle dataset has extra ones)
data = data[['v1', 'v2']]
data.columns = ['label', 'message']

# Split data
x_train, x_test, y_train, y_test = train_test_split(
    data['message'], data['label'], test_size=0.2, random_state=42
)

# Convert text to numeric (bag of words)
cv = CountVectorizer(stop_words='english')
x_train_cv = cv.fit_transform(x_train)

# Train Naive Bayes model
model = MultinomialNB()
model.fit(x_train_cv, y_train)

# Save the model and vectorizer
pickle.dump(model, open('spam_model.pkl', 'wb'))
pickle.dump(cv, open('vectorizer.pkl', 'wb'))

# Check accuracy
accuracy = model.score(cv.transform(x_test), y_test)
print("✅ Model trained successfully!")
print("Accuracy:", round(accuracy * 100, 2), "%")

import pandas as pd

# Read the dataset
data = pd.read_csv('news.csv')  # Replace with your dataset path
print(data.head())  # Display the first few rows of the dataset
print(data.info())  # Get information about the dataset (columns, data types, etc.)
print(data['label'].value_counts())  # Check distribution of labels

from sklearn.feature_extraction.text import TfidfVectorizer

# Separate features and labels
X = data['text']
y = data['label']

# Initialize TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)

# Convert text data into numerical format
X_tfidf = tfidf_vectorizer.fit_transform(X)

from sklearn.model_selection import train_test_split
from sklearn.linear_model import PassiveAggressiveClassifier

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

# Build the PassiveAggressiveClassifier model
pac = PassiveAggressiveClassifier(max_iter=50)
pac.fit(X_train, y_train)

from sklearn.metrics import accuracy_score, confusion_matrix

# Make predictions on the test set
y_pred = pac.predict(X_test)

# Evaluate model performance
accuracy = accuracy_score(y_test, y_pred)
confusion = confusion_matrix(y_test, y_pred)

# Display the model's performance
print(f'Accuracy: {accuracy}')
print(f'Confusion Matrix:\n{confusion}')

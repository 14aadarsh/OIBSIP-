#Email spam detection with machine learning

# Importing necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

#Load the dataset
data = pd.read_csv('OSISTASK4.csv', encoding='latin-1')

#Data preprocessing
data = data[['v1', 'v2']]
data.columns = ['label', 'message']

# Convert labels to binary (spam=1, ham=0)
data['label'] = data['label'].map({'spam': 1, 'ham': 0})

# Splitting the data into train and test sets
X = data['message']
y = data['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature extraction
vectorizer = CountVectorizer()
X_train_counts = vectorizer.fit_transform(X_train)
X_test_counts = vectorizer.transform(X_test)

# Training the model
model = MultinomialNB()
model.fit(X_train_counts, y_train)

# Predicting on test set
y_pred = model.predict(X_test_counts)

# Evaluating the model
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print(f'Accuracy: {accuracy * 100:.2f}%')
print('Confusion Matrix:')
print(cm)
print('Classification Report:')
print(classification_report(y_test, y_pred))

# Visualizing the confusion matrix
plt.figure(figsize=(10, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
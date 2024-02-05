import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt
import seaborn as sns
from flask import Flask, request, jsonify

# Load your dataset (replace 'your_dataset.csv' with the actual file name)
df = pd.read_csv('"D:\\NEXUS\\project-2\\archive"')

# Data Preprocessing
def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word.lower()) for word in tokens if word.isalpha()]
    tokens = [word for word in tokens if word not in stop_words]
    
    return ' '.join(tokens)

df['processed_text'] = df['text'].apply(preprocess_text)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(df['processed_text'], df['sentiment'], test_size=0.2, random_state=42)

# Text Vectorization
vectorizer = TfidfVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Model Selection (Multinomial Naive Bayes)
model = MultinomialNB()
model.fit(X_train_vectorized, y_train)

# Model Evaluation
y_pred = model.predict(X_test_vectorized)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print(f'Confusion Matrix:\n{conf_matrix}')
print(f'Classification Report:\n{classification_rep}')

# Cross-Validation
cv_scores = cross_val_score(model, X_train_vectorized, y_train, cv=5)
print(f'Cross-Validation Scores: {cv_scores}')
print(f'Mean CV Score: {cv_scores.mean()}')

# Feature Importance (for RandomForestClassifier)
rf_model = RandomForestClassifier()
rf_model.fit(X_train_vectorized, y_train)
feature_importance = pd.DataFrame({'Feature': vectorizer.get_feature_names_out(), 'Importance': rf_model.feature_importances_})
sorted_feature_importance = feature_importance.sort_values(by='Importance', ascending=False)
print(sorted_feature_importance.head(10))  # Display top 10 important features

# Deployment (Flask API)
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict_sentiment():
    text = request.json['text']
    processed_text = preprocess_text(text)
    vectorized_text = vectorizer.transform([processed_text])
    prediction = model.predict(vectorized_text)[0]
    return jsonify({'sentiment': prediction})

if __name__ == '__main__':
    app.run(debug=True)

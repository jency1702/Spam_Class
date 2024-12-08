import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Use st.cache_data instead of st.cache
@st.cache_data
def load_data():
    
    try:
        return pd.read_csv('emails.csv')[['text', 'spam']].dropna()
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

def preprocess_data(data):
    X = data['text']

    y = pd.to_numeric(data['spam'], errors='coerce').fillna(0).astype(int)
    vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
    X = vectorizer.fit_transform(X)
    return X, y, vectorizer

def train_model(X_train, X_test, y_train, y_test):
    model = MultinomialNB()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    return model, accuracy, report, cm

st.title("Spam Email Classification using NLP and Machine Learning")

# Load the dataset
data = load_data()

if data is not None:
    X, y, vectorizer = preprocess_data(data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model, accuracy, report, cm = train_model(X_train, X_test, y_train, y_test)

    st.subheader("Model Performance")
    st.write(f"Accuracy: {accuracy:.2f}")
    st.subheader("Classification Report")
    st.text(report)
    st.subheader("Confusion Matrix")
    st.write(cm)

    st.subheader("Test the Model with Your Email")
    input_text = st.text_area("Enter the email content")

    if st.button("Predict"):
        if input_text:
            input_data = vectorizer.transform([input_text])
            prediction = model.predict(input_data)
            result = "Spam" if prediction[0] == 1 else "Not Spam"
            st.write(f"The email is classified as: {result}")
else:
    st.error("Could not load the dataset. Please check the file path and content.")



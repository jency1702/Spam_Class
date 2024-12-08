## Spam Email Classification using NLP and Machine Learning


This project classifies emails as **Spam** or **Not Spam** using Natural Language Processing (NLP) and Machine Learning. The classification model uses **Multinomial Naive Bayes** and **TF-IDF Vectorization** to transform email content into numerical data for prediction.

## Features

- **Email Text Preprocessing**: Email content is processed using `TfidfVectorizer` to convert text into numerical form.
- **Naive Bayes Classifier**: The model uses the Multinomial Naive Bayes algorithm for spam classification.
- **Model Evaluation**: The performance is evaluated using accuracy, confusion matrix, and classification report.
- **Interactive User Interface**: An interactive web interface built with **Streamlit** allows users to input custom email content and receive a spam classification.

## Dataset

The dataset consists of two columns:
- **text**: The email content.
- **spam**: A binary label indicating if the email is spam (`1`) or not (`0`).

Ensure the dataset (`emails.csv`) is placed in the project directory.

## Requirements

- Python 3.x
- Streamlit
- Scikit-learn
- Pandas

You can install the required libraries using the following command:

```bash
pip install -r requirements.txt
```

### `requirements.txt`
```txt
streamlit
pandas
scikit-learn
```

## Project Workflow

1. **Data Loading**: The data is loaded from a CSV file and preprocessed to remove missing values.
2. **Text Vectorization**: Email content is transformed into numerical vectors using `TfidfVectorizer`.
3. **Model Training**: A **Multinomial Naive Bayes** model is trained on the transformed data.
4. **Model Evaluation**: The model’s performance is evaluated using accuracy, classification report, and confusion matrix.
5. **Interactive Interface**: Users can input email content to predict whether the email is **Spam** or **Not Spam**.

## How to Run the Project

### 1. Clone the repository
```bash
git clone https://github.com/your-username/spam-email-classification.git
cd spam-email-classification
```

### 2. Install dependencies
Make sure you have Python installed, then run:
```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit app
To run the application, use:
```bash
streamlit run spam.py
```

This will start the app in your default web browser.

## How to Use the Application

- Once the application starts, you will see performance metrics such as accuracy, confusion matrix, and classification report.
- Enter email text in the input field and click **"Predict"** to classify the email as **Spam** or **Not Spam**.

Example Input:
```text
Congratulations! You have won a free vacation. Click here to claim your prize.
```
The application will classify this email as **Spam**.

## Model Performance

- **Accuracy**: The model achieves an accuracy of over 90% on the test set.
- **Confusion Matrix**: Visualizes the true positive, false positive, true negative, and false negative counts.
- **Classification Report**: Provides precision, recall, and F1 score for both classes (spam and not spam).

## Future Improvements

- Implement other machine learning algorithms such as **SVM** or **Random Forest** for better performance.
- Improve text preprocessing by adding techniques like stemming, lemmatization, and using n-grams.
- Implement email scraping functionality to classify live emails from a user's inbox.


## Contributing

Contributions are welcome! Feel free to fork the repository, make improvements, and submit pull requests. If you find any bugs, open an issue, and I'll get back to you.


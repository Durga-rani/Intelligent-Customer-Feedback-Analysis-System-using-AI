

import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.utils import shuffle

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

def simulate_feedback_data(n=1000):
    feedback = [
        "I love the product! Great service.",
        "Terrible experience. Will never buy again!",
        "Okay-ish. Could be better.",
        "Support team was helpful but slow.",
        "Amazing quality and fast delivery!",
        "Worst packaging ever. Item damaged.",
        "Neutral. Nothing special to mention.",
        "Very satisfied with the purchase.",
        "Not happy. Expected more.",
        "Good value for money."
    ]
    data = pd.DataFrame({
        'feedback_id': range(1, n+1),
        'feedback_text': np.random.choice(feedback, n)
    })
    return data


def clean_text(text):
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)  # Remove URLs
    text = re.sub(r'\@w+|\#','', text)  # Remove mentions and hashtags
    text = re.sub(r'[^A-Za-z\s]', '', text)  # Remove special characters
    text = text.lower()  # Lowercase
    return text


def preprocess_data(df):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))

    def process(text):
        text = clean_text(text)
        tokens = nltk.word_tokenize(text)
        tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words]
        return ' '.join(tokens)

    df['cleaned_text'] = df['feedback_text'].astype(str).apply(process)
    df.drop_duplicates(subset='cleaned_text', inplace=True)
    df.dropna(subset=['cleaned_text'], inplace=True)
    df = shuffle(df).reset_index(drop=True)
    return df


if __name__ == "__main__":
    raw_data = simulate_feedback_data(1000)
    cleaned_data = preprocess_data(raw_data)
    cleaned_data.to_csv("cleaned_feedback.csv", index=False)
    print("✅ Cleaned dataset saved as 'cleaned_feedback.csv'")

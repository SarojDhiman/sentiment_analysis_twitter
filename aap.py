import streamlit as st
import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from torch.nn.functional import softmax
from new_model import predict_sentiment

# Load pre-trained model and tokenizer (outside the app function for efficiency)
roberta = "cardiffnlp/twitter-roberta-base-sentiment"
model = AutoModelForSequenceClassification.from_pretrained(roberta)
tokenizer = AutoTokenizer.from_pretrained(roberta)

def analyze_sentiment(df):
    results = []
    for index, row in df.iterrows():
        text = row['tweet']  # Assuming the column name is 'tweet'

        # Call the predict_sentiment function from model.py
        predicted_sentiment, predicted_emoji, probs = predict_sentiment(text, model, tokenizer)

        results.append({
            "Text": text,
            "Predicted Sentiment": predicted_sentiment,
            "Emoji": predicted_emoji,
            "Probabilities": probs.tolist()  # Convert probs to list for DataFrame compatibility
        })

    return pd.DataFrame(results)

# Streamlit app
st.title("Sentiment Analysis App")

uploaded_file = st.file_uploader("Upload a CSV file", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    results = analyze_sentiment(df)

    st.write("Sentiment Analysis Results:")
    st.dataframe(results)


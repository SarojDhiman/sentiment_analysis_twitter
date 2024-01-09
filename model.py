from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from torch.nn.functional import softmax
import emoji
import pandas as pd

# Load pre-trained RoBERTa model and tokenizer
# model_name = 'ardiffnlp/twitter-roberta-base-sentiment'
# tokenizer = RobertaTokenizer.from_pretrained(model_name)
# model = RobertaForSequenceClassification.from_pretrained(model_name)

roberta = "cardiffnlp/twitter-roberta-base-sentiment"

model = AutoModelForSequenceClassification.from_pretrained(roberta)
tokenizer = AutoTokenizer.from_pretrained(roberta)

# Example CSV file with a 'tweet' column
csv_file_path = 'hindus_section.csv'
df = pd.read_csv(csv_file_path)

# Process each text entry in the CSV file
for index, row in df.iterrows():
    # Example text with emojis
    text = row['tweet']

    # Tokenize and encode the text
    inputs = tokenizer(text, return_tensors='pt', truncation=True)

    # Forward pass through the model
    outputs = model(**inputs)

    # Get probabilities for each class (e.g., positive, negative)
    probs = softmax(outputs.logits, dim=1).detach().numpy()[0]

    # Interpret results
    sentiment_labels = ['Negative', 'Neutral', 'Positive']
    predicted_sentiment = sentiment_labels[probs.argmax()]

    # Map sentiment labels to emojis
    emoji_mapping = {
        'Negative': '😠',
        'Neutral': '😐',
        'Positive': '😊'
    }

    # Get the corresponding emoji for the predicted sentiment
    predicted_emoji = emoji_mapping.get(predicted_sentiment, '')

    print(f"Text: {text}")
    print(f"Predicted Sentiment: {predicted_sentiment} {predicted_emoji}")
    print(f"Sentiment Probabilities: {probs}")
    print("\n")

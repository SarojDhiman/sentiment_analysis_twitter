# necessary packages
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer, AutoConfig
from scipy.special import softmax

# read the data from a CSV file
df = pd.read_csv("hindus_section.csv")

# since we are only interested in the content of the tweets, we will select it
df = df[["tweet"]]

# twitter-roberta-base-sentiment model works better with minimalistic amount of manipulation to the data
# just let the model know that there is username http links
def preprocess(text):
    new_text = []
    for t in text.split(" "):
        t = "@user" if t.startswith("@") and len(t) > 1 else t
        t = "http" if t.startswith("http") else t
        new_text.append(t)
    return " ".join(new_text)

# apply the preprocessing functions to the content
df["roberta_content"] = df["tweet"].apply(preprocess)

# model, tokenizer, and so on are located in the following repo
model_path = f"cardiffnlp/twitter-roberta-base-sentiment-latest"
tokenizer = AutoTokenizer.from_pretrained(model_path)
config = AutoConfig.from_pretrained(model_path)
roberta_model = AutoModelForSequenceClassification.from_pretrained(model_path)

# a function that takes text and model to calculate the probability of each sentiment
def sentiment_analyzer(text, model):
    encoded_input = tokenizer(text, return_tensors="pt")
    output = model(**encoded_input)
    scores = output.logits[0].detach().numpy()
    scores = np.round(softmax(scores), 2)
    scores_dict = {"neg": scores[0], "neu": scores[1], "pos": scores[2]}
    return scores_dict

# apply the roberta function
df["probabilities"] = df["roberta_content"].apply(sentiment_analyzer, model=roberta_model)

# since roberta model returned probability of each sentiment as a dictionary
# let's convert each probability into a separate column
probabilities = df["probabilities"].apply(pd.Series)
df = df.join(probabilities)
df = df.drop("probabilities", axis=1)

# now calculate the polarity for each text by:
# first multiplying each probability to its weights (-1=> negative, 0=> neutral and +1=> positive)
# then sum the values and pass through Tanh function to scale values from -1 up to +1
# finally, we can assign labels for each text, depending on the polarity, e.g., -1.0 until -0.25 negative
polarity_weights = torch.tensor([-1, 0, 1])
probs = torch.tensor(df[["neg", "neu", "pos"]].values)
polarity = polarity_weights * probs
polarity = polarity.sum(dim=-1)
polarity_scaled = nn.Tanh()(polarity)
df["roberta_polarity"] = polarity_scaled.numpy()
df["roberta_sentiment"] = pd.cut(df["roberta_polarity"],
    bins=[-1.0, -0.25, 0.25, 1.0], labels=["Negative", "Neutral", "Positive"])
df = df.drop(["neu", "neg", "pos"], axis=1)
# Save the DataFrame to a new CSV file
df.to_csv("output_file.csv", index=False)

print(df.head())

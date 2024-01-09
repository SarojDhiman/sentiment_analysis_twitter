
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# read the data from a CSV file
df = pd.read_csv("hindus_section.csv")

# since we are only interested in the content of the tweets, we will select it
df = df[["tweet"]]

# preprocess data
# remove mentions
df["vader_content"] = df["tweet"].replace(r"@[A-Za-z0-9_]+", "", regex=True)
# remove URLs
df["vader_content"] = df["vader_content"].replace(r"http\S+|www\.\S+", "", regex=True)
# remove hashtag symbols (we will not remove complete hashtags, since people use part of sentence as a hashtags, e.g. Economy got worsen since the #pandemic started)
df["vader_content"] = df["vader_content"].replace(r"#", "", regex=True)

# Initialize the VADER sentiment analyzer
vader_model = SentimentIntensityAnalyzer()
# apply VADER model to the dataset
df["vader_scores"] = df["vader_content"].apply(
    lambda text: vader_model.polarity_scores(text)
)

# VADER returned scores as a dictionary.
# however, it already has a compound score, which is kind of scaled score from -1 up to +1, so we don't need to compute it as we did for Roberta using Tanh function
# finally, we can assign labels for each text as we did for Roberta model, depending on the polarity, e.g., -1.0 until -0.25 negative
df["vader_polarity"] = df["vader_scores"].apply(
    lambda score_dict: score_dict["compound"]
)

df["vader_sentiment"] = pd.cut(
    df["vader_polarity"],
    bins=[-1.0, -0.25, 0.25, 1.0],
    labels=["Negative", "Neutral", "Positive"],
)
df = df.drop(["content", "vader_content", "vader_scores"], axis=1)
print(df.head())

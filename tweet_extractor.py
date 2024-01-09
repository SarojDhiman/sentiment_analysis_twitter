import pandas as pd
import re

# Load data
df = pd.read_csv("tweet_data.csv")

# Preprocess text function
def preprocess_text(text):
    text = re.sub(r"http\S+", "", text)  # Remove URLs
    text = re.sub(r"#\S+", "", text)  # Remove hashtags
    text = re.sub(r"@\S+", "", text)  # Remove usernames
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)  # Remove punctuation and other characters
    text = text.lower()  # Convert to lowercase
    return text

# Categorize tweets function (rule-based example)
def categorize_tweet(text):
    if any(word in text for word in ["movie", "film", "actor", "cinema"]):
        return "movies"
    elif any(word in text for word in ["hindu", "hindus", "hinduism"]):
        return "hindus"
    elif any(word in text for word in ["sports", "athlete", "game"]):
        return "sports"
    elif any(word in text for word in ["technology", "innovation", "gadget"]):
        return "technology"
    else:
        return "other"

# Apply preprocessing and categorization
df["tweet"] = df["tweet"].apply(preprocess_text)
df["category"] = df["tweet"].apply(categorize_tweet)

# Organize queries into sections
movies_section = df[df["category"] == "movies"]["tweet"]
hindus_section = df[df["category"] == "hindus"]["tweet"]
sports_section = df[df["category"] == "sports"]["tweet"]
technology_section = df[df["category"] == "technology"]["tweet"]
other_section = df[df["category"] == "other"]["tweet"]

# Save the sections to separate CSV files
movies_section.to_csv("movies_section.csv", index=False)
hindus_section.to_csv("hindus_section.csv", index=False)
sports_section.to_csv("sports_section.csv", index=False)
technology_section.to_csv("technology_section.csv", index=False)
other_section.to_csv("other_section.csv", index=False)

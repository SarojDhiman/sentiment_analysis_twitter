# from googletrans import Translator
# import pandas as pd
# import time

# from googletrans import Translator
# import time

# # Function to translate a text to English with timeout and retry
# def translate_to_english(text, max_retries=3):
#     retries = 0
#     while retries < max_retries:
#         try:
#             translator = Translator(timeout=5)  # Set timeout to 5 seconds (you can adjust as needed)
#             translated_text = translator.translate(text, dest='en').text
#             return translated_text
#         except Exception as e:
#             if 'int' in str(e) and 'as_dict' in str(e):
#                 print(f"Error translating text (skipping): {text}. Error: {e}")
#                 return text  # Return the original text
#             print(f"Error translating text: {text}. Error: {e}")
#             retries += 1
#             time.sleep(1)  # Wait for 1 second before retrying

#     # If max_retries is reached, return the original text
#     print(f"Max retries reached for text: {text}")
#     return text


# # Read CSV file
# df = pd.read_csv('train_multiple.csv')

# # Assuming the column 'text' contains the queries in different languages
# df['english_queries'] = df['text'].apply(translate_to_english)

# # Save the translated queries to a new CSV file
# df.to_csv('translated_queries.csv', index=False)


from mtranslate import translate
import pandas as pd

# Function to translate a text to English
def translate_to_english(text):
    try:
        translated_text = translate(text, 'en')
        return translated_text
    except Exception as e:
        print(f"Error translating text: {text}. Error: {e}")
        return text

# Read CSV file
df = pd.read_csv('train_multiple.csv')

# Assuming the column 'text' contains the queries in different languages
df['english_queries'] = df['text'].apply(translate_to_english)

# Save the translated queries to a new CSV file
df.to_csv('translated_queries.csv', index=False)

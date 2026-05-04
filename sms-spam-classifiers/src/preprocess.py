import pandas as pd
import string

def load_data(filepath):
    df = pd.read_csv(filepath, sep='\t', header=None, names=['label', 'message'])
    return df

def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

def preprocess_data(df):
    df['cleaned'] = df['message'].apply(clean_text)
    return df
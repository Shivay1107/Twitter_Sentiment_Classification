# Data Cleaning and Preprocessing Module
import re
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder 

class CleanPreprocess:
    def __init__(self, datafame):
        self.datafame = datafame
        self.label_encoder = LabelEncoder()

    # Text Cleaning Function
    def clean_text(self, text):
        text = text.lower()
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'\@\w+|\#','', text)
        text = re.sub(r'[^A-Za-z0-9\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def remove_urls(self, text):
        return re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE) 
    
    def remove_special_characters(self, text):
        return re.sub(r'[^A-Za-z0-9\s]', '', text)
    
    def remove_extra_whitespace(self, text):
        return re.sub(r'\s+', ' ', text).strip()
    
    def remove_mentions(self, text):
        return re.sub(r'\@\w+','', text)
    
    def remove_hashtags(self, text):
        return re.sub(r'\#','', text)
    
    def remove_emojis(self, text):
        emoji_pattern = re.compile("["
                                   u"\U0001F600-\U0001F64F"  # emoticons
                                   u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                   u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                   u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                   "]+", flags=re.UNICODE)
        return emoji_pattern.sub(r'', text)
    
    
    

    def preprocess(self,col):
        self.datafame['cleaned_text'] = self.datafame[col].apply(self.clean_text)
        # self.datafame['label_encoded'] = self.label_encoder.fit_transform(self.datafame['label'])
        return self.datafame
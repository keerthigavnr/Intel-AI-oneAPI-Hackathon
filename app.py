from flask import Flask, render_template, request
import pickle
import pandas as pd
import numpy as np
import os
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from scipy.sparse import csr_matrix
from flask import Flask, render_template, request, redirect, url_for

# Uncomment the following lines if you haven't downloaded stopwords and wordnet
# nltk.download('stopwords')
# nltk.download('wordnet')

abuse_info = {
    'physical': {
        'description': 'Physical abuse involves intentional use of force against the victim causing injury.',
        'links': [
            {'text': 'National Domestic Violence Hotline', 'url': 'https://www.thehotline.org/'},
            {'text': 'Help Guide - Domestic Violence and Abuse', 'url': 'https://www.helpguide.org/articles/abuse/domestic-violence-and-abuse.htm'}
        ]
    },
    'emotional': {
        'description': 'Emotional abuse involves undermining a person’s sense of self-worth through manipulation.',
        'links': [
            {'text': 'Psychology Today - Understanding Emotional Abuse', 'url': 'https://www.psychologytoday.com/us/basics/emotional-abuse'},
            {'text': 'Verywell Mind - How to Recognize the Signs of Mental and Emotional Abuse', 'url': 'https://www.verywellmind.com/identify-and-cope-with-emotional-abuse-4156673'}
        ]
    },
    # Add other abuse types here with their respective descriptions and links
}

stop_words = set(stopwords.words('english'))
lemma = WordNetLemmatizer()

app = Flask(__name__)

# Load the trained TfidfVectorizer
# with open('tfidf_vectorizer.pkl', 'rb') as tfidf_vectorizer_file:
#     tfidf_vectorizer = pickle.load(tfidf_vectorizer_file)

with open('log_reg_model.pkl', 'rb') as log_reg_model_file:
    log_reg_model = pickle.load(log_reg_model_file)


# LE = LabelEncoder()

with open('label_encoder.pkl', 'rb') as label_encoder_file:
    LE = pickle.load(label_encoder_file)

def text_preprocessing(text):
    text = str(text)
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)
    text = re.sub('<.*?>', ' ', text)
    return text

def drop_stopwords(text):
    dropped = [word for word in text.split() if word not in stop_words]
    tokens = [lemma.lemmatize(word) for word in dropped]
    final_text = ' '.join(tokens)
    return final_text

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        user_input = request.form['user_input']

        user_input_processed = [text_preprocessing(user_input)]

        user_input_processed = [drop_stopwords(text) for text in user_input_processed]
        print("Processed Input:", user_input_processed)

        prediction= log_reg_model.predict(user_input_processed)
        print("Numeric Predictions:", prediction)

        predicted_class = str(LE.inverse_transform(prediction)[0])

        abuse_details = abuse_info.get(predicted_class, {'description': 'No details available.', 'links': []})
        if predicted_class == 'physical':
            return redirect(url_for('physical'))
        elif predicted_class == 'emotional':
            return redirect(url_for('emotional'))
        elif predicted_class == 'sexual':
            return redirect(url_for('sexual'))
        elif predicted_class == 'verbal':
            return redirect(url_for('verbal'))
        
        elif predicted_class == 'stalking':
            return redirect(url_for('stalkingn'))
        
        
        return render_template('result.html', prediction=predicted_class, abuse_details=abuse_details)
@app.route('/physical')
def physical():
    return render_template('physical.html')

@app.route('/emotional')
def emotional():
    return render_template('emotional.html') 
@app.route('/verbal')
def verbal():
    return render_template('verbal.html')  
 
@app.route('/sexual')
def sexual():
    return render_template('sexual.html')  
@app.route('/stalking')
def stalking():
    return render_template('stalking.html')  

if __name__ == '__main__':
    app.run(debug=True)


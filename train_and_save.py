# train_and_save.py
import pandas as pd, re, pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import nltk
nltk.download('stopwords')

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()
    stop_words = set(stopwords.words('english'))
    stemmer = SnowballStemmer('english')
    tokens = [stemmer.stem(w) for w in tokens if w not in stop_words]
    return ' '.join(tokens)

df = pd.read_csv('emails.csv')
df['clean_text'] = df['text'].apply(clean_text)

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['clean_text'])
y = df['spam']

model = MultinomialNB().fit(X, y)

with open('vectorizer.pkl', 'wb') as f: pickle.dump(vectorizer, f)
with open('model.pkl', 'wb') as f: pickle.dump(model, f)
print("Saved: vectorizer.pkl, model.pkl")

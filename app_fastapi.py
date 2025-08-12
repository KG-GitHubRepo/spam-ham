# app_fastapi.py
import re, pickle
from fastapi import FastAPI
from pydantic import BaseModel
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import nltk
import uvicorn

# Ensure stopwords exist on first run (safe to keep)
try:
    _ = stopwords.words('english')
except:
    nltk.download('stopwords')

app = FastAPI(title="Spam Detector API")

with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

stop_words = set(stopwords.words('english'))
stemmer = SnowballStemmer('english')

def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = [stemmer.stem(w) for w in text.split() if w not in stop_words]
    return ' '.join(tokens)

class EmailIn(BaseModel):
    text: str

class PredictionOut(BaseModel):
    spam: bool
    label: str

@app.get("/")
def root():
    return {"message": "Spam Detector API. POST /predict {text: ...}"}

@app.post("/predict", response_model=PredictionOut)
def predict(payload: EmailIn):
    cleaned = clean_text(payload.text)
    X = vectorizer.transform([cleaned])
    pred = int(model.predict(X)[0])
    return {"spam": bool(pred), "label": "Spam" if pred == 1 else "Not Spam"}

if __name__ == "__main__":
    uvicorn.run("app_fastapi:app", host="0.0.0.0", port=8000)

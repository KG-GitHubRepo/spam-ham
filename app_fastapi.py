# app_fastapi.py
from fastapi.responses import HTMLResponse

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


@app.get("/app", response_class=HTMLResponse)
def app_page():
    return """
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Email Spam Detector (Naive Bayes)</title>
<style>
  :root { --bg:#f7f7fb; --card:#ffffff; --text:#1f2937; --muted:#6b7280; --accent:#6d28d9; --accent2:#9333ea; --success:#16a34a; --danger:#dc2626; }
  body{margin:0;background:linear-gradient(180deg,#faf9ff,#f4f4ff 40%,#f7f7fb);color:var(--text);font-family:ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,Arial}
  .wrap{max-width:900px;margin:48px auto;padding:0 16px}
  .card{background:var(--card);border-radius:16px;box-shadow:0 10px 30px rgba(0,0,0,.06);padding:28px}
  .title{font-size:34px;font-weight:800;margin:0 0 6px;display:flex;gap:12px;align-items:center}
  .sub{color:var(--muted);margin:8px 0 22px}
  label{display:flex;align-items:center;gap:8px;font-weight:600;margin:12px 0 8px}
  textarea{width:100%;min-height:160px;padding:16px 18px;border:1px solid #e5e7eb;border-radius:12px;font-size:16px;background:#fbfbff;resize:vertical;outline:none}
  textarea:focus{border-color:#c4b5fd;box-shadow:0 0 0 4px rgba(124,58,237,.08)}
  .btn{display:inline-flex;gap:8px;border:0;padding:10px 16px;border-radius:12px;font-weight:700;cursor:pointer;background:linear-gradient(135deg,var(--accent),var(--accent2));color:#fff;box-shadow:0 8px 18px rgba(109,40,217,.28)}
  .badge{display:inline-flex;align-items:center;gap:10px;padding:10px 14px;border-radius:10px;font-weight:700}
  .ok{background:#ecfdf5;color:var(--success);border:1px solid #bbf7d0}
  .bad{background:#fef2f2;color:var(--danger);border:1px solid #fecaca}
</style>
</head>
<body>
  <div class="wrap">
    <div class="card">
      <h1 class="title">📧 Email Spam Detector (Naive Bayes)</h1>
      <p class="sub">Enter email text below and click Predict:</p>

      <label>✏️ Your email:</label>
      <textarea id="txt" placeholder="Hi. Let me know if you need any other information. Thanks"></textarea>

      <div style="margin-top:14px">
        <button class="btn" onclick="predict()">🚀 Predict</button>
      </div>

      <div class="result" style="margin-top:22px">
        <h3>Result:</h3>
        <div id="out" class="badge" style="display:none"></div>
      </div>
    </div>
  </div>

<script>
async function predict(){
  const el = document.getElementById('out');
  el.style.display = 'inline-flex';
  el.className = 'badge'; el.textContent = 'Predicting...';
  try{
    const text = document.getElementById('txt').value || '';
    const res = await fetch('/predict', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({text})
    });
    const data = await res.json();
    el.className = 'badge ' + (data.spam ? 'bad' : 'ok');
    el.textContent = data.spam ? '🚫 Spam' : '✅ Not Spam';
  }catch(err){
    el.className = 'badge bad';
    el.textContent = 'Error: ' + err.message;
  }
}
</script>
</body>
</html>
"""


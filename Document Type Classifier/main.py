from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import joblib
import os

app = FastAPI()

model = joblib.load("document_classifier.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("form.html", {"request": request, "prediction": None})

@app.post("/predict", response_class=HTMLResponse)
def predict(request: Request, text: str = Form(...)):
    try:
        X_input = vectorizer.transform([text])
        prediction = model.predict(X_input)[0]
        return templates.TemplateResponse("form.html", {"request": request, "prediction": prediction, "text": text})
    except Exception as e:
        return templates.TemplateResponse("form.html", {
            "request": request,
            "prediction": f"Error: {str(e)}",
            "text": text
        })

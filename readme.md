# Classifier Model
## Model Aim

Train a simple machine learning model to classify document types like Invoice, Resume, or Report using text content.

## Dataset

Used synthetic data for demonstration. You can replace it with an open-source dataset like:
- [Document Classification Dataset on Kaggle](https://www.kaggle.com/)

## Preprocessing

- Removed punctuation, digits, and URLs
- Converted text to lowercase

##  Model

- TF-IDF vectorizer + Multinomial Naive Bayes
- Trained using `scikit-learn` pipeline

## Create a API Using fastAPI

- FastAPI backend with prediction and HTML form endpoints
- HTML + CSS UI to classify text documents
- Swagger UI available at /docs

## Running the App

# Train the model
- python create_model.py

# Start the FastAPI server
- uvicorn main:app --reload

- Visit: http://127.0.0.1:8000 
- Docs: http://127.0.0.1:8000/docs

## Results

Example classification report with synthetic data:
- Accuracy ~90%
- Performs well on distinct keywords

## 📦 Requirements

```bash
pip install pandas scikit-learn numpy

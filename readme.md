# Author: @Gautam Kumar

# Email Categorization 

#### Aim
- This project processes raw email text files which i have download from (https://www.cs.cmu.edu/~enron/), extracts relevant metadata (from, subject, body), cleans the email content, and classifies them into predefined categories using simple keyword-based matching below are the list for key words :
- "Market/Finance": ["chart", "matrix", "api", "crude", "stocks", "transactions", "amount", "finance"],
- "HR/Admin": ["calendar", "ranked", "play", "camp", "paperwork", "update", "Y", "schedule"],
- "Personal": ["love", "hello", "thankful", "sorry", "greg", "personal"],
- "Technical": ["delivery point", "website", "server", "ECA", "records", "data"]

- JSON Output: Exports cleaned and categorized emails into output_file.json

#### How to run script?
- python email_classifier.py and you will get a file "output_file.json"



# Classifier Model
#### Model Aim

Train a simple machine learning model to classify document types like Invoice, Resume, or Report using text content.

#### Dataset

Used synthetic data for demonstration. You can replace it with an open-source dataset like:
- [Document Classification Dataset on Kaggle](https://www.kaggle.com/)

#### Preprocessing

- Removed punctuation, digits, and URLs
- Converted text to lowercase

####  Model

- TF-IDF vectorizer + Multinomial Naive Bayes
- Trained using `scikit-learn` pipeline

#### Create a API Using fastAPI

- FastAPI backend with prediction and HTML form endpoints
- HTML + CSS UI to classify text documents
- Swagger UI available at /docs

#### Running the App

#### Train the model
- python create_model.py

#### Start the FastAPI server
- uvicorn main:app --reload

- Visit: http://127.0.0.1:8000 
- Docs: http://127.0.0.1:8000/docs

#### Results

Example classification report with synthetic data:
- Accuracy ~90%
- Performs well on distinct keywords

#### Requirements

- Create a virtual environments using below commands
- mkdir project_name
- cd project_name
- python -m venv .gbm
- source .gbm/bin/activate

- then run below commands
- bash pip install -r requirements.txt


## Thank you


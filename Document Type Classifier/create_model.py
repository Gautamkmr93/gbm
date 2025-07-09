# Author: @Gautam Kumar
import pandas as pd
import numpy as np
import logging
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score,confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt
from sklearn.base import ClassifierMixin
from typing import Tuple, List, Union, Dict
import os

logging.basicConfig(filename="machine_learning_model.log", filemode="a",format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)

def preprocess_labels_and_text(df: pd.DataFrame) -> pd.DataFrame:
    try:
        if '5485' not in df.columns:
            raise KeyError("Required column '5485' is missing.")
        df = df[~df['5485'].isna() & (df['5485'].astype(str).str.len() > 1)].copy()
        df['labels'] = df['5485'].astype(str).str[0].astype(int)
        df['doc_text'] = df['5485'].astype(str).str[1:]
        return df.drop(columns=['5485'])
    except Exception as e:
        logging.exception("Preprocessing failed")
        raise

def prepare_data(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    df_processed = preprocess_labels_and_text(df)
    return df_processed['doc_text'], df_processed['labels']


def find_best_max_features(
    X: List[str],
    y: Union[List[int], np.ndarray],
    max_features_values: List[int],
    cv_folds: int = 5
) -> int:
    results = {}
    try:
        for max_features in max_features_values:
            vectorizer = TfidfVectorizer(stop_words='english', max_features=max_features)
            X_tfidf = vectorizer.fit_transform(X)
            model = LogisticRegression(max_iter=1000)
            scores = cross_val_score(model, X_tfidf, y, cv=cv_folds, scoring='accuracy')
            results[max_features] = np.mean(scores)
        best_max_features = max(results, key=results.get)
        logging.info(f"Best max_features = {best_max_features} with accuracy = {results[best_max_features]:.4f}")
        return best_max_features
    except Exception as e:
        logging.exception("Error in finding best max_features")
        raise

def train_and_evaluate_models(
    X_train, y_train, X_test, y_test,
    models: List[ClassifierMixin], names: List[str]
) -> pd.DataFrame:
    results = []
    for name, model in zip(names, models):
        try:
            model.fit(X_train, y_train)
            joblib.dump(model, 'document_classifier.pkl')
            logging.info(f"{name} model saved.")
            metrics = evaluate_model(model, X_train, y_train, X_test, y_test)
            results.append({"Classifier": name, **metrics})
        except Exception as e:
            logging.exception(f"Training failed for {name}")
    return pd.DataFrame(results)

def evaluate_model(
    model: ClassifierMixin,
    X_train, y_train, X_test, y_test,
    plot_confusion: bool = True
) -> Dict[str, float]:
    y_test_pred = model.predict(X_test)
    metrics = {
        "Accuracy": accuracy_score(y_test, y_test_pred),
        "F1 Score": f1_score(y_test, y_test_pred, average='weighted', zero_division=1),
        "Precision": precision_score(y_test, y_test_pred, average='weighted', zero_division=1),
        "Recall": recall_score(y_test, y_test_pred, average='weighted', zero_division=1)
    }
    if plot_confusion:
        cm = confusion_matrix(y_test, y_test_pred)
        ConfusionMatrixDisplay(cm).plot(cmap='Blues')
        plt.title(f"Confusion Matrix - {type(model).__name__}")
        plt.show()
    logging.info(f"Metrics: {metrics}")
    return metrics


if __name__ == "__main__":
    for dirname, _, filenames in os.walk('/Users/gautamkumar/Desktop/GBM/Document Type Classifier/DataSet'):
        for filename in filenames:
            file_path = os.path.join(dirname, filename)
    df = pd.read_csv(file_path)
    X, y = prepare_data(df)
    y = LabelEncoder().fit_transform(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    max_features_values = [100, 500, 1000, 5000]
    best_max_features = find_best_max_features(X_train, y_train, max_features_values)
    tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=best_max_features)
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    X_test_tfidf = tfidf_vectorizer.transform(X_test)
    joblib.dump(tfidf_vectorizer, 'tfidf_vectorizer.pkl')
    models = [RandomForestClassifier(n_estimators=100, class_weight='balanced')]
    names = ["Random Forest"]
    results = train_and_evaluate_models(X_train_tfidf, y_train, X_test_tfidf, y_test, models, names)
    print(results)

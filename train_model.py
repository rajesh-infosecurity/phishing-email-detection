"""Train and evaluate a baseline phishing-email text classifier."""
from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    ConfusionMatrixDisplay
)

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "emails.csv"
MODEL_PATH = ROOT / "model.joblib"
REPORT_PATH = ROOT / "evaluation_report.json"
MATRIX_PATH = ROOT / "confusion_matrix.png"

def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    required = {"text", "label"}
    if not required.issubset(df.columns):
        raise ValueError("CSV must contain columns named 'text' and 'label'.")

    df = df.dropna(subset=["text", "label"]).copy()
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"] != ""]
    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)

    if set(df["label"].unique()) != {0, 1}:
        raise ValueError("Dataset must contain both labels: 0=legitimate, 1=phishing.")
    if df["label"].value_counts().min() < 2:
        raise ValueError("At least two examples per class are required for a stratified split.")

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"], test_size=0.25, random_state=42, stratify=df["label"]
    )

    model = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), lowercase=True, strip_accents="unicode")),
        ("classifier", MultinomialNB())
    ])
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    report = classification_report(
        y_test, predictions, labels=[0, 1],
        target_names=["legitimate", "phishing"], output_dict=True, zero_division=0
    )
    metrics = {
        "dataset_rows": int(len(df)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "accuracy": float(accuracy_score(y_test, predictions)),
        "classification_report": report,
        "label_mapping": {"0": "legitimate", "1": "phishing"},
        "warning": "Small/demo datasets do not establish real-world security performance."
    }

    joblib.dump(model, MODEL_PATH)
    REPORT_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    matrix = confusion_matrix(y_test, predictions, labels=[0, 1])
    display = ConfusionMatrixDisplay(matrix, display_labels=["Legitimate", "Phishing"])
    display.plot(values_format="d")
    plt.title("Phishing Email Detection - Confusion Matrix")
    plt.tight_layout()
    plt.savefig(MATRIX_PATH, dpi=160)
    plt.close()

    print(f"Dataset rows: {len(df)} | Train: {len(X_train)} | Test: {len(X_test)}")
    print(f"Accuracy: {metrics['accuracy']:.3f}")
    print("\nClassification report:")
    print(classification_report(y_test, predictions, labels=[0, 1],
          target_names=["legitimate", "phishing"], zero_division=0))
    print(f"Saved model: {MODEL_PATH.name}")
    print(f"Saved metrics: {REPORT_PATH.name}")
    print(f"Saved confusion matrix: {MATRIX_PATH.name}")
    print("\nNote: metrics are only meaningful when the dataset is large and representative.")

if __name__ == "__main__":
    main()

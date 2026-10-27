"""Interactive phishing-email classifier demo.

This is an educational baseline, not a production email security gateway.
Train the model first with: python train_model.py
"""
from pathlib import Path
import joblib

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "model.joblib"

def main():
    if not MODEL_PATH.exists():
        print("Model not found. First run: python train_model.py")
        return

    model = joblib.load(MODEL_PATH)
    print("\nPhishing Email Detection (educational demo)")
    print("Paste one email message. Enter a blank line to exit.\n")

    while True:
        try:
            message = input("Email text> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not message:
            print("Exiting.")
            break

        prediction = model.predict([message])[0]
        probabilities = model.predict_proba([message])[0]
        classes = list(model.classes_)
        confidence = float(probabilities[classes.index(prediction)]) * 100

        label = "PHISHING / SUSPICIOUS" if int(prediction) == 1 else "LIKELY LEGITIMATE"
        print(f"Prediction: {label}")
        print(f"Model confidence: {confidence:.1f}%")
        print("Reminder: this is a model estimate, not proof. Verify links and sender identity manually.\n")

if __name__ == "__main__":
    main()

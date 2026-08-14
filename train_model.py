"""
Trains the exercise classifier from every CSV in data/raw/ (produced by
collect_data.py) and saves the model to models/exercise_classifier.pkl.

Usage:
    python train_model.py
"""
import glob
import os
import pickle

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score


def load_dataset(data_dir="data/raw"):
    csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}. Run collect_data.py first.")

    frames = [pd.read_csv(f) for f in csv_files]
    df = pd.concat(frames, ignore_index=True)
    return df


def main():
    df = load_dataset()
    print(f"Loaded {len(df)} samples across labels: {df['label'].unique().tolist()}")

    X = df.drop(columns=["label"])
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
    print(classification_report(y_test, y_pred))

    os.makedirs("models", exist_ok=True)
    out_path = "models/exercise_classifier.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(model, f)
    print(f"Model saved to {out_path}")


if __name__ == "__main__":
    main()

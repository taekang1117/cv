# train_model.py — Train RandomForest for Bolt vs Screw
# Run: python3 train_model.py
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

DATA_FILE  = "training_data.csv"
MODEL_FILE = "bolt_screw_model.pkl"

def main():
    print(f"Loading data from {DATA_FILE}...")
    try:
        df = pd.read_csv(DATA_FILE)
    except FileNotFoundError:
        print(f"Error: {DATA_FILE} not found. Run collect_data.py first!")
        return

    if df.empty:
        print("Dataset is empty.")
        return

    required_cols = ['area', 'aspect_ratio', 'circularity', 'solidity', 'perimeter', 'label']
    if not all(col in df.columns for col in required_cols):
        print(f"Error: Dataset missing columns. Required: {required_cols}")
        print(f"Found: {list(df.columns)}")
        return

    X = df[['area', 'aspect_ratio', 'circularity', 'solidity', 'perimeter']]
    y = df['label']   # 1 = Bolt, 0 = Screw

    print(f"Total samples: {len(df)}")
    print(f"Class distribution:\n  Bolt  (1): {(y==1).sum()}\n  Screw (0): {(y==0).sum()}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("\nTraining RandomForestClassifier...")
    clf = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
    clf.fit(X_train, y_train)

    print("Evaluating on test set...")
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=['Screw', 'Bolt']))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nRetraining on full dataset and saving...")
    clf.fit(X, y)
    with open(MODEL_FILE, "wb") as f:
        pickle.dump(clf, f)
    print(f"Model saved to {MODEL_FILE}")

if __name__ == "__main__":
    main()

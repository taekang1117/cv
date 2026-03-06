# train_model.py - Train RandomForest, KNN, and SVM for Bolt vs Screw
# Run: python3 train_model.py
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

DATA_FILE  = "training_data.csv"
RF_MODEL_FILE  = "bolt_nut_model.pkl"
KNN_MODEL_FILE = "bolt_nut_knn.pkl"
SVM_MODEL_FILE = "bolt_nut_svm.pkl"

def evaluate(name, clf, X_test, y_test):
    print(f"\n{'='*50}")
    print(f"Results - {name}")
    print(f"{'='*50}")
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=['Nut', 'Bolt']))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

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
    y = df['label']   # 1 = Bolt, 0 = Nut

    print(f"Total samples: {len(df)}")
    print(f"Class distribution:\n  Bolt  (1): {(y==1).sum()}\n  Nut (0): {(y==0).sum()}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # --- Feature Scaling for KNN and SVM ---
    # Random Forest does not need scaling because it uses decision thresholds.
    # KNN and SVM are distance-based so features must be on the same scale,
    # otherwise 'area' (values in thousands) would dominate 'circularity' (values 0-1).
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    # ----------------------------------------
    # 1. Random Forest
    # ----------------------------------------
    print("\nTraining Random Forest...")
    rf = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
    rf.fit(X_train, y_train)
    evaluate("Random Forest", rf, X_test, y_test)

    # ----------------------------------------
    # 2. K-Nearest Neighbors
    # ----------------------------------------
    print("\nTraining K-Nearest Neighbors (k=5)...")
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train_scaled, y_train)
    evaluate("K-Nearest Neighbors (k=5)", knn, X_test_scaled, y_test)

    # ----------------------------------------
    # 3. Support Vector Machine
    # ----------------------------------------
    print("\nTraining Support Vector Machine...")
    svm = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42)
    svm.fit(X_train_scaled, y_train)
    evaluate("Support Vector Machine (RBF kernel)", svm, X_test_scaled, y_test)
    
    # ----------------------------------------
    # 4. Voting Ensemble (RF + KNN + SVM)
    # ----------------------------------------
    print("\nEvaluating Voting Ensemble...")
    rf_probs  = rf.predict_proba(X_test)
    knn_probs = knn.predict_proba(X_test_scaled)
    svm_probs = svm.predict_proba(X_test_scaled)
    
    avg_probs = (rf_probs + knn_probs + svm_probs) / 3.0
    ensemble_preds = np.argmax(avg_probs, axis=1)
    
    print(f"\n{'='*50}")
    print(f"Results - Voting Ensemble")
    print(f"{'='*50}")
    print(classification_report(y_test, ensemble_preds, target_names=['Nut', 'Bolt']))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, ensemble_preds))

    # ----------------------------------------
    # Save all three models
    # ----------------------------------------
    print("\nRetraining all models on full dataset and saving...")

    rf.fit(X, y)
    with open(RF_MODEL_FILE, "wb") as f:
        pickle.dump(rf, f)
    print(f"Random Forest saved to {RF_MODEL_FILE}")

    X_scaled_full = scaler.fit_transform(X)

    knn.fit(X_scaled_full, y)
    with open(KNN_MODEL_FILE, "wb") as f:
        pickle.dump((knn, scaler), f)
    print(f"KNN saved to {KNN_MODEL_FILE}")

    svm.fit(X_scaled_full, y)
    with open(SVM_MODEL_FILE, "wb") as f:
        pickle.dump((svm, scaler), f)
    print(f"SVM saved to {SVM_MODEL_FILE}")

    print("\nDone. All three models saved.")

if __name__ == "__main__":
    main()

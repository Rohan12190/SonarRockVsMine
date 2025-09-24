import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
import os
import warnings

# Suppress warnings for a cleaner output
warnings.filterwarnings('ignore')

# --- 1. DATA LOADING ---
def load_data(file_path="data/sonar_acoustic_dataset.csv"):
    if not os.path.exists(file_path):
        print(f"Error: Data file not found at '{file_path}'")
        return None, None
    data = pd.read_csv(file_path)
    X = data.iloc[:, :13]
    y = data["Sound Source"]
    print("Data loaded successfully!")
    return X, y

# --- 2. K-VALUE OPTIMIZATION ---
def find_optimal_k(X_train, y_train, X_test, y_test, k_range=range(1, 21)):
    print("\n--- Finding optimal K value for KNN ---")
    best_k = 0
    best_score = 0
    
    for k in k_range:
        knn_pipeline = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=k))
        knn_pipeline.fit(X_train, y_train)
        score = knn_pipeline.score(X_test, y_test)
        if score > best_score:
            best_score = score
            best_k = k
            
    print(f"Optimal K found: {best_k} with accuracy: {best_score:.4f}")
    return best_k

# --- 3. MAIN SCRIPT EXECUTION ---
if __name__ == "__main__":
    results_dir = 'results'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # --- Load and Split Data ---
    X, y = load_data()
    if X is None:
        exit()
        
    # --- Correct split with stratification ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # --- Find Optimal K for KNN ---
    optimal_k = find_optimal_k(X_train, y_train, X_test, y_test)
    
    # --- Define All Models with CORRECTED Hyperparameters ---
    models = {
        'KNN': make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=optimal_k)),
        'SVM': make_pipeline(StandardScaler(), SVC(probability=True, random_state=42)),
        'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
    }

    # --- Evaluate Standalone Models ---
    print("\n--- Evaluating Standalone Model Performance ---")
    standalone_scores = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        score = accuracy_score(y_test, y_pred)
        standalone_scores[name] = score
        print(f"  {name} Accuracy: {score:.4f}")
        
    # --- Evaluate Stacked Models ---
    print("\n--- Evaluating Stacked Model Performance ---")
    stacked_scores = {}
    base_models_for_stacking = [
        ('SVM', models['SVM']),
        ('Random Forest', models['Random Forest']),
        ('Gradient Boosting', models['Gradient Boosting'])
    ]

    for name, model in base_models_for_stacking:
        estimators = [
            ('knn', models['KNN']),
            (name.lower().replace(" ", ""), model)
        ]
        meta_model = LogisticRegression()
        stacked_model = StackingClassifier(estimators=estimators, final_estimator=meta_model, cv=5)
        stacked_model.fit(X_train, y_train)
        y_pred_stacked = stacked_model.predict(X_test)
        stacked_score = accuracy_score(y_test, y_pred_stacked)
        stacked_model_name = f"Stacked (KNN + {name})"
        stacked_scores[stacked_model_name] = stacked_score
        print(f"  {stacked_model_name} Accuracy: {stacked_score:.4f}")

    # --- FINAL RESULTS SUMMARY FOR R ---
    print("\n" + "="*50)
    print("      FINAL SUMMARY FOR R VISUALIZATION")
    print("="*50)
    print("\nStandalone Model Scores:")
    for name, score in standalone_scores.items():
        print(f"{name}: {score:.4f}")
    
    print("\nStacked Model Scores:")
    for name, score in stacked_scores.items():
        print(f"{name}: {score:.4f}")
    print("="*50)


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import os

# 1. DATA LOADING
def load_data(file_path="data/sonar_acoustic_dataset.csv"):
    if not os.path.exists(file_path):
        print(f"Error: Data file not found at '{file_path}'")
        return None, None
    data = pd.read_csv(file_path)
    X = data.iloc[:, :13]
    y = data["Sound Source"]
    print("Data loaded successfully!")
    return X, y

# 2. K-VALUE OPTIMIZATION FUNCTION
def find_optimal_k(X_train, X_test, y_train, y_test, k_range=range(1, 21)):
    """
    Find the optimal K value for KNN by testing different values.
    """
    print("Finding optimal K value for KNN...")
    best_k = 5  # default
    best_score = 0
    k_scores = []
    
    for k in k_range:
        knn_pipeline = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=k))
        knn_pipeline.fit(X_train, y_train)
        score = knn_pipeline.score(X_test, y_test)
        k_scores.append(score)
        
        if score > best_score:
            best_score = score
            best_k = k
    
    print(f"Optimal K: {best_k} with accuracy: {best_score:.4f}")
    
    # Plot K optimization
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, k_scores, marker='o', linestyle='--')
    plt.title('KNN Performance for Different K Values')
    plt.xlabel('Number of Neighbors (K)')
    plt.ylabel('Accuracy')
    plt.xticks(k_range)
    plt.grid(True)
    plt.savefig(f'{results_dir}/k_optimization_rf.png')
    plt.close()
    
    return best_k, best_score

# 3. VISUALIZATION FUNCTION
def plot_confusion_matrix(y_true, y_pred, class_names, model_name, save_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'{model_name} Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig(save_path)
    plt.close()

# 4. MAIN SCRIPT EXECUTION
if __name__ == "__main__":
    results_dir = 'results'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # --- Load and Split Data ---
    X, y = load_data()
    if X is None:
        exit()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # --- Find Optimal K Value ---
    optimal_k, optimal_k_score = find_optimal_k(X_train, X_test, y_train, y_test)

    # --- STANDALONE KNN MODEL (FOR COMPARISON) ---
    print("\n--- Evaluating Standalone KNN Model ---")
    standalone_knn_pipeline = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=optimal_k))
    standalone_knn_pipeline.fit(X_train, y_train)
    y_pred_knn = standalone_knn_pipeline.predict(X_test)
    standalone_accuracy = accuracy_score(y_test, y_pred_knn)
    print("Standalone KNN Performance:")
    print(f"  K Value: {optimal_k}")
    print(f"  Accuracy: {standalone_accuracy:.4f}")
    print(classification_report(y_test, y_pred_knn))

    class_names = ['Source 0', 'Source 1', 'Source 2', 'Source 3']
    plot_confusion_matrix(y_test, y_pred_knn, class_names, "Standalone KNN", f'{results_dir}/standalone_knn_rf_cm.png')

    # --- STACKED MODEL (KNN + RANDOM FOREST) ---
    print("\n--- Evaluating Stacked Model (KNN + Random Forest) ---")
    
    estimators = [
        ('knn', make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=optimal_k))),
        ('rf', RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)) # No scaling needed for RF
    ]
    meta_model = LogisticRegression()
    stacked_model = StackingClassifier(estimators=estimators, final_estimator=meta_model, cv=5)
    stacked_model.fit(X_train, y_train)
    y_pred_stacked = stacked_model.predict(X_test)
    stacked_accuracy = accuracy_score(y_test, y_pred_stacked)
    
    print("Stacked Model Performance:")
    print(f"  Accuracy: {stacked_accuracy:.4f}")
    print(classification_report(y_test, y_pred_stacked))

    plot_confusion_matrix(y_test, y_pred_stacked, class_names, "Stacked (KNN+RF)", f'{results_dir}/stacked_rf_cm.png')
    
    # --- PERFORMANCE COMPARISON ---
    print("\n--- PERFORMANCE COMPARISON ---")
    print(f"Standalone KNN Accuracy: {standalone_accuracy:.4f}")
    print(f"Stacked Model Accuracy:  {stacked_accuracy:.4f}")
    improvement = stacked_accuracy - standalone_accuracy
    print(f"Improvement: {improvement:.4f} ({improvement*100:.2f}%)")
    
    print("\n--- ANALYSIS COMPLETE ---")
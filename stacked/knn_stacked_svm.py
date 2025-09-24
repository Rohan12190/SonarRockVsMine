import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
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
    plt.savefig(f'{results_dir}/k_optimization_svm.png')
    plt.close()
    
    return best_k, best_score

# 3. VISUALIZATION FUNCTION
def plot_confusion_matrix(y_true, y_pred, class_names, model_name, save_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='plasma',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'{model_name} Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig(save_path)
    plt.close()

# 4. INDIVIDUAL MODEL EVALUATION
def evaluate_individual_models(X_train, X_test, y_train, y_test, optimal_k):
    """
    Evaluate individual models before stacking.
    """
    print("--- Evaluating Individual Models ---")
    
    # KNN Model
    knn_pipeline = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=optimal_k))
    knn_pipeline.fit(X_train, y_train)
    knn_score = knn_pipeline.score(X_test, y_test)
    print(f"KNN (K={optimal_k}) Accuracy: {knn_score:.4f}")
    
    # SVM Model
    svm_pipeline = make_pipeline(StandardScaler(), SVC(probability=True, random_state=42))
    svm_pipeline.fit(X_train, y_train)
    svm_score = svm_pipeline.score(X_test, y_test)
    print(f"SVM Accuracy: {svm_score:.4f}")
    
    return knn_score, svm_score

# 5. MAIN SCRIPT EXECUTION
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

    # --- Evaluate Individual Models ---
    knn_individual_score, svm_individual_score = evaluate_individual_models(X_train, X_test, y_train, y_test, optimal_k)

    # --- STANDALONE KNN MODEL (FOR COMPARISON) ---
    print("\n--- Evaluating Standalone KNN Model ---")
    # Create a pipeline to scale data and then apply KNN
    standalone_knn_pipeline = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=optimal_k))
    standalone_knn_pipeline.fit(X_train, y_train)
    y_pred_knn = standalone_knn_pipeline.predict(X_test)
    standalone_accuracy = accuracy_score(y_test, y_pred_knn)
    
    print("Standalone KNN Performance:")
    print(f"  K Value: {optimal_k}")
    print(f"  Accuracy: {standalone_accuracy:.4f}")
    print(classification_report(y_test, y_pred_knn))
    
    class_names = ['Source 0', 'Source 1', 'Source 2', 'Source 3']
    plot_confusion_matrix(y_test, y_pred_knn, class_names, "Standalone KNN", f'{results_dir}/standalone_knn_svm_cm.png')

    # --- STACKED MODEL (KNN + SVM) ---
    print("\n--- Evaluating Stacked Model (KNN + SVM) ---")
    
    # 1. Define the base models with optimal K
    estimators = [
        ('knn', make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=optimal_k))),
        ('svm', make_pipeline(StandardScaler(), SVC(probability=True, random_state=42)))
    ]

    # 2. Define the meta-model that combines the base models' predictions
    meta_model = LogisticRegression()

    # 3. Create the Stacking Classifier
    stacked_model = StackingClassifier(estimators=estimators, final_estimator=meta_model, cv=5)

    # 4. Train and evaluate the stacked model
    stacked_model.fit(X_train, y_train)
    y_pred_stacked = stacked_model.predict(X_test)
    stacked_accuracy = accuracy_score(y_test, y_pred_stacked)
    
    print("Stacked Model Performance:")
    print(f"  Accuracy: {stacked_accuracy:.4f}")
    print(classification_report(y_test, y_pred_stacked))

    plot_confusion_matrix(y_test, y_pred_stacked, class_names, "Stacked Model (KNN+SVM)", f'{results_dir}/stacked_model_svm_cm.png')
    
    # --- COMPREHENSIVE PERFORMANCE COMPARISON ---
    print("\n--- COMPREHENSIVE PERFORMANCE COMPARISON ---")
    print(f"KNN Individual Score:    {knn_individual_score:.4f}")
    print(f"SVM Individual Score:    {svm_individual_score:.4f}")
    print(f"Standalone KNN Accuracy: {standalone_accuracy:.4f}")
    print(f"Stacked Model Accuracy:  {stacked_accuracy:.4f}")
    
    improvement = stacked_accuracy - standalone_accuracy
    print(f"Stacking Improvement: {improvement:.4f} ({improvement*100:.2f}%)")
    
    if stacked_accuracy > max(knn_individual_score, svm_individual_score):
        print("✓ Stacking successfully")
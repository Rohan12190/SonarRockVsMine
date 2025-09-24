import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
import os

#  1. DATA LOADING AND PREPARATION
def load_data(file_path="data/sonar_acoustic_dataset.csv"):
    """
    Loads the sonar data from a CSV file.
    """
    if not os.path.exists(file_path):
        print(f"Error: Data file not found at '{file_path}'")
        return None, None

    data = pd.read_csv(file_path)
    X = data.iloc[:, :13]
    y = data["Sound Source"]

    print("Data loaded successfully!")
    print(f"Dataset has {data.shape[0]} samples and {X.shape[1]} features.")
    return X, y

#  VISUALIZATION FUNCTIONS
def plot_class_distribution(y, save_path):
    """
    Creates and saves a bar chart of the class distribution.
    """
    plt.figure(figsize=(8, 6))
    sns.countplot(x=y.map({0: 'Source 0', 1: 'Source 1', 2: 'Source 2', 3: 'Source 3'}))
    plt.title('Class Distribution', fontsize=16)
    plt.xlabel('Sound Source', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.savefig(save_path)
    plt.close()

def plot_correlation_matrix(X, save_path):
    """
    Creates and saves a heatmap of the feature correlation matrix.
    """
    plt.figure(figsize=(12, 10))
    sns.heatmap(X.corr(), annot=False, cmap='coolwarm')
    plt.title('Feature Correlation Matrix', fontsize=16)
    plt.savefig(save_path)
    plt.close()

def plot_k_optimization(k_values, accuracies, save_path):
    """
    Creates and saves a plot of accuracy vs. K value.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, accuracies, marker='o', linestyle='--')
    plt.title('KNN Performance for Different K Values')
    plt.xlabel('Number of Neighbors (K)')
    plt.ylabel('Accuracy')
    plt.xticks(k_values)
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

def plot_confusion_matrix(y_true, y_pred, class_names, save_path):
    """
    Creates and saves a confusion matrix heatmap.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                  xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig(save_path)
    plt.close()

# REAL-WORLD APPLICATION FUNCTIONS
def predict_sound_source(model, scaler, new_data, save_path=None):
    """
    Simulates real-world prediction on new acoustic data
    """
    new_data_scaled = scaler.transform(new_data)
    prediction = model.predict(new_data_scaled)
    probabilities = model.predict_proba(new_data_scaled)
    
    plt.figure(figsize=(8, 6))
    classes = ['Source 0', 'Source 1', 'Source 2', 'Source 3']
    plt.bar(classes, probabilities[0])
    plt.title('Prediction Confidence for New Sample')
    plt.ylabel('Probability')
    plt.ylim(0, 1)
    for i, prob in enumerate(probabilities[0]):
        plt.text(i, prob + 0.01, f'{prob:.1%}', ha='center')
    
    if save_path:
        plt.savefig(save_path)
    plt.close()
    
    return prediction[0], probabilities[0]

def simulate_real_world_scenario(model, scaler, X_test, images_dir):
    """
    Simulates real-world usage with multiple test samples
    """
    print("\n--- REAL-WORLD SIMULATION ---")
    random_indices = np.random.choice(X_test.index, size=5, replace=False)
    
    for idx, sample_idx in enumerate(random_indices):
        sample = X_test.loc[[sample_idx]]
        save_path = os.path.join(images_dir, f'prediction_sample_{idx+1}.png')
        prediction, probabilities = predict_sound_source(model, scaler, sample, save_path)
        
        print(f"\nSample {idx+1}:")
        print(f"  Predicted: Source {prediction}")
        print(f"  Confidence: {probabilities[prediction]:.1%}")

# REPORTING AND ANALYSIS FUNCTIONS
def generate_report(X, y, best_k, accuracy, precision, recall, report_path):
    """
    Generates a comprehensive performance report and saves it to a file.
    """
    class_dist = y.value_counts().sort_index()
    
    report = f"""SONAR ACOUSTIC CLASSIFICATION REPORT
====================================
Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

DATASET SUMMARY
---------------
- Total samples: {len(X)}
- Number of features: {X.shape[1]}
- Number of classes: {len(y.unique())}

Class Distribution:
{'-' * 20}
"""
    for class_label, count in class_dist.items():
        report += f"Source {class_label}: {count} samples ({count/len(y)*100:.1f}%)\n"
    
    report += f"""
BEST MODEL CONFIGURATION
------------------------
- Algorithm: K-Nearest Neighbors (KNN)
- Optimal K value: {best_k}
- Distance metric: Euclidean (default)
- Weights: Uniform

PERFORMANCE METRICS
-------------------
- Accuracy: {accuracy:.2%}
- Precision (weighted): {precision:.2%}
- Recall (weighted): {recall:.2%}
- F1-Score (estimated): {2 * (precision * recall) / (precision + recall):.2%}
"""
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"\nDetailed report saved to '{report_path}'")

def analyze_errors(y_true, y_pred, X_test, images_dir, reports_dir):
    """
    Performs and saves a detailed analysis of misclassified samples.
    """
    print("\n--- ERROR ANALYSIS ---")
    misclassified = y_true != y_pred
    misclassified_indices = np.where(misclassified)[0]
    
    if len(misclassified_indices) > 0:
        print(f"Total misclassified: {len(misclassified_indices)} out of {len(y_true)} ({len(misclassified_indices)/len(y_true)*100:.1f}%)")
        
        misclass_df = pd.DataFrame({'True': y_true.iloc[misclassified_indices].values, 'Predicted': y_pred[misclassified_indices]})
        confusion_pairs = misclass_df.groupby(['True', 'Predicted']).size().reset_index(name='Count').sort_values('Count', ascending=False)
        
        # Visualize and save error distribution plot
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        error_by_class = misclass_df.groupby('True').size()
        total_by_class = y_true.value_counts()
        error_rate_by_class = (error_by_class / total_by_class * 100).fillna(0)
        error_rate_by_class.plot(kind='bar')
        plt.title('Error Rate by True Class')
        plt.ylabel('Error Rate (%)')
        plt.xticks(rotation=0)
        
        plt.subplot(1, 2, 2)
        confusion_matrix_errors = pd.crosstab(misclass_df['True'], misclass_df['Predicted'])
        sns.heatmap(confusion_matrix_errors, annot=True, fmt='d', cmap='Reds')
        plt.title('Misclassification Patterns')
        
        plt.tight_layout()
        plt.savefig(os.path.join(images_dir, 'error_analysis.png'))
        plt.close()
        
        # Generate and save error analysis report
        error_report = f"""ERROR ANALYSIS REPORT
====================
Summary:
- Total test samples: {len(y_true)}
- Misclassified samples: {len(misclassified_indices)}
- Overall error rate: {len(misclassified_indices)/len(y_true)*100:.1f}%

Most Common Confusions:
{confusion_pairs.head().to_string(index=False)}
"""
        with open(os.path.join(reports_dir, 'error_analysis_report.txt'), 'w') as f:
            f.write(error_report)
        print(f"Error analysis report saved to '{os.path.join(reports_dir, 'error_analysis_report.txt')}'")
    else:
        print("No misclassifications found! Perfect classification on test set.")

# MAIN SCRIPT EXECUTION
if __name__ == "__main__":
    # --- Setup Directories ---
    results_dir = 'results'
    images_dir = os.path.join(results_dir, 'images')
    reports_dir = os.path.join(results_dir, 'reports')
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    # --- Step 1: Load and Prepare Data ---
    print("--- STEP 1: LOADING AND PREPARING DATA ---")
    X, y = load_data()
    if X is None:
        exit()

    # --- Step 2: Exploratory Data Analysis (EDA) ---
    print("\n--- STEP 2: EXPLORATORY DATA ANALYSIS ---")
    plot_class_distribution(y, os.path.join(images_dir, 'class_distribution.png'))
    plot_correlation_matrix(X, os.path.join(images_dir, 'correlation_matrix.png'))
    print(f"EDA plots saved in '{images_dir}'")

    # --- Step 3: Data Splitting and Scaling ---
    print("\n--- STEP 3: SPLITTING AND SCALING DATA ---")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("Data split and scaled successfully.")

    # --- Step 4: Find the Best K Value ---
    print("\n--- STEP 4: FINDING THE BEST K VALUE ---")
    k_values = range(1, 21)
    accuracies = [KNeighborsClassifier(n_neighbors=k).fit(X_train_scaled, y_train).score(X_test_scaled, y_test) for k in k_values]
    best_accuracy = max(accuracies)
    best_k = k_values[accuracies.index(best_accuracy)]
    print(f"Best K value found: {best_k} with an accuracy of {best_accuracy:.2%}")
    plot_k_optimization(k_values, accuracies, os.path.join(images_dir, 'k_optimization.png'))

    # --- Step 5: Train and Evaluate Final Model ---
    print("\n--- STEP 5: FINAL MODEL PERFORMANCE ---")
    final_knn = KNeighborsClassifier(n_neighbors=best_k)
    final_knn.fit(X_train_scaled, y_train)
    y_pred = final_knn.predict(X_test_scaled)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    print(f"Accuracy: {accuracy:.2%}, Precision: {precision:.2%}, Recall: {recall:.2%}")

    class_names = [f'Source {i}' for i in sorted(y.unique())]
    plot_confusion_matrix(y_test, y_pred, class_names, os.path.join(images_dir, 'confusion_matrix.png'))

    # --- Step 6: Real-World Simulation ---
    simulate_real_world_scenario(final_knn, scaler, X_test, images_dir)

    # --- Step 7: Error Analysis ---
    analyze_errors(y_test, y_pred, X_test, images_dir, reports_dir)

    # --- Step 8: Generate Final Report ---
    generate_report(X, y, best_k, accuracy, precision, recall, os.path.join(reports_dir, 'classification_report.txt'))

    print("\n--- SCRIPT FINISHED SUCCESSFULLY ---")

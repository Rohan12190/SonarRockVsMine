# 🌊 Sonar Signal Classifier & Interactive Dashboard

<p align="center">
<img src="https://img.shields.io/badge/Python-3.9+-blue.svg?style=for-the-badge&logo=python" alt="Python Badge">
<img src="https://img.shields.io/badge/Streamlit-1.25+-red.svg?style=for-the-badge&logo=streamlit" alt="Streamlit Badge">
<img src="https://img.shields.io/badge/Scikit--learn-1.3+-orange.svg?style=for-the-badge&logo=scikit-learn" alt="Scikit-learn Badge">
<img src="https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge" alt="License Badge">
</p>

An end-to-end machine learning project to classify the source of underwater acoustic signals. This repository contains the complete workflow from data analysis and model experimentation to a fully interactive web dashboard for live predictions.

## 🚀 Live Demo & Preview

This project is deployed as an interactive Streamlit dashboard. You can interact with the live application here:

➡️ **[Launch the Interactive Dashboard](# "Replace with your deployment link!")**

<p align="center">
<em><img width="1919" height="1079" alt="Screenshot 2025-09-25 003157" src="https://github.com/user-attachments/assets/1a47703f-2436-4373-bc25-555866f03b8f" />
</p>

> **Tip:** To create a great GIF, use a tool like [ScreenToGif](https://www.screentogif.com/) (Windows) or [Kap](https://getkap.co/) (macOS) to record your dashboard in action.

## ✨ Key Features

- 📊 **Comprehensive Model Comparison**: Benchmarks KNN, Support Vector Machines (SVM), Random Forest, and Gradient Boosting
- 🧩 **Ensemble Modeling**: Investigates the performance of stacking ensembles with KNN as a base learner to test for synergy between models
- 🔬 **In-Depth Analysis**: Includes Python scripts for hyperparameter optimization, detailed error analysis, and performance metric reporting
- 📈 **Rich Data Visualization**: Generates insightful plots for Exploratory Data Analysis (EDA) and model comparisons using Python and R
- 🌐 **Interactive Dashboard**: A beautiful and responsive web application built with Streamlit for real-time predictions and viewing results

## 🛠️ Tech Stack & Dependencies

| Category | Technology / Library | Purpose |
|----------|---------------------|---------|
| Core Language | Python 3.9+ | Main language for analysis and modeling |
| Data Handling | Pandas, NumPy | Data manipulation, loading, and numerical operations |
| ML & Modeling | Scikit-learn | ML models, pipelines, and evaluation metrics |
| Web Dashboard | Streamlit | Building the interactive web application |
| Visualization | Matplotlib, Seaborn | Data visualization in Python |
| UI Components | streamlit-option-menu | Creating the navigation menu in the dashboard |
| Advanced Plotting | R | Advanced statistical plotting for model comparison |

## 📂 Repository Structure

The project is organized to separate data, analysis scripts, and the final dashboard application.

```
sonar_project/
├── data/
│   └── sonar_acoustic_dataset.csv
├── dashboard/
│   └── dashboard.py
├── R/
│   └── sonar_visualizations.R
├── .gitignore
├── new.py
├── run_all_experiments.py
├── requirements.txt
└── README.md
```

## 🔧 Setup and Local Installation

To run this project on your local machine, follow these steps:

### 1. Clone the Repository

```bash
git clone https://github.com/<YOUR-USERNAME>/sonar_project.git
cd sonar_project
```

### 2. Create and Activate a Virtual Environment

Using a virtual environment is highly recommended to manage dependencies.

**On macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**On Windows:**
```bash
python -m venv venv
.\venv\Scripts\activate
```

### 3. Install Dependencies

Install all the required Python libraries using the requirements.txt file.

```bash
pip install -r requirements.txt
```

## 📈 How to Use

### 1. Run the Analysis Scripts

These scripts generate all the reports, models, and visualizations. Run them from the root directory.

**Run the in-depth KNN analysis and generate EDA plots:**
```bash
python new.py
```

**Run the comparison of all standalone and stacked models:**
```bash
python run_all_experiments.py
```

These scripts will populate a `results/` directory with `images/` and `reports/`.

### 2. Launch the Interactive Dashboard

To start the web application, run the following command:

```bash
streamlit run dashboard/dashboard.py
```

Navigate to `http://localhost:8501` in your web browser to view the dashboard.

## 📜 License

This project is licensed under the MIT License. See the LICENSE file for more details.

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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
from streamlit_option_menu import option_menu

st.set_page_config(
    page_title="Sonar Classification Dashboard",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

def load_css():
    st.markdown("""
        <style>
        [data-testid="stAppViewContainer"] > .main {
            background: linear-gradient(135deg, #0D1B2A 0%, #1B263B 74%);
        }
        
        div[data-testid="stHorizontalBlock"] > div:first-child {
            background-color: rgba(27, 38, 59, 0.7);
            backdrop-filter: blur(10px);
            border-radius: 50px;
            padding: 10px 20px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
            margin: 15px auto;
            max-width: fit-content;
        }

        [data-testid="stSidebar"] {
            background-color: #1B263B;
        }

        .stMarkdown, .st-emotion-cache-1y4p8pa, h1, h2, h3, h4, h5, h6 {
            color: #E0E1DD;
        }

        .st-emotion-cache-1tpl0xr p {
            color: #A9A9A9;
        }
        
        .stButton>button {
            border: 2px solid #415A77;
            background-color: transparent;
            color: #E0E1DD;
        }
        .stButton>button:hover {
            border-color: #778DA9;
            background-color: #415A77;
            color: #FFFFFF;
        }
        </style>
    """, unsafe_allow_html=True)

load_css()

@st.cache_data
def load_data():
    # This is the corrected file path for Streamlit deployment
    file_path = "data/sonar_acoustic_dataset.csv"
    if not os.path.exists(file_path):
        st.error(f"Error: Data file not found at '{file_path}'. Please ensure the data folder is in the root of the repository.")
        return None, None
    data = pd.read_csv(file_path)
    X = data.iloc[:, :13]
    y = data["Sound Source"]
    return X, y

@st.cache_resource
def train_all_models(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    models = {
        'KNN': make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=1)),
        'SVM': make_pipeline(StandardScaler(), SVC(probability=True, random_state=42)),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
    }

    for name, model in models.items():
        model.fit(X_train, y_train)

    for name, model_instance in [('SVM', models['SVM']), ('Random Forest', models['Random Forest']), ('Gradient Boosting', models['Gradient Boosting'])]:
        estimators = [('knn', models['KNN']), (name.lower().replace(" ", ""), model_instance)]
        stacked_model = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression(), cv=5)
        stacked_model.fit(X_train, y_train)
        stacked_model_name = f"Stacked (KNN + {name})"
        models[stacked_model_name] = stacked_model
        
    return models

X, y = load_data()
if X is not None:
    models = train_all_models(X, y)

page = option_menu(
    menu_title=None,
    options=["Project Overview", "Exploratory Data Analysis", "Model Performance", "Live Sonar Prediction"],
    icons=["house-door-fill", "bar-chart-line-fill", "graph-up-arrow", "broadcast-pin"],
    orientation="horizontal",
    styles={
        "container": {"padding": "0!important", "background-color": "transparent", "border-radius": "10px"},
        "icon": {"color": "#778DA9", "font-size": "22px"}, 
        "nav-link": {
            "font-size": "16px",
            "text-align": "center",
            "margin":"0px 5px",
            "padding": "10px",
            "--hover-color": "#1B263B"
        },
        "nav-link-selected": {"background-color": "#415A77"},
    }
)

st.sidebar.title("🌊 Sonar Classifier")
st.sidebar.info("This dashboard analyzes and compares machine learning models for classifying underwater sonar signals.")

if page == "Project Overview":
    st.title("Project Overview")
    st.markdown("""
    This project aims to accurately classify the source of underwater acoustic signals using machine learning. 
    We start with a baseline K-Nearest Neighbors (KNN) model and then explore advanced **stacked ensemble models** to improve performance.

    ### Dataset
    The dataset contains **13 acoustic features** extracted from sonar signals, with the goal of classifying the sound source into one of four categories.
    """)
    if X is not None:
        st.dataframe(X.head(10))

elif page == "Exploratory Data Analysis":
    st.title("Exploratory Data Analysis (EDA)")
    
    if X is not None:
        st.subheader("Dataset Characteristics")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("##### Class Distribution")
            fig, ax = plt.subplots()
            sns.countplot(x=y.map({0: 'Source 0', 1: 'Source 1', 2: 'Source 2', 3: 'Source 3'}), ax=ax)
            st.pyplot(fig)
        
        with col2:
            st.markdown("##### Feature Correlation Matrix")
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(X.corr(), annot=False, cmap='viridis', ax=ax)
            st.pyplot(fig)
            
        st.markdown("---")
        st.subheader("Model Analysis Visuals")
        
        # Paths are relative to the root for Streamlit deployment
        k_path = "results/images/k_optimization.png"
        error_path = "results/images/error_analysis.png"
        heatmap_path = "results/images/Rplot3.png"

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("##### K-Value Optimization for KNN")
            if os.path.exists(k_path):
                st.image(k_path, caption="Finding the optimal number of neighbors for the baseline KNN model.", use_container_width=True)
            else:
                st.warning(f"Plot not found at: {k_path}")
        
        with col2:
            st.markdown("##### KNN Error Analysis")
            if os.path.exists(error_path):
                st.image(error_path, caption="Analysis of misclassifications made by the baseline KNN model.", use_container_width=True)
            else:
                st.warning(f"Plot not found at: {error_path}")

        st.markdown("---")
        st.markdown("##### Model Comparison Matrix (from R)")
        if os.path.exists(heatmap_path):
            st.image(heatmap_path, caption="Heatmap showing pairwise accuracy differences between standalone models.", use_container_width=True)
        else:
            st.warning(f"Plot not found at: {heatmap_path}")

elif page == "Model Performance":
    st.title("Model Performance Comparison")
    st.markdown("This section displays the final performance comparison visuals generated by the R script.")
    
    if X is not None:
        lollipop_path = "results/images/Rplot1.png"
        dumbbell_path = "results/images/Rplot2.png"

        st.subheader("Standalone Model Performance")
        if os.path.exists(lollipop_path):
            st.image(lollipop_path, use_container_width=True)
        else:
            st.warning(f"Plot not found at: {lollipop_path}")

        st.subheader("Stacked Ensemble Performance vs. KNN Baseline")
        if os.path.exists(dumbbell_path):
            st.image(dumbbell_path, use_container_width=True)
        else:
            st.warning(f"Plot not found at: {dumbbell_path}")

elif page == "Live Sonar Prediction":
    st.title("Live Sonar Prediction")
    st.markdown("Use the sliders in the sidebar to input sonar features and classify the sound source in real-time.")
    
    if X is not None:
        st.sidebar.header("Input Sonar Features")
        input_features = {col: st.sidebar.slider(
            label=col, min_value=float(X[col].min()), max_value=float(X[col].max()), 
            value=float(X[col].mean()), step=0.01) for col in X.columns}
            
        model_name = st.sidebar.selectbox("Choose a model:", list(models.keys()))
        
        if st.sidebar.button("Classify Sound Source", type="primary"):
            input_df = pd.DataFrame([input_features])
            prediction = models[model_name].predict(input_df)[0]
            probabilities = models[model_name].predict_proba(input_df)[0]
            
            st.subheader(f"Prediction using {model_name}")
            col1, col2 = st.columns([1, 2])
            with col1:
                st.metric("Predicted Sound Source", f"Source {prediction}")
                st.metric("Confidence", f"{probabilities[prediction]:.2%}")
            
            with col2:
                prob_df = pd.DataFrame({"Source": [f"Source {i}" for i in range(4)], "Probability": probabilities})
                fig, ax = plt.subplots()
                sns.barplot(x="Probability", y="Source", data=prob_df, ax=ax, palette="viridis")
                ax.set_title("Prediction Probabilities")
                st.pyplot(fig)


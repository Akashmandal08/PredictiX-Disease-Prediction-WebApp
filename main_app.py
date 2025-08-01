import pickle
import streamlit as st
from streamlit_option_menu import option_menu
import json
import pandas as pd
import os
import base64

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="PredictiX - AI Disease Diagnosis",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- HELPER FUNCTION TO LOAD FILES ---
# This caches the files to prevent reloading on every interaction, improving performance.
@st.cache_data
def load_file(file_path):
    """Loads a file safely from the specified path."""
    if not os.path.exists(file_path):
        st.error(f"Error: The file was not found at {file_path}")
        st.stop()
    try:
        if file_path.endswith('.pkl') or file_path.endswith('.sav'):
            with open(file_path, 'rb') as f:
                return pickle.load(f)
        elif file_path.endswith('.json'):
            with open(file_path, 'r') as f:
                return json.load(f)
    except Exception as e:
        st.error(f"Error loading {file_path}: {e}")
        st.stop()

# --- LOAD ALL MODELS AND PREPROCESSING FILES ---
# Grouping file loading for better organization
try:
    # Diabetes Prediction Files
    diabetes_files = {
        "all_features": load_file("Preprocessing Files/ML-Project-2-Diabetes_Prediction_Pre_Processing_Files/columns.pkl"),
        "scaler": load_file("Preprocessing Files/ML-Project-2-Diabetes_Prediction_Pre_Processing_Files/scaler.pkl"),
        "best_features_svc": load_file("Best Features/ML-Project-2-Diabetes_Prediction_Best_Features/best_features_svc.json"),
        "best_features_lr": load_file("Best Features/ML-Project-2-Diabetes_Prediction_Best_Features/best_features_lr.json"),
        "best_features_rfc": load_file("Best Features/ML-Project-2-Diabetes_Prediction_Best_Features/best_features_rfc.json"),
        "model_svc": load_file("Models/ML-Project-2-Diabetes_Prediction_Models/diabetes_disease_trained_svc_model.sav"),
        "model_lr": load_file("Models/ML-Project-2-Diabetes_Prediction_Models/diabetes_disease_trained_lr_model.sav"),
        "model_rfc": load_file("Models/ML-Project-2-Diabetes_Prediction_Models/diabetes_disease_trained_rfc_model.sav"),
    }

    # Heart Disease Prediction Files
    heart_disease_files = {
        "all_columns": load_file("Preprocessing Files/ML-Project-9-Heart_Disease_Prediction_Pre_Processing_Files/columns.pkl"),
        "cat_columns": load_file("Preprocessing Files/ML-Project-9-Heart_Disease_Prediction_Pre_Processing_Files/cat_columns.pkl"),
        "encoder": load_file("Preprocessing Files/ML-Project-9-Heart_Disease_Prediction_Pre_Processing_Files/encoder.pkl"),
        "encoded_columns": load_file("Preprocessing Files/ML-Project-9-Heart_Disease_Prediction_Pre_Processing_Files/encoded_columns.pkl"),
        "training_columns": load_file("Preprocessing Files/ML-Project-9-Heart_Disease_Prediction_Pre_Processing_Files/training_columns.pkl"),
        "scaler": load_file("Preprocessing Files/ML-Project-9-Heart_Disease_Prediction_Pre_Processing_Files/scaler.pkl"),
        "best_features_xgb": load_file("Best Features/ML-Project-9-Heart_Disease_Prediction_Best_Features/best_features_xgb.json"),
        "best_features_rfc": load_file("Best Features/ML-Project-9-Heart_Disease_Prediction_Best_Features/best_features_rfc.json"),
        "best_features_lr": load_file("Best Features/ML-Project-9-Heart_Disease_Prediction_Best_Features/best_features_lr.json"),
        "model_xgb": load_file("Models/ML-Project-9-Heart_Disease_Prediction_Models/heart_disease_trained_xgb_model.sav"),
        "model_rfc": load_file("Models/ML-Project-9-Heart_Disease_Prediction_Models/heart_disease_trained_rfc_model.sav"),
        "model_lr": load_file("Models/ML-Project-9-Heart_Disease_Prediction_Models/heart_disease_trained_lr_model.sav"),
    }

    # Parkinson's Disease Prediction Files
    parkinson_files = {
        "all_features": load_file("Preprocessing Files/ML-Project-14-Parkinson's_Disease_Prediction_Pre_Processing_Files/columns.pkl"),
        "scaler": load_file("Preprocessing Files/ML-Project-14-Parkinson's_Disease_Prediction_Pre_Processing_Files/scaler.pkl"),
        "best_features_knn": load_file("Best Features/ML-Project-14-Parkinson's_Disease_Prediction_Best_Features/best_features_knn.json"),
        "best_features_xgb": load_file("Best Features/ML-Project-14-Parkinson's_Disease_Prediction_Best_Features/best_features_xgb.json"),
        "best_features_rfc": load_file("Best Features/ML-Project-14-Parkinson's_Disease_Prediction_Best_Features/best_features_rfc.json"),
        "model_knn": load_file("Models/ML-Project-14-Parkinson's_Disease_Prediction_Models/parkinsons_disease_trained_knn_model.sav"),
        "model_xgb": load_file("Models/ML-Project-14-Parkinson's_Disease_Prediction_Models/parkinsons_disease_trained_xgb_model.sav"),
        "model_rfc": load_file("Models/ML-Project-14-Parkinson's_Disease_Prediction_Models/parkinsons_disease_trained_rfc_model.sav"),
    }

    # Breast Cancer Prediction Files
    breast_cancer_files = {
        "all_features": load_file("Preprocessing Files/ML-Project-19-Breast_Cancer_Classification_Pre_Processing_Files/columns.pkl"),
        "scaler": load_file("Preprocessing Files/ML-Project-19-Breast_Cancer_Classification_Pre_Processing_Files/scaler.pkl"),
        "best_features_lr": load_file("Best Features/ML-Project-19-Breast_Cancer_Classification_Best_Features/best_features_lr.json"),
        "best_features_xgb": load_file("Best Features/ML-Project-19-Breast_Cancer_Classification_Best_Features/best_features_xgb.json"),
        "best_features_knn": load_file("Best Features/ML-Project-19-Breast_Cancer_Classification_Best_Features/best_features_knn.json"),
        "model_lr": load_file("Models/ML-Project-19-Breast_Cancer_Classification_Models/parkinsons_disease_trained_lr_model.sav"),
        "model_xgb": load_file("Models/ML-Project-19-Breast_Cancer_Classification_Models/parkinsons_disease_trained_xgb_model.sav"),
        "model_knn": load_file("Models/ML-Project-19-Breast_Cancer_Classification_Models/parkinsons_disease_trained_knn_model.sav"),
    }

except Exception as e:
    st.error("An error occurred during file loading. Please ensure all model and data files are in their correct directories.")
    st.exception(e)
    st.stop()


# --- UI & STYLING ---
def load_css():
    """Injects custom CSS for a modern, eye-catching UI."""
    st.markdown("""
        <style>
            /* Import Google Font */
            @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');

            /* Overall App Styling */
            body {
                font-family: 'Poppins', sans-serif;
            }
            .stApp {
                background-color: #1a1a2e; /* Dark blue background */
            }
            
            /* Sidebar Styling */
            .css-1d391kg {
                background-color: #16213e; /* Slightly lighter blue for sidebar */
                border-right: 2px solid #0f3460;
            }

            /* Logo in Sidebar */
            .sidebar-logo {
                text-align: center;
                padding: 2rem 1rem 1rem 1rem;
            }
            .sidebar-logo img {
                width: 120px;
                border-radius: 50%;
                box-shadow: 0px 10px 30px rgba(0, 212, 255, 0.3);
                border: 3px solid #e94560;
            }

            /* Option Menu Styling */
            div[data-testid="stSidebarNav"] .st-emotion-cache-1cypcdb {
                padding: 10px 0;
            }
            div[data-testid="stSidebarNav"] .st-emotion-cache-1cypcdb a {
                font-size: 1.1rem;
                font-weight: 600;
                color: #c9c9c9;
                border-radius: 10px;
                margin: 5px 15px;
                transition: background-color 0.3s ease-in-out, color 0.3s ease-in-out;
            }
            div[data-testid="stSidebarNav"] .st-emotion-cache-1cypcdb a:hover {
                background-color: rgba(233, 69, 96, 0.2);
                color: #e94560;
            }
            div[data-testid="stSidebarNav"] .st-emotion-cache-1cypcdb a[aria-current="page"] {
                background-color: #e94560;
                color: white;
            }

            /* Main Content Styling */
            .block-container {
                padding-top: 2rem;
            }
            h1, h2, h3, h4, h5, h6 {
                font-weight: 700;
                color: #ffffff;
            }
            h1 {
                color: #e94560;
                text-align: center;
                margin-bottom: 1rem;
            }
            
            /* Styled Containers (Cards) */
            .st-emotion-cache-1r4qj8v {
                background: linear-gradient(145deg, #1a1a2e, #16213e);
                border: 1px solid #0f3460;
                border-radius: 15px;
                padding: 2rem;
                box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
                backdrop-filter: blur(4px);
                -webkit-backdrop-filter: blur(4px);
            }

            /* Input Widgets Styling */
            .stNumberInput > div > div > input, .stSelectbox > div > div {
                border-radius: 8px;
                border: 2px solid #0f3460;
                background-color: #16213e;
                color: white;
                transition: border-color 0.3s, box-shadow 0.3s;
            }
            .stNumberInput > div > div > input:focus, .stSelectbox > div > div:focus-within {
                border-color: #e94560;
                box-shadow: 0 0 8px rgba(233, 69, 96, 0.5);
            }
            
            /* Button Styling */
            .stButton > button {
                border: 2px solid #e94560;
                background: transparent;
                color: #e94560;
                border-radius: 10px;
                padding: 10px 20px;
                font-size: 1.2rem;
                font-weight: 700;
                width: 100%;
                transition: all 0.3s ease-in-out;
            }
            .stButton > button:hover {
                background-color: #e94560;
                color: white;
                box-shadow: 0 0 20px rgba(233, 69, 96, 0.5);
                transform: translateY(-3px);
            }
             .stButton > button:active {
                transform: translateY(0);
            }
            
            /* Prediction Result Styling */
            .result-box {
                border-radius: 15px;
                padding: 25px;
                margin-top: 25px;
                text-align: center;
                font-size: 1.8rem;
                font-weight: 700;
                animation: fadeIn 1s ease-in-out;
                border: 3px solid;
            }
            .result-positive {
                border-color: #ff4b4b;
                background: linear-gradient(45deg, rgba(255, 75, 75, 0.1), rgba(255, 75, 75, 0.2));
                color: #ff4b4b;
                text-shadow: 0 0 15px rgba(255, 75, 75, 0.7);
            }
            .result-negative {
                border-color: #3dd56d;
                background: linear-gradient(45deg, rgba(61, 213, 109, 0.1), rgba(61, 213, 109, 0.2));
                color: #3dd56d;
                text-shadow: 0 0 15px rgba(61, 213, 109, 0.7);
            }
            
            @keyframes fadeIn {
                from { opacity: 0; transform: scale(0.8); }
                to { opacity: 1; transform: scale(1); }
            }
            
            /* Expander Styling */
            .st-emotion-cache-p5msec {
                border: 1px solid #0f3460;
                border-radius: 10px;
                background-color: #16213e;
            }
        </style>
    """, unsafe_allow_html=True)


# --- PREDICTION FUNCTIONS ---
def diabetes_prediction(input_data):
    df = pd.DataFrame([input_data], columns=diabetes_files["all_features"])
    df[diabetes_files["all_features"]] = diabetes_files["scaler"].transform(df[diabetes_files["all_features"]])
    predictions = {
        "Support Vector Classifier": diabetes_files["model_svc"].predict(df[diabetes_files["best_features_svc"]])[0],
        "Logistic Regression": diabetes_files["model_lr"].predict(df[diabetes_files["best_features_lr"]])[0],
        "Random Forest Classifier": diabetes_files["model_rfc"].predict(df[diabetes_files["best_features_rfc"]])[0],
    }
    return predictions

def heart_disease_prediction(input_data):
    df = pd.DataFrame([input_data], columns=heart_disease_files["all_columns"])
    df[heart_disease_files["cat_columns"]] = df[heart_disease_files["cat_columns"]].astype('str')
    encoded = heart_disease_files["encoder"].transform(df[heart_disease_files["cat_columns"]])
    encoded_df = pd.DataFrame(encoded, columns=heart_disease_files["encoded_columns"])
    final_df = pd.concat([df.drop(heart_disease_files["cat_columns"], axis=1).reset_index(drop=True), encoded_df], axis=1)
    scaled_df = pd.DataFrame(heart_disease_files["scaler"].transform(final_df), columns=heart_disease_files["training_columns"])
    predictions = {
        "XG Boost Classifier": heart_disease_files["model_xgb"].predict(scaled_df[heart_disease_files["best_features_xgb"]])[0],
        "Random Forest Classifier": heart_disease_files["model_rfc"].predict(scaled_df[heart_disease_files["best_features_rfc"]])[0],
        "Logistic Regression": heart_disease_files["model_lr"].predict(scaled_df[heart_disease_files["best_features_lr"]])[0],
    }
    return predictions

def parkinson_disease_prediction(input_data):
    df = pd.DataFrame([input_data], columns=parkinson_files["all_features"])
    df[parkinson_files["all_features"]] = parkinson_files["scaler"].transform(df[parkinson_files["all_features"]])
    predictions = {
        "K Neighbors Classifier": parkinson_files["model_knn"].predict(df[parkinson_files["best_features_knn"]])[0],
        "XG Boost Classifier": parkinson_files["model_xgb"].predict(df[parkinson_files["best_features_xgb"]])[0],
        "Random Forest Classifier": parkinson_files["model_rfc"].predict(df[parkinson_files["best_features_rfc"]])[0],
    }
    return predictions

def breast_cancer_prediction(input_data):
    df = pd.DataFrame([input_data], columns=breast_cancer_files["all_features"])
    df[breast_cancer_files["all_features"]] = breast_cancer_files["scaler"].transform(df[breast_cancer_files["all_features"]])
    predictions = {
        "Logistic Regression": breast_cancer_files["model_lr"].predict(df[breast_cancer_files["best_features_lr"]])[0],
        "XG Boost Classifier": breast_cancer_files["model_xgb"].predict(df[breast_cancer_files["best_features_xgb"]])[0],
        "K Neighbors Classifier": breast_cancer_files["model_knn"].predict(df[breast_cancer_files["best_features_knn"]])[0],
    }
    return predictions

# --- HELPER FUNCTION TO DISPLAY PREDICTION ---
def display_prediction(prediction_result, positive_text, negative_text):
    if prediction_result == 1:
        st.markdown(f'<div class="result-box result-positive">{positive_text}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="result-box result-negative">{negative_text}</div>', unsafe_allow_html=True)


# --- MAIN APPLICATION LOGIC ---
def main():
    load_css()

    with st.sidebar:
        # --- LOGO ---
        try:
            with open("logo.png", "rb") as f:
                data = base64.b64encode(f.read()).decode("utf-8")
            st.markdown(
                f'<div class="sidebar-logo"><img src="data:image/png;base64,{data}" alt="Logo"></div>',
                unsafe_allow_html=True,
            )
        except FileNotFoundError:
            st.markdown(
                '<div class="sidebar-logo" style="color: #e94560; font-weight: 700;">PredictiX</div>',
                unsafe_allow_html=True,
            )
        
        # --- NAVIGATION MENU ---
        selected = option_menu(
            menu_title=None,
            options=['Diabetes', 'Heart Disease', "Parkinson's", 'Breast Cancer'],
            icons=['droplet-fill', 'heart-pulse-fill', 'person-arms-up', 'gender-female'],
            default_index=0,
        )

    # --- DIABETES PREDICTION PAGE ---
    if selected == 'Diabetes':
        st.title("🩸 Diabetes Prediction")
        
        with st.container():
            col1, col2, col3 = st.columns(3)
            with col1:
                Pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, step=1, format="%d")
                BloodPressure = st.number_input("Blood Pressure (mm Hg)", min_value=0.0, format="%.2f")
                DiabetesPedigreeFunction = st.number_input("Diabetes Pedigree Function", min_value=0.0, format="%.3f")
            with col2:
                Glucose = st.number_input("Glucose Level (mg/dL)", min_value=0.0, format="%.2f")
                SkinThickness = st.number_input("Skin Thickness (mm)", min_value=0.0, format="%.2f")
                BMI = st.number_input("Body Mass Index (BMI)", min_value=0.0, format="%.2f")
            with col3:
                Insulin = st.number_input("Insulin Level (mu U/ml)", min_value=0.0, format="%.2f")
                Age = st.number_input("Age (years)", min_value=0, max_value=120, step=1, format="%d")

        if st.button("Predict Diabetes Status"):
            input_data = [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]
            predictions = diabetes_prediction(input_data)
            display_prediction(predictions["Random Forest Classifier"], "Positive: The person is Diabetic", "Negative: The person is Not Diabetic")

            with st.expander("View Advanced Model Predictions"):
                for model, result in predictions.items():
                    st.write(f"**{model}:** {'Diabetic' if result == 1 else 'Not Diabetic'}")

    # --- HEART DISEASE PREDICTION PAGE ---
    if selected == 'Heart Disease':
        st.title("❤️ Heart Disease Prediction")
        
        with st.container():
            col1, col2, col3 = st.columns(3)
            with col1:
                age = st.number_input("Age", min_value=1, max_value=120, step=1)
                resting_bp = st.number_input("Resting Blood Pressure", min_value=0)
                resting_ecg = st.selectbox('Resting ECG', [0, 1, 2], format_func=lambda x: f"Type {x}")
                oldpeak = st.number_input("Oldpeak")
                thal = st.selectbox('Thal', [0, 1, 2, 3], format_func=lambda x: {0: 'None', 1: 'Normal', 2: 'Fixed Defect', 3: 'Reversible Defect'}[x])
            with col2:
                sex = st.selectbox('Sex', [0, 1], format_func=lambda x: 'Female' if x == 0 else 'Male')
                serum_cholestoral = st.number_input("Serum Cholestoral (mg/dl)")
                max_heart_achieved = st.number_input("Max Heart Rate Achieved")
                slope_of_peak_exercise = st.selectbox('Slope of Peak Exercise', [0, 1, 2], format_func=lambda x: f"Type {x}")
            with col3:
                chest_pain = st.selectbox('Chest Pain Type', [0, 1, 2, 3], format_func=lambda x: f"Type {x}")
                fasting_blood_sugar = st.selectbox('Fasting Blood Sugar > 120 mg/dl', [0, 1], format_func=lambda x: 'False' if x == 0 else 'True')
                exercise_induced_angina = st.selectbox('Exercise Induced Angina', [0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes')
                number_of_major_vessels = st.selectbox('Number of Major Vessels', [0, 1, 2, 3, 4])
                
        if st.button("Predict Heart Disease Status"):
            input_data = [age, sex, chest_pain, resting_bp, serum_cholestoral, fasting_blood_sugar, resting_ecg, max_heart_achieved, exercise_induced_angina, oldpeak, slope_of_peak_exercise, number_of_major_vessels, thal]
            predictions = heart_disease_prediction(input_data)
            display_prediction(predictions["XG Boost Classifier"], "Positive: This person has Heart Disease", "Negative: This person does Not have Heart Disease")
            
            with st.expander("View Advanced Model Predictions"):
                for model, result in predictions.items():
                    st.write(f"**{model}:** {'Heart Disease Positive' if result == 1 else 'Heart Disease Negative'}")

    # --- PARKINSON'S DISEASE PREDICTION PAGE ---
    if selected == "Parkinson's":
        st.title("🧠 Parkinson's Disease Prediction")

        with st.container():
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                Fo = st.number_input("MDVP:Fo(Hz)", format="%.6f")
                Jitter_per = st.number_input("MDVP:Jitter(%)", format="%.6f")
                PPQ = st.number_input("MDVP:PPQ", format="%.6f")
                Shimmer_dB = st.number_input("MDVP:Shimmer(dB)", format="%.6f")
                APQ = st.number_input("MDVP:APQ", format="%.6f")
                RPDE = st.number_input("RPDE", format="%.6f")
            with col2:
                Fhi = st.number_input("MDVP:Fhi(Hz)", format="%.6f")
                Jitter_Abs = st.number_input("MDVP:Jitter(Abs)", format="%.6f")
                Jitter_DDP = st.number_input("Jitter:DDP", format="%.6f")
                Shimmer_APQ3 = st.number_input("Shimmer:APQ3", format="%.6f")
                Shimmer_DDA = st.number_input("Shimmer:DDA", format="%.6f")
                DFA = st.number_input("DFA", format="%.6f")
            with col3:
                Flo = st.number_input("MDVP:Flo(Hz)", format="%.6f")
                RAP = st.number_input("MDVP:RAP", format="%.6f")
                Shimmer = st.number_input("MDVP:Shimmer", format="%.6f")
                Shimmer_APQ5 = st.number_input("Shimmer:APQ5", format="%.6f")
                NHR = st.number_input("NHR", format="%.6f")
                spread1 = st.number_input("spread1", format="%.6f")
            with col4:
                HNR = st.number_input("HNR", format="%.6f")
                spread2 = st.number_input("spread2", format="%.6f")
                D2 = st.number_input("D2", format="%.6f")
                PPE = st.number_input("PPE", format="%.6f")

        if st.button("Predict Parkinson's Status"):
            input_data = [Fo, Fhi, Flo, Jitter_per, Jitter_Abs, RAP, PPQ, Jitter_DDP, Shimmer, Shimmer_dB, Shimmer_APQ3, Shimmer_APQ5, APQ, Shimmer_DDA, NHR, HNR, RPDE, DFA, spread1, spread2, D2, PPE]
            predictions = parkinson_disease_prediction(input_data)
            display_prediction(predictions["K Neighbors Classifier"], "Positive: This person has Parkinson's Disease", "Negative: This person does Not have Parkinson's Disease")

            with st.expander("View Advanced Model Predictions"):
                for model, result in predictions.items():
                    st.write(f"**{model}:** {'Parkinson\'s Positive' if result == 1 else 'Parkinson\'s Negative'}")
    
    # --- BREAST CANCER PREDICTION PAGE ---
    if selected == 'Breast Cancer':
        st.title('🚺 Breast Cancer Prediction')
        
        with st.container():
            st.subheader("Mean Values")
            col1, col2, col3 = st.columns(3)
            with col1:
                mean_radius = st.number_input("Mean Radius", format="%.6f")
                mean_area = st.number_input("Mean Area", format="%.6f")
                mean_concavity = st.number_input("Mean Concavity", format="%.6f")
            with col2:
                mean_texture = st.number_input("Mean Texture", format="%.6f")
                mean_smoothness = st.number_input("Mean Smoothness", format="%.6f")
                mean_concave_points = st.number_input("Mean Concave Points", format="%.6f")
            with col3:
                mean_perimeter = st.number_input("Mean Perimeter", format="%.6f")
                mean_compactness = st.number_input("Mean Compactness", format="%.6f")
                mean_symmetry = st.number_input("Mean Symmetry", format="%.6f")
                mean_fractal_dimension = st.number_input("Mean Fractal Dimension", format="%.6f")

        with st.expander("Error and Worst Values"):
            st.subheader("Standard Error Values")
            col1, col2, col3 = st.columns(3)
            with col1:
                radius_error = st.number_input("Radius Error", format="%.6f")
                perimeter_error = st.number_input("Perimeter Error", format="%.6f")
                compactness_error = st.number_input("Compactness Error", format="%.6f")
                symmetry_error = st.number_input("Symmetry Error", format="%.6f")
            with col2:
                texture_error = st.number_input("Texture Error", format="%.6f")
                area_error = st.number_input("Area Error", format="%.6f")
                concavity_error = st.number_input("Concavity Error", format="%.6f")
                fractal_dimension_error = st.number_input("Fractal Dimension Error", format="%.6f")
            with col3:
                smoothness_error = st.number_input("Smoothness Error", format="%.6f")
                concave_points_error = st.number_input("Concave Points Error", format="%.6f")
            
            st.subheader("Worst Values")
            col1, col2, col3 = st.columns(3)
            with col1:
                worst_radius = st.number_input("Worst Radius", format="%.6f")
                worst_area = st.number_input("Worst Area", format="%.6f")
                worst_concavity = st.number_input("Worst Concavity", format="%.6f")
            with col2:
                worst_texture = st.number_input("Worst Texture", format="%.6f")
                worst_smoothness = st.number_input("Worst Smoothness", format="%.6f")
                worst_concave_points = st.number_input("Worst Concave Points", format="%.6f")
            with col3:
                worst_perimeter = st.number_input("Worst Perimeter", format="%.6f")
                worst_compactness = st.number_input("Worst Compactness", format="%.6f")
                worst_symmetry = st.number_input("Worst Symmetry", format="%.6f")
                worst_fractal_dimension = st.number_input("Worst Fractal Dimension", format="%.6f")

        if st.button("Predict Breast Cancer Status"):
            input_data = [mean_radius, mean_texture, mean_perimeter, mean_area, mean_smoothness, mean_compactness, mean_concavity, mean_concave_points, mean_symmetry, mean_fractal_dimension, radius_error, texture_error, perimeter_error, area_error, smoothness_error, compactness_error, concavity_error, concave_points_error, symmetry_error, fractal_dimension_error, worst_radius, worst_texture, worst_perimeter, worst_area, worst_smoothness, worst_compactness, worst_concavity, worst_concave_points, worst_symmetry, worst_fractal_dimension]
            predictions = breast_cancer_prediction(input_data)
            display_prediction(predictions["XG Boost Classifier"], "The tumor is Malignant", "The tumor is Benign")

            with st.expander("View Advanced Model Predictions"):
                for model, result in predictions.items():
                    st.write(f"**{model}:** {'Malignant' if result == 1 else 'Benign'}")

if __name__ == '__main__':
    main()
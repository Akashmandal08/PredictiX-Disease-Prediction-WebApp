# PredictiX - AI-Powered Multiple Disease Prediction System

**Author**: Akash Mandal  
**GitHub**: [Akashmandal08](https://github.com/Akashmandal08)  
**Email**: akashmandal.9490@gmail.com  
**Repository**: [PredictiX-Disease-Prediction-WebApp](https://github.com/Akashmandal08/PredictiX-Disease-Prediction-WebApp.git)

---

## 🧠 Introduction and Motivation

Chronic diseases like **Diabetes**, **Heart Disease**, **Parkinson's**, and **Breast Cancer** are leading causes of illness and death globally. Early detection improves outcomes significantly, but delays in diagnosis due to lack of awareness or access to care are common.

**PredictiX** is an AI-powered web application built to bridge this gap by providing users a simple, accessible, and intuitive platform to perform a preliminary risk assessment for multiple diseases using machine learning models.

---

## ❓ Problem Statement

To design a unified web platform that:
- Allows users to input health data
- Predicts the likelihood of four diseases using trained ML models
- Displays the results clearly with an intuitive interface
- Provides educational insights but does **not** replace professional medical advice

---

## 🎯 Objectives

- Select and train suitable ML models for each disease
- Preprocess standard UCI datasets for accuracy
- Build a clean and responsive web interface with Streamlit
- Enable real-time predictions using serialized models
- Emphasize the system’s role as an informative tool, not a diagnostic tool

---

## ⚙️ Methodology

1. **Data Collection**: Standard datasets from the UCI repository.
2. **Preprocessing**: Missing value handling, encoding, and scaling.
3. **Model Training**: Using Logistic Regression, KNN, SVC, Random Forest, and XGBoost.
4. **Model Persistence**: Save models and preprocessing objects using Pickle.
5. **Backend**: Core prediction logic in Python.
6. **Frontend**: Streamlit-based interface for interactive use.
7. **Integration**: End-to-end system deployment as a single web app.

---

## 🧪 Technologies Used

- Python 3.9+
- Pandas, NumPy
- Scikit-learn, XGBoost
- Streamlit
- Pickle
- UCI Datasets

---

## 📦 How to Set Up and Run Locally

### 1. Clone the Repository

```bash
git clone https://github.com/Akashmandal08/PredictiX-Disease-Prediction-WebApp.git
cd PredictiX-Disease-Prediction-WebApp
```

### 2. Create and Activate a Virtual Environment

**For Windows:**

```bash
python -m venv venv
venv\Scripts\activate
```

**For macOS/Linux:**

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Required Packages

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit App

```bash
streamlit run app.py
```

This will open the app in your default browser at `http://localhost:8501`.

---

## 📈 Future Enhancements

- Add more diseases and predictive models
- Include user login and history tracking
- Enable automatic retraining with new data
- Introduce Explainable AI (XAI) to show influential features

---

## 📌 Notes

- This tool is meant for educational and awareness purposes only.
- It is **not a diagnostic tool** and should not replace medical consultation.
- Model predictions depend on dataset quality and user input accuracy.

---

## 📄 License

This project is licensed under the MIT License.

---

## 📬 Contact

**Akash Mandal**  
📧 Email: akashmandal.9490@gmail.com  
🔗 GitHub: [Akashmandal08](https://github.com/Akashmandal08)  
📂 Repository: [PredictiX-Disease-Prediction-WebApp](https://github.com/Akashmandal08/PredictiX-Disease-Prediction-WebApp.git)

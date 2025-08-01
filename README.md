# Multiple Disease Prediction System WebApp
This repository contains a Multiple Disease Prediction System WebApp developed using Streamlit and hosted on Streamlit Cloud. The web app integrates four different disease prediction systems, each utilizing machine learning models to provide accurate predictions. The diseases covered are:


1. Diabetes Prediction System

2. Heart Disease Prediction System

3. Parkinson Disease Prediction System

4. Breast Cancer Prediction System


Table of Contents:

* OverviewPredictiX - AI-Powered Multiple Disease Prediction System
<!-- Replace with a URL to a screenshot of your app -->

<img width="1920" height="1020" alt="SUGAR" src="https://github.com/user-attachments/assets/d6d51b54-c668-4b1f-b8bf-ba2c5704bdac" />

<img width="1920" height="1020" alt="HEART" src="https://github.com/user-attachments/assets/a2ef63d1-2bf9-43be-a546-661b1c59e0c2" />

<img width="1920" height="1020" alt="PARKISONAS" src="https://github.com/user-attachments/assets/7d5d2747-fae9-4cc2-b5ae-bea6b20b634d" />

<img width="1920" height="1020" alt="BREST CANCER" src="https://github.com/user-attachments/assets/015c5d37-eab7-4555-80b5-805348f815c8" />



PredictiX is an intelligent, user-friendly web application that leverages the power of machine learning to provide preliminary predictions for four major chronic diseases: Diabetes, Heart Disease, Parkinson's Disease, and Breast Cancer. This tool is designed to serve as a health awareness platform, empowering users to assess their risk based on relevant medical parameters and encouraging them to seek timely professional medical advice.



📋 Table of Contents
About The Project

Key Features

Built With

Project Structure

Getting Started

Prerequisites

Installation

Usage

Models and Datasets

Disclaimer

Contributing

License

Contact

📖 About The Project

The early detection of chronic diseases can be life-saving. PredictiX was created to make preliminary health assessment more accessible to everyone. By consolidating multiple machine learning models into a single, intuitive platform, this project aims to:

Promote Health Awareness: Provide users with a simple tool to understand potential health risks based on their data.

Leverage AI for Good: Showcase a practical application of machine learning in the healthcare domain.

Encourage Proactive Care: Motivate users who receive a high-risk prediction to consult with healthcare professionals for a formal diagnosis.

The application features a clean, modern interface built with Streamlit, ensuring a seamless user experience across different devices.



✨ Key Features

Four Prediction Modules: Separate, dedicated pages for Diabetes, Heart Disease, Parkinson's, and Breast Cancer prediction.

Multi-Model Approach: Each disease prediction is supported by multiple machine learning models (e.g., Logistic Regression, SVM, Random Forest, XGBoost) to provide a more robust assessment.

Interactive & User-Friendly UI: A sleek, modern dark-themed interface with clear input fields and visually engaging prediction results.

Responsive Design: The application is fully functional on both desktop and mobile browsers.

Detailed Input Fields: Comprehensive forms that capture the necessary medical parameters for each specific disease prediction.



🛠️ Built With
This project is built with a modern Python technology stack:

Python: The core programming language.

Streamlit: The open-source framework for building and deploying the web application.

Pandas: For data manipulation and analysis.

Scikit-learn: For building and evaluating machine learning models.

XGBoost: For implementing the high-performance gradient boosting models.

Pickle: For serializing and saving the trained models.



📂 Project Structure
The project is organized into a clear and maintainable directory structure:

.
├── 📂 Best Features/
│   ├── ... (JSON files with best features for each model)
├── 📂 Datasets/
│   ├── ... (CSV files for each disease)
├── 📂 Models/
│   ├── ... (Saved .sav model files for each disease)
├── 📂 Notebooks/
│   ├── ... (Jupyter notebooks for data exploration and model training)
├── 📂 Preprocessing Files/
│   ├── ... (Saved .pkl files for scalers, encoders, etc.)
├── 📜 app.py                # The main Streamlit application script
├── 📜 logo.png              # Application logo
├── 📜 README.md             # This file
└── 📜 requirements.txt      # Python dependencies



🚀 Getting Started
Follow these steps to set up and run the project on your local machine.

Prerequisites
Make sure you have Python (version 3.9 or higher) and pip installed on your system.

Installation
Clone the Repository

git clone https://github.com/your-username/PredictiX.git
cd PredictiX

Create a Virtual Environment
It's highly recommended to create a virtual environment to keep the project's dependencies isolated.

# For Windows
python -m venv venv
venv\Scripts\activate

# For macOS/Linux
python3 -m venv venv
source venv/bin/activate

Install Dependencies
Install all the required Python packages from the requirements.txt file.

pip install -r requirements.txt


🖥️ Usage
Once you have installed all the dependencies, you can run the Streamlit application with a single command:

streamlit run app.py

This will open a new tab in your default web browser at http://localhost:8501, where you can interact with the application.

Use the sidebar to navigate between the different disease prediction pages.

Fill in all the required medical details in the input form.

Click the "Predict" button to see the result.

You can also view the predictions from other underlying models in the "Advanced Model Predictions" expander.


🧠 Models and Datasets
The machine learning models were trained on the following standard, publicly available datasets from the UCI Machine Learning Repository:

Disease

Dataset

Models Used

Diabetes

Pima Indians Diabetes Database

SVC, Logistic Regression, Random Forest

Heart Disease

Cleveland Heart Disease Dataset

XGBoost, Random Forest, Logistic Regression

Parkinson's

Parkinson's Disease Telemonitoring

KNN, XGBoost, Random Forest

Breast Cancer

Wisconsin Diagnostic Breast Cancer

Logistic Regression, XGBoost, KNN

The Jupyter notebooks containing the complete data preprocessing, model training, and evaluation process can be found in the Notebooks/ directory.


⚠️ Disclaimer
This application is an academic and demonstrative project. The predictions generated by PredictiX are not a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition. Never disregard professional medical advice or delay in seeking it because of something you have read on this site.


🤝 Contributing
Contributions are what make the open-source community such an amazing place to learn, inspire, and create. Any contributions you make are greatly appreciated.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".

Fork the Project

Create your Feature Branch (git checkout -b feature/AmazingFeature)

Commit your Changes (git commit -m 'Add some AmazingFeature')

Push to the Branch (git push origin feature/AmazingFeature)

Open a Pull Request

📄 License
Distributed under the MIT License. See LICENSE for more information.

📧 Contact
[Akash Mandal] - [akashmandal.9490@gmail.com]

Project Link: https://github.com/Akashmandal08/PredictiX
* Installation
* Usage
* Dataset Description
* Technologies Used
* Model Development Process
* Models Used
* Model Evaluation
* Conclusion
* Contributing

# Overview
This web application allows users to select from four different disease prediction systems and get predictions based on the input features. Each prediction system was developed through extensive data analysis and model selection processes, ensuring high accuracy and reliability.

# Installation
To run this project locally, please follow these steps:
1. Clone the repository
2. Navigate to the project directory
3. Install the required dependencies


# Usage
To start the Streamlit web app, run the following command in your terminal: streamlit run main_app.py
This will launch the web app in your default web browser. You can then select the desired disease prediction system from the sidebar and input the required features to get a prediction.

# Dataset Description
1. Diabetes Prediction System

Description: This dataset contains 768 instances of patient data, with 8 features including glucose levels, blood pressure, and insulin levels, used to predict diabetes.

2. Heart Disease Prediction System

Description: This dataset includes 1025 instances with 14 features such as age, sex, chest pain type, and resting blood pressure, used to predict the presence of heart disease.

3. Parkinson Disease Prediction System

Description: This dataset has 195 instances with 22 features including average vocal fundamental frequency, measures of variation in fundamental frequency, and measures of variation in amplitude, used to predict Parkinson's disease.

4. Breast Cancer Prediction System

Description: This dataset contains 569 instances with 30 features such as radius, texture, perimeter, and area, used to predict breast cancer.


# Technologies Used
Programming Language: Python

Web Framework: Streamlit

Machine Learning Libraries: Scikit-learn, XGBoost

Data Analysis and Visualization: Pandas, NumPy, Matplotlib, Seaborn


# Model Development Process
Each disease prediction system was developed through the following steps:

1. Importing the Dependencies

2. Exploratory Data Analysis (EDA)

3. Data Preprocessing
   * Handling missing values
   * Handling outliers
   * Label encoding/One-hot encoding
   * Standardizing the data

4. Model Selection
   * Selected the most common 5 classification models
   * Trained each model and checked cross-validation scores
   * Chose the top 3 models based on cross-validation scores

5. Model Building and Evaluation
   * Selected best features using Recursive Feature Elimination (RFE)
   * Performed hyperparameter tuning using Grid Search CV
   * Built the final model with the best hyperparameters and features
   * Evaluated the model using classification reports


# Models Used
The top 3 models for each disease prediction system are as follows:

1. Diabetes Prediction System
- Support Vector Classifier: Effective in high-dimensional spaces.
- Logistic Regression: Simple and effective binary classification model.
- Random Forest Classifier: Ensemble method that reduces overfitting.

2. Heart Disease Prediction System
- XGBoost: Boosting algorithm known for high performance.
- Random Forest Classifier: Robust and handles missing values well.
- Logistic Regression: Interpretable and performs well with binary classification.

3. Parkinson Disease Prediction System
- K-Nearest Neighbour: Simple algorithm that works well with small datasets.
- XGBoost: Powerful gradient boosting framework.
- Random Forest Classifier: Effective and reduces overfitting.

4. Breast Cancer Prediction System
- Logistic Regression: Highly interpretable and performs well with binary classification.
- XGBoost: Excellent performance with complex datasets.
- K-Nearest Neighbour: Effective with smaller datasets and straightforward implementation.


# Model Evaluation

1. Diabetes Prediction System
Model	Accuracy
- Support Vector Classifier	69.480%
- Logistic Regression	70.129%
- Random Forest Classifier	75.324%

2. Heart Disease Prediction System
Model	Accuracy
- XGBoost	100%
- Random Forest Classifier	100%
- Logistic Regression	88.311%%

3. Parkinson Disease Prediction System
Model	Accuracy
- K-Nearest Neighbour	100%
- XGBoost	92.307%
- Random Forest Classifier	94.871%

4. Breast Cancer Prediction System
Model	Accuracy
- Logistic Regression	97.368%
- XGBoost	97.368%
- K-Nearest Neighbour	96.491%

# Conclusion
This Multiple Disease Prediction System WebApp provides an easy-to-use interface for predicting the likelihood of various diseases based on input features. The models used are well-validated and tuned for high accuracy. The system aims to assist in early diagnosis and better decision-making in healthcare.


# Contributing
Contributions are welcome! If you have any suggestions or improvements, please create a pull request or open an issue.

# Contact
If you have any questions or suggestions, feel free to contact me at akashmandal.9490gmail.com

# 🏠 California House Price Prediction using XGBoost

An end-to-end Machine Learning project that predicts California house prices using **XGBoost Regression**. The project includes data preprocessing, feature engineering, hyperparameter tuning, model evaluation, and deployment with **Streamlit**.

---

## 📌 Project Overview

This project predicts the median house price in California based on housing features such as:

- Longitude
- Latitude
- Housing Median Age
- Total Rooms
- Total Bedrooms
- Population
- Households
- Median Income
- Ocean Proximity

The trained model is deployed as a **Streamlit web application**, allowing users to enter house details and receive a predicted house price instantly.

---

## 🎯 Objectives

- Build an accurate regression model.
- Improve performance using Feature Engineering.
- Tune the model using GridSearchCV.
- Deploy the trained model using Streamlit.

---

## 📂 Dataset

**Dataset:** California Housing Dataset

Features include:

- Longitude
- Latitude
- Housing Median Age
- Total Rooms
- Total Bedrooms
- Population
- Households
- Median Income
- Ocean Proximity

Target:

- Median House Value

---

## 🛠 Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Joblib
- Streamlit

---

## 📊 Machine Learning Pipeline

1. Data Cleaning
2. Exploratory Data Analysis (EDA)
3. Feature Engineering
4. One-Hot Encoding
5. Train-Test Split
6. Feature Scaling (StandardScaler)
7. XGBoost Regression
8. Hyperparameter Tuning (GridSearchCV)
9. Model Evaluation
10. Model Deployment

---

## 🚀 Feature Engineering

The following features were created to improve model performance:

- Rooms per Household
- Bedrooms per Room
- Population per Household

---

## 📈 Model Performance

| Metric | Score |
|---------|--------|
| Model | XGBoost Regressor |
| R² Score | **0.8425** |
| RMSE | **45429.15** |

---

## 🌐 Streamlit Application

The web application allows users to:

- Enter house details
- Select Ocean Proximity
- Predict California House Price instantly

---

## 📁 Project Structure

```
California-House-Price-Prediction/
│
├── app.py
├── House_Price_Prediction.ipynb
├── house_price_xgboost.pkl
├── scaler.pkl
├── housing.csv
├── requirements.txt
└── README.md
```

---

## ▶️ How to Run

### Clone Repository

```bash
git clone https://github.com/trexop2006/California-House-Price-Prediction.git
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Streamlit App

```bash
streamlit run app.py
```

---

## 💡 Future Improvements

- Improve UI with custom CSS
- Deploy on Streamlit Community Cloud
- Add prediction history
- Add data visualization dashboard
- Compare multiple regression algorithms

---

## 👨‍💻 Developer

**Hardik Chaturvedi**

BCA Student | Machine Learning Enthusiast

GitHub: https://github.com/trexop2006

---

## ⭐ If you like this project

Please consider giving this repository a **Star ⭐**.

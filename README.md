#  Real Estate Investment Advisor

### Predicting Property Profitability & Future Value

##  Project Overview

This project is a **Machine Learning-based web application** that helps real estate investors make smarter decisions.

It provides:

*  **Investment Classification** → Whether a property is a *Good Investment* or not
*  **Price Prediction** → Estimated property value after 5 years

The system uses historical housing data and machine learning models to generate **data-driven insights** for property buyers and investors.

---

##  Features

* Data preprocessing and feature engineering
* Exploratory Data Analysis (EDA) with visualizations
* Classification model for investment decision
* Regression model for future price prediction
* Interactive web app using Streamlit
* Model tracking using MLflow

---

##  Technologies Used

* Python
* Pandas, NumPy
* Matplotlib, Seaborn
* Scikit-learn
* XGBoost
* Streamlit
* MLflow

---

## 📊 Dataset

The dataset contains real estate property details such as:

* Location (State, City, Locality)
* Property Type
* BHK, Size (SqFt)
* Price (Lakhs)
* Amenities, Parking, Security
* Nearby Schools & Hospitals
* Property Age and Availability

---

##  Project Workflow

### 1️ Data Preprocessing

* Handled missing values and duplicates
* Encoded categorical variables
* Scaled numerical features
* Created new features like:

  * Price per SqFt
  * Age of Property

---

### 2️ Exploratory Data Analysis (EDA)

* Price distribution analysis
* Size vs Price relationship
* Location-based pricing trends
* Correlation heatmaps

---

### 3️ Feature Engineering

* Created **Good Investment** label
* Generated **Future Price (5 years)** using growth rate

---

### 4️ Model Building

#### 🔹 Classification Model

* Algorithm: Random Forest / XGBoost
* Output: Good Investment (Yes/No)

#### 🔹 Regression Model

* Algorithm: Random Forest Regressor
* Output: Future Price Prediction

---

### 5️ Model Evaluation

**Classification Metrics:**

* Accuracy
* F1 Score
* Confusion Matrix

**Regression Metrics:**

* RMSE
* MAE
* R² Score

---

### 6️ Streamlit Application

* User inputs property details
* Predicts:

  * Investment suitability
  * Future price
* Displays results with insights

---

## 📁 Project Structure

```
real-estate-investment-advisor/
│
├── data/
│   └── india_housing_prices.csv
│
├── notebook/
│   └── real_estate_project.ipynb
│
├── model/
│   ├── classification_model.pkl
│   └── regression_model.pkl
│
├── app.py
├── requirements.txt
└── README.md
```

---

## ▶️ How to Run the Project

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/real-estate-investment-advisor.git
cd real-estate-investment-advisor
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run Streamlit App

```bash
streamlit run app.py
```

---

##  Results

* Achieved high accuracy in classification model
* Reliable future price predictions with low error
* Interactive and user-friendly web application

---

##  Business Use Cases

* Helps investors identify high-return properties
* Supports real estate companies with automated analysis
* Improves decision-making using data insights

---

##  Future Improvements

* Add location-based dynamic growth rates
* Integrate real-time property data APIs
* Enhance UI with advanced visualizations
* Deploy on cloud platforms

---

##  Conclusion

This project demonstrates the use of **Machine Learning in Real Estate Analytics**, enabling smarter investment decisions through predictive modeling and interactive visualization.

---



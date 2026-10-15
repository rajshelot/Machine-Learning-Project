# 📊 Customer Churn Prediction

Predict whether a bank customer is likely to leave the bank using Machine Learning. This project analyzes customer demographics, account details, and banking behavior to build a predictive model that helps identify customers at risk of churning.

---

## 🚀 Project Overview

Customer churn prediction is an important business problem in the banking industry. Retaining existing customers is more cost-effective than acquiring new ones. This project uses customer data to predict whether a customer will exit the bank.

---

## 📂 Dataset

- **Dataset Name:** Churn_Modelling.csv
- **Total Records:** 10,000
- **Features:** 14
- **Target Variable:** `Exited`
  - **0** → Customer Stayed
  - **1** → Customer Left

### Features

| Feature | Description |
|----------|-------------|
| RowNumber | Record Index |
| CustomerId | Unique Customer ID |
| Surname | Customer Surname |
| CreditScore | Customer Credit Score |
| Geography | Country of Customer |
| Gender | Male/Female |
| Age | Customer Age |
| Tenure | Years with Bank |
| Balance | Account Balance |
| NumOfProducts | Number of Bank Products |
| HasCrCard | Has Credit Card (0/1) |
| IsActiveMember | Active Member (0/1) |
| EstimatedSalary | Estimated Salary |
| Exited | Target Variable |

---

## 🎯 Objective

Build a machine learning model that predicts whether a customer will churn based on customer information and banking details.

---

## 🛠 Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Jupyter Notebook

---

## 📊 Project Workflow

1. Load Dataset
2. Data Cleaning
3. Exploratory Data Analysis (EDA)
4. Feature Engineering
5. Data Preprocessing
6. Train-Test Split
7. Model Training
8. Model Evaluation
9. Prediction

---

## 🤖 Machine Learning Models

You can train and compare models such as:

- Logistic Regression
- Decision Tree
- Random Forest
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)
- XGBoost (Optional)

---

## 📈 Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC Score
- Confusion Matrix

---

## 📁 Project Structure

```
Customer-Churn-Prediction/
│
├── Churn_Modelling.csv
├── Customer_Churn_Prediction.ipynb
├── random_forest_churn_model.pkl
├── scaler.pkl
├── app.py
├── requirements.txt
├── README.md
└── images/
```

---

## ⚙️ Installation

Clone the repository

```bash
git clone https://github.com/darshanbhor2006/customer-churn-prediction.git
```

Move into the project directory

```bash
cd customer-churn-prediction
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the application

```bash
streamlit run app.py
```

---

## 📊 Expected Output

The model predicts whether a customer is likely to:

- Stay with the bank
- Leave the bank (Churn)

along with the prediction confidence (if implemented).

---

## 📌 Future Improvements

- Hyperparameter Tuning
- Cross Validation
- Feature Importance Visualization
- Model Deployment using Streamlit
- Cloud Deployment (Render/Streamlit Community Cloud)

---

## 👩‍💻 Author

**Darshan Bhor**

Artificial Intelligence & Data Science Student

---

# 🚗 Automobile Price Prediction System

## 📌 Project Overview

The **Automobile Price Prediction System** is a Machine Learning-based web application that predicts the estimated price of an automobile based on its specifications.

The project uses a **Random Forest Regression** algorithm trained on an Automobile dataset. The trained model is saved as a pickle file and integrated with a **Streamlit** web application for easy and interactive price prediction.

Users can enter automobile specifications such as engine size, horsepower, curb weight, mileage, dimensions, and other features to obtain an estimated automobile price.

---

## 🎯 Objectives

* Predict automobile prices using Machine Learning.
* Analyze automobile specifications that influence price.
* Train and evaluate a regression model.
* Save the trained model using Pickle.
* Build an interactive Streamlit web application.
* Provide quick automobile price predictions through a user-friendly interface.

---

## 🛠️ Technologies Used

* 🐍 Python
* 🧮 Pandas
* 🔢 NumPy
* 🤖 Scikit-learn
* 📊 Matplotlib
* 📈 Seaborn
* 🌐 Streamlit
* 📦 Pickle
* 📓 Jupyter Notebook

---

## 🤖 Machine Learning Model

The project uses:

### Random Forest Regression

`RandomForestRegressor(random_state=42)`

Random Forest Regression combines multiple decision trees to produce a more reliable prediction for continuous numerical values.

The model is trained using automobile specifications and predicts:

**Target Variable: `price`**

---

## 📊 Features Used

The trained model uses the following automobile features:

| Feature             | Description                   |
| ------------------- | ----------------------------- |
| `symboling`         | Automobile risk/symbol rating |
| `wheel-base`        | Wheelbase of the automobile   |
| `length`            | Overall automobile length     |
| `width`             | Overall automobile width      |
| `height`            | Overall automobile height     |
| `curb-weight`       | Weight of the automobile      |
| `num-of-cylinders`  | Number of engine cylinders    |
| `engine-size`       | Engine size                   |
| `compression-ratio` | Engine compression ratio      |
| `horsepower`        | Engine horsepower             |
| `peak-rpm`          | Maximum engine RPM            |
| `city-mpg`          | Mileage in city driving       |
| `highway-mpg`       | Mileage on highway            |

### 🎯 Target

```text
price
```

---

## 📈 Model Performance

The Random Forest Regression model produced the following results in the Jupyter Notebook:

### Test Data

* **Mean Squared Error:** 5,030,141.00
* **Root Mean Squared Error:** 2,242.80
* **Mean Absolute Error:** 1,508.43
* **R² Score:** **0.9355**

### Training Data

* **Mean Squared Error:** 1,532,868.89
* **Root Mean Squared Error:** 1,238.09
* **Mean Absolute Error:** 683.84
* **R² Score:** **0.9733**

The test R² score of approximately **93.55%** indicates that the model explains a large portion of the variation in automobile prices.

---

## 📂 Project Structure

```text
Automobile_Price_Prediction/
│
├── 📓 Linear regression_Auto Data Set.ipynb
│
├── 🌐 app.py
│
├── 🤖 auto_price_model.pkl
│
├── 📋 feature_columns.pkl
│
├── 📊 autos_dataset.csv
│
└── 📄 README.md
```

---

## 📓 Jupyter Notebook

The Jupyter Notebook contains the complete Machine Learning workflow:

1. Import required libraries
2. Load the Automobile dataset
3. Explore the dataset
4. Perform data preprocessing
5. Select input and target variables
6. Split data into training and testing sets
7. Train regression models
8. Evaluate model performance
9. Select Random Forest Regression
10. Save the trained model using Pickle

---

## 💾 Model Saving

After training the Random Forest model, the model is saved using Pickle.

```python
import pickle

with open("auto_price_model.pkl", "wb") as file:
    pickle.dump(rf_reg, file)

with open("feature_columns.pkl", "wb") as file:
    pickle.dump(list(X.columns), file)
```

This generates:

```text
auto_price_model.pkl
feature_columns.pkl
```

---

## 🌐 Streamlit Web Application

The Streamlit application provides an interactive interface where users can enter automobile specifications.

### User Inputs

* Symboling
* Wheel Base
* Length
* Width
* Height
* Curb Weight
* Number of Cylinders
* Engine Size
* Compression Ratio
* Horsepower
* Peak RPM
* City MPG
* Highway MPG

After entering the values, the user clicks:

```text
🚀 Predict Automobile Price
```

The application then displays the predicted automobile price.

Example:

```text
💰 Predicted Automobile Price

$9,580.95
```

The displayed value is generated dynamically by the trained Machine Learning model.

---

## ⚙️ Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/darshanbhor2006/Machine-Learning-Projects/new/main/Car%20Price%20Prediction
```

### Step 2: Open the Project Folder

```bash
cd Automobile_Price_Prediction
```

### Step 3: Install Required Libraries

```bash
pip install pandas numpy scikit-learn matplotlib seaborn streamlit
```

---

## ▶️ Run the Application

Run the following command:

```bash
streamlit run app.py
```

The application will open in your browser.

Usually, Streamlit runs at:

```text
http://localhost:8501
```

---

## 🧪 How to Use

### Step 1

Open the Streamlit application.

### Step 2

Enter the automobile specifications.

### Step 3

Click:

```text
🚀 Predict Automobile Price
```

### Step 4

The trained Random Forest model processes the input.

### Step 5

The estimated automobile price is displayed on the screen.

---

## 🔄 Project Workflow

```text
Automobile Dataset
        ↓
Data Preprocessing
        ↓
Feature Selection
        ↓
Train-Test Split
        ↓
Random Forest Regression
        ↓
Model Evaluation
        ↓
Save Model using Pickle
        ↓
Streamlit Application
        ↓
User Input
        ↓
Price Prediction
```

---

## ⭐ Key Features

* ✅ Machine Learning-based price prediction
* ✅ Random Forest Regression
* ✅ Interactive Streamlit interface
* ✅ 13 automobile input features
* ✅ Pickle-based model deployment
* ✅ Real-time prediction
* ✅ Simple and user-friendly interface
* ✅ Model performance evaluation
* ✅ Prediction details displayed to the user

---

## 🔮 Future Enhancements

* Add automobile brand and model selection.
* Add graphical data visualizations.
* Add prediction history.
* Add database integration.
* Add user authentication.
* Deploy the application online.
* Add multiple Machine Learning algorithms for comparison.
* Add downloadable prediction reports.
* Improve model accuracy using hyperparameter tuning.

---

## 🎓 Applications

This system can be useful for:

* 🚗 Automobile price estimation
* 📊 Market analysis
* 💼 Automobile businesses
* 🏪 Used-car dealerships
* 👨‍💼 Sales and pricing support
* 🎓 Machine Learning education and projects

---

## 👨‍💻 Project Information

**Project:** Automobile Price Prediction System

**Domain:** Machine Learning 

**Model:** Random Forest Regression

**Interface:** Streamlit

**Programming Language:** Python

**Target:** Automobile Price Prediction

---

## 📜 License

This project is created for **educational and academic purposes**.

You are free to modify and improve the project for learning and development.

---

## 🙏 Acknowledgement

This project was developed as a Machine Learning project to demonstrate the complete workflow from **data preprocessing and model training to model deployment using Streamlit**.

---

## ⭐ If You Like This Project

If you find this project useful, consider giving the repository a ⭐ on GitHub.

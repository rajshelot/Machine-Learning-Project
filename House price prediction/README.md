# 🏠 House Price Prediction

## 📌 Project Overview

House Price Prediction is a **Machine Learning project** that predicts the price of a house based on different property features such as location, BHK, size, property age, floor number, nearby facilities, parking, security, and other amenities.

The main goal of this project is to help users get an estimated house price based on the available property information.

---

## 🎯 Objectives

* Predict house prices using Machine Learning.
* Analyze different factors that affect property prices.
* Understand the relationship between house features and prices.
* Build a model that can provide estimated house prices.
* Create a practical project that can be used as a real-world application.

---

## 📂 Dataset Information

The dataset contains **999 records and 23 columns**.

### Dataset Features

| Column                           | Description                            |
| -------------------------------- | -------------------------------------- |
| `ID`                             | Unique ID of the property              |
| `State`                          | State where the property is located    |
| `City`                           | City of the property                   |
| `Locality`                       | Local area of the property             |
| `Property_Type`                  | Type of property                       |
| `BHK`                            | Number of bedrooms, halls and kitchens |
| `Size_in_SqFt`                   | Property size in square feet           |
| `Price_in_Lakhs`                 | House price in lakhs                   |
| `Price_per_SqFt`                 | Price per square foot                  |
| `Year_Built`                     | Year in which the property was built   |
| `Furnished_Status`               | Furnishing status                      |
| `Floor_No`                       | Floor number                           |
| `Total_Floors`                   | Total floors in the building           |
| `Age_of_Property`                | Age of the property                    |
| `Nearby_Schools`                 | Number of nearby schools               |
| `Nearby_Hospitals`               | Number of nearby hospitals             |
| `Public_Transport_Accessibility` | Availability of public transportation  |
| `Parking_Space`                  | Parking availability                   |
| `Security`                       | Security availability                  |
| `Amenities`                      | Available amenities                    |
| `Facing`                         | Direction the property faces           |
| `Owner_Type`                     | Type of property owner                 |
| `Availability_Status`            | Property availability status           |

---

## 🛠️ Technologies Used

* **Python**
* **Pandas**
* **NumPy**
* **Matplotlib**
* **Seaborn**
* **Scikit-learn**
* **Flask / Streamlit** *(if used for deployment)*
* **Jupyter Notebook**
* **Git & GitHub**

---

## 🔄 Project Workflow

```text
Dataset
   ↓
Data Collection
   ↓
Data Cleaning
   ↓
Exploratory Data Analysis
   ↓
Feature Engineering
   ↓
Categorical Encoding
   ↓
Train-Test Split
   ↓
Model Training
   ↓
Model Evaluation
   ↓
House Price Prediction
   ↓
Web Application
```

---

## 🧹 Data Preprocessing

The dataset is prepared before training the Machine Learning model.

The preprocessing steps include:

1. Loading the dataset.
2. Checking the dataset structure.
3. Handling missing values.
4. Checking duplicate records.
5. Separating numerical and categorical features.
6. Encoding categorical variables.
7. Selecting important features.
8. Splitting the dataset into training and testing sets.

---

## 🤖 Machine Learning

This project uses **Supervised Machine Learning** because the target variable, `Price_in_Lakhs`, is already available in the dataset.

The model learns the relationship between property features and house prices.

Possible regression algorithms include:

* Linear Regression
* Decision Tree Regressor
* Random Forest Regressor
* Gradient Boosting Regressor

The best-performing model can be selected based on evaluation metrics.

---

## 📊 Model Evaluation

The model can be evaluated using regression metrics such as:

### Mean Absolute Error (MAE)

Measures the average absolute difference between actual and predicted prices.

### Mean Squared Error (MSE)

Measures the average squared difference between actual and predicted prices.

### Root Mean Squared Error (RMSE)

Shows the prediction error in the same unit as the target variable.

### R² Score

Shows how well the model explains the variation in house prices.

---

## 📈 Expected Output

The system takes property details such as:

```text
State
City
Locality
Property Type
BHK
Size
Year Built
Floor Number
Furnishing Status
Parking
Security
Amenities
```

and predicts:

```text
Estimated House Price
```

For example:

```text
Input:
BHK: 3
Size: 1500 SqFt
City: Pune
Property Type: Apartment

Output:
Predicted House Price: ₹XX Lakhs
```

---

## 📁 Project Structure

House Price Prediction/
│
├── .ipynb_checkpoints/
├── app.py
├── encoder.pkl
├── features.pkl
├── House Price Prediction Dataset.csv
├── House Price Prediction.ipynb
├── house_price_model.pkl
└── vectorizer.pkl
```

> The exact project structure can be changed according to the files used in your project.

---

## 🚀 How to Run the Project

### 1. Clone the Repository

```bash
git clone YOUR_GITHUB_REPOSITORY_LINK
```

### 2. Open the Project Folder

```bash
cd House-Price-Prediction
```

### 3. Create a Virtual Environment

```bash
python -m venv venv
```

### 4. Activate the Virtual Environment

For Windows:

```bash
venv\Scripts\activate
```

### 5. Install Required Libraries

```bash
pip install -r requirements.txt
```

### 6. Run the Application

If using Flask:

```bash
python app.py
```

If using Streamlit:

```bash
streamlit run app.py
```

---

## 💡 Real-World Use Case

House prices depend on many factors such as location, property size, number of bedrooms, property age, nearby facilities, parking, and amenities.

This project uses Machine Learning to analyze these factors and provide an estimated property price.

It can be useful for:

* 🏠 Home Buyers
* 🏢 Real Estate Companies
* 💼 Property Agents
* 📊 Real Estate Analysts
* 👨‍💻 Students learning Machine Learning

---

## 🔮 Future Improvements

* Improve prediction accuracy using advanced ML algorithms.
* Add more real-world housing data.
* Add interactive visualizations.
* Deploy the application online.
* Add a map-based property search.
* Add price comparison between different cities.
* Add automatic model retraining with new data.

---

## 👨‍💻 Author

**Darshan Bhor**

Machine Learning | Python | Data Science | NLP

---

## ⭐ Project Highlights

* ✅ Real-world Machine Learning project
* ✅ House price prediction
* ✅ Regression problem
* ✅ Data preprocessing
* ✅ Exploratory Data Analysis
* ✅ Feature engineering
* ✅ Model training and evaluation
* ✅ Web application deployment

---

## 📜 License

This project is created for **educational and learning purposes**.

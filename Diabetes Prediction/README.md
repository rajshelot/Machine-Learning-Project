# 🩺 Diabetes Prediction & Analytics Dashboard

An interactive **Diabetes Prediction and Analytics Dashboard** built using Machine Learning and deployed as a static web application using **Netlify**.

The project uses a **Decision Tree Classifier** to predict whether a patient is likely to be diabetic based on clinical parameters such as glucose level, blood pressure, BMI, age, insulin level, and other health-related features.

## 🌐 Live Demo

🚀 **Deployed Website:**
https://dibetesprediction.netlify.app/

---

## 📌 Project Overview

Diabetes is a common health condition that can be influenced by several clinical and lifestyle factors. This project demonstrates how Machine Learning can be used to analyze diabetes-related data and provide a prediction based on patient health parameters.

The original project was developed as an interactive Streamlit dashboard using a **Decision Tree Classifier**. The Netlify version converts the dashboard into a **static HTML, CSS, and JavaScript application**, allowing it to run directly in a web browser without requiring a Python server.

---

## 🎯 Objectives

* Predict diabetes risk using Machine Learning.
* Analyze important diabetes-related health parameters.
* Provide an interactive prediction interface.
* Visualize the dataset using an EDA dashboard.
* Evaluate Decision Tree model performance.
* Deploy the application as a static website using Netlify.
* Create a user-friendly and responsive dashboard.

---

## 🧠 Machine Learning Model

### Decision Tree Classifier

The project uses a **Decision Tree Classifier** with the following configuration:

```python
DecisionTreeClassifier(
    criterion="gini",
    random_state=42
)
```

The dataset is divided into:

* **80% Training Data**
* **20% Testing Data**

The model is trained using the following input features:

| Feature                  | Description                  |
| ------------------------ | ---------------------------- |
| Pregnancies              | Number of pregnancies        |
| Glucose                  | Plasma glucose concentration |
| BloodPressure            | Blood pressure measurement   |
| SkinThickness            | Skin fold thickness          |
| Insulin                  | Insulin level                |
| BMI                      | Body Mass Index              |
| DiabetesPedigreeFunction | Diabetes pedigree function   |
| Age                      | Patient age                  |

### Model Result

**Test Accuracy: approximately 74.68%**

> Accuracy can vary if the dataset or training configuration is changed.

---

## 📊 Dashboard Features

### 🏠 1. Project Overview & Prediction

The dashboard provides:

* Total number of records
* Number of features
* Model test accuracy
* Patient input form
* Real-time diabetes prediction
* Prediction confidence
* Dataset preview

Users can enter:

* Pregnancies
* Glucose
* Blood Pressure
* Skin Thickness
* Insulin
* BMI
* Diabetes Pedigree Function
* Age

and click:

**🚀 Run Diagnostic Prediction**

---

### 📈 2. Executive EDA Dashboard

The dashboard provides visual analysis including:

#### Target Distribution

Shows the distribution between:

* Non-Diabetic
* Diabetic

#### Glucose vs BMI

A scatter plot showing the relationship between:

* Glucose
* BMI
* Diabetes Outcome

#### Feature Correlation Heatmap

Displays correlations between the major clinical features and the target variable.

---

### 🤖 3. Model Performance

The model performance section displays:

* Training Accuracy
* Testing Accuracy
* Confusion Matrix
* Classification Report
* Precision
* Recall
* F1-score
* Test Support

---

### ℹ️ 4. About Section

Contains information about:

* Project technology stack
* Machine Learning algorithm
* Data processing
* Visualization
* Developer profile
* Project architecture

---

## 🛠️ Technologies Used

### Machine Learning

* Python
* Scikit-learn
* Decision Tree Classifier

### Data Processing

* Pandas
* NumPy

### Data Visualization

* Matplotlib
* Seaborn

### Web Technologies

* HTML5
* CSS3
* JavaScript

### Deployment

* Netlify
* GitHub

---

## 📂 Project Structure

```text
Diabetes-Prediction/
│
├── .ipynb_checkpoints/
│
├── app.py
│
├── Decision Tree Diabetes Data.ipynb
│
├── diabetes.csv
│
├── Diabetes_model.pkl
│
└── README.md
```

The main Netlify deployment uses:

```text
index.html
```

The prediction logic is embedded directly into JavaScript, so the static deployment does not require:

```text
app.py
Python
Streamlit
Flask
```

---

## ⚙️ How the Application Works

```text
Patient Health Parameters
          ↓
     Input Validation
          ↓
   Decision Tree Model
          ↓
    Prediction Result
       ↙       ↘
Non-Diabetic   Diabetic
          ↓
    Confidence Score
```

---

## 🌐 Netlify Deployment

The project is designed for simple static deployment.

### Step 1 — Clone Repository

```bash
https://github.com/darshanbhor2006/Machine-Learning-Projects/new/main/Diabetes%20Prediction
```

### Step 2 — Open Project

```bash
cd Diabetes-Prediction
```

### Step 3 — Deploy

Upload the project folder to Netlify using **Deploy manually**, or connect the GitHub repository to Netlify.

Since the project contains a static `index.html`, no build command is required.

### Netlify Settings

```text
Build command: None
Publish directory: .
```

---

## 💡 Why Static Deployment?

The original application uses Streamlit and Python. However, Netlify is primarily suitable for static websites and frontend applications.

Therefore, this version embeds the Decision Tree prediction logic into JavaScript.

### Original Architecture

```text
Python
  ↓
Pandas
  ↓
Scikit-learn
  ↓
Decision Tree
  ↓
Streamlit
```

### Netlify Architecture

```text
HTML
  ↓
CSS
  ↓
JavaScript
  ↓
Embedded Decision Tree Logic
  ↓
Prediction Result
```

This allows the application to run completely inside the user's browser.

---

## 🎨 Dashboard Highlights

* 🩺 Medical-themed interface
* 🎨 Colorful gradient design
* 📊 Interactive analytics sections
* 📱 Responsive layout
* 🧠 Machine Learning prediction
* ⚡ Fast static deployment
* 🌐 No backend server required

---

## ⚠️ Disclaimer

This project is created for **educational and demonstration purposes only**.

The prediction generated by this application should **not be considered a medical diagnosis**. Users should consult qualified healthcare professionals for medical advice, diagnosis, or treatment.

---

## 👨‍💻 Developer

**Darshan Bhor**

**Role:** Machine Learning Developer

📧 Email: darshanbhor2006@gmail.com

---

## ⭐ Future Improvements

* Add additional Machine Learning algorithms.
* Compare Decision Tree, Random Forest, Logistic Regression, and KNN.
* Add feature importance visualization.
* Improve model accuracy through hyperparameter tuning.
* Add more advanced interactive charts.
* Add patient history tracking.
* Connect the frontend to a Python/FastAPI backend.
* Add authentication and database support.
* Deploy a full-stack version.

---

## ⭐ Support

If you find this project useful, consider giving the repository a ⭐ **Star** on GitHub.

---
📜 License

This project is developed for educational, learning, and portfolio purposes.

© 2026 Darshan Bhor. All Rights Reserved.

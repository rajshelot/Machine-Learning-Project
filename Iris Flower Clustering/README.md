# 🌸 Iris K-Means Clustering

## 📌 Project Overview

This project demonstrates **Unsupervised Machine Learning using the K-Means Clustering algorithm** on the famous Iris dataset.

The project groups Iris flowers into clusters based on four numerical features:

* Sepal Length
* Sepal Width
* Petal Length
* Petal Width

The project includes a **Jupyter Notebook for data analysis and model development** and a **Streamlit web application** for interactive cluster prediction and visualization.

---

## 🎯 Objectives

* Apply K-Means Clustering to the Iris dataset.
* Determine a suitable number of clusters.
* Use the **Elbow Method** to analyze clustering performance.
* Use the **Silhouette Score** for cluster evaluation.
* Standardize the input features before clustering.
* Visualize the resulting clusters.
* Build an interactive Streamlit application.
* Save and reuse the trained K-Means model.

---

## 🚀 Key Features

* 🌸 Iris dataset analysis
* 🧹 Data preprocessing
* 📊 Exploratory Data Analysis
* 📏 Feature scaling using StandardScaler
* 🤖 K-Means Clustering
* 📉 Elbow Method
* 📐 Silhouette Score
* 🎯 Cluster prediction
* 📊 PCA-based cluster visualization
* 🎯 Cluster center visualization
* 💻 Interactive Streamlit dashboard
* 💾 Saved trained model
* 🔍 Interactive Iris flower input

---

## 🧠 Machine Learning Algorithm

### K-Means Clustering

K-Means is an **unsupervised machine learning algorithm** used to divide data points into a predefined number of clusters.

The algorithm works by:

1. Selecting initial cluster centroids.
2. Assigning each data point to the nearest centroid.
3. Updating the centroid of each cluster.
4. Repeating the process until the centroids stabilize.

For this project, the final clustering configuration uses:

```text
Number of Clusters (K) = 3
```

The Iris dataset contains three known flower species, while the K-Means model itself is trained without using the species labels.

---

## 📊 Dataset

The project uses the **Iris dataset**.

The dataset contains four numerical features:

| Feature      | Description               |
| ------------ | ------------------------- |
| Sepal Length | Length of the sepal in cm |
| Sepal Width  | Width of the sepal in cm  |
| Petal Length | Length of the petal in cm |
| Petal Width  | Width of the petal in cm  |

The original Iris dataset contains three species:

```text
Setosa
Versicolor
Virginica
```

The species labels are not used during K-Means training because K-Means is an unsupervised learning algorithm.

---

## 🔄 Machine Learning Workflow

```text
Iris Dataset
     ↓
Data Loading
     ↓
Data Exploration
     ↓
Feature Selection
     ↓
Feature Scaling
     ↓
Find Optimal K
     ↓
Elbow Method
     ↓
Silhouette Score
     ↓
K-Means Clustering
     ↓
Cluster Evaluation
     ↓
PCA Visualization
     ↓
Save Model
     ↓
Streamlit Application
     ↓
Interactive Cluster Prediction
```

---

## 📈 Elbow Method

The **Elbow Method** is used to help determine a suitable value of K.

It calculates the Within-Cluster Sum of Squares / inertia for different numbers of clusters.

The value of K is selected by examining where adding additional clusters provides diminishing improvement.

Example:

```text
K = 2
K = 3
K = 4
K = 5
...
```

The project uses this analysis along with the Silhouette Score to select the clustering configuration.

---

## 📐 Silhouette Score

The **Silhouette Score** measures how well data points fit within their assigned clusters.

The score considers:

* Similarity of a point to its own cluster.
* Separation of the point from other clusters.

A higher Silhouette Score indicates better separation between clusters.

---

## 🎯 Cluster Prediction

The Streamlit application allows users to enter:

```text
Sepal Length
Sepal Width
Petal Length
Petal Width
```

The values are standardized using the same scaler used during model training.

The trained K-Means model then predicts the corresponding cluster.

Example:

```text
Input:
Sepal Length = 5.1
Sepal Width  = 3.5
Petal Length = 1.4
Petal Width  = 0.2

Output:
Predicted Cluster: Cluster 0
```

The cluster number represents the group assigned by K-Means and does not inherently correspond to a particular Iris species.

---

## 📊 PCA Visualization

Since the Iris dataset contains four features, **Principal Component Analysis (PCA)** is used to reduce the feature space to two dimensions for visualization.

The Streamlit application displays:

* Principal Component 1
* Principal Component 2
* Cluster assignments

This provides a visual representation of the separation between the clusters.

---

## 💻 Streamlit Application

The `app.py` file provides an interactive web application.

### Application Features

* 🌸 Iris flower input form
* 🔍 Cluster prediction
* 📊 Input data display
* 🎯 Cluster information
* 📍 Cluster centers
* 📈 PCA cluster visualization
* 🤖 K-Means model information
* 📋 Feature descriptions

---

## 📁 Project Structure

```text
Iris Flower Clustering/
│
├── app.py
│
├── Iris_KMeans_Clustering.ipynb
│
├── iris_kmeans_model.pkl
│
└── README.md
```

---

## 🛠️ Technologies Used

| Technology          | Purpose              |
| ------------------- | -------------------- |
| 🐍 Python           | Programming language |
| 🐼 Pandas           | Data processing      |
| 🔢 NumPy            | Numerical operations |
| 🤖 Scikit-learn     | Machine Learning     |
| 📊 Matplotlib       | Data visualization   |
| 📈 Streamlit        | Web application      |
| 📓 Jupyter Notebook | Model development    |
| 💾 Pickle           | Model serialization  |

---

## ⚙️ Installation

### Step 1: Clone the Repository

```bash
https://github.com/darshanbhor2006
```

### Step 2: Open the Project Folder

```bash
cd Iris Flower Clustering
```

### Step 3: Create a Virtual Environment

```bash
python -m venv venv
```

### Step 4: Activate the Virtual Environment

For Windows:

```bash
venv\Scripts\activate
```

### Step 5: Install Required Libraries

```bash
pip install streamlit pandas numpy matplotlib scikit-learn
```

---

## ▶️ Run the Streamlit Application

Run the following command:

```bash
streamlit run app.py
```

The application will open automatically in your web browser.

If it does not open automatically, visit:

```text
http://localhost:8501
```

---

## 📓 Run the Jupyter Notebook

Install Jupyter if required:

```bash
pip install notebook
```

Run:

```bash
jupyter notebook
```

Then open:

```text
Iris_KMeans_Clustering.ipynb
```

Run the notebook cells from top to bottom.

---

## 💾 Saved Model

The trained model is stored in:

```text
iris_kmeans_model.pkl
```

The saved file contains:

```text
K-Means Model
Scaler
Feature Names
```

The Streamlit application loads this file to perform cluster predictions.

---

## 📋 Example Input

```text
Sepal Length : 5.1 cm
Sepal Width  : 3.5 cm
Petal Length : 1.4 cm
Petal Width  : 0.2 cm
```

The application processes the input and displays the predicted cluster.

---

## 🎓 Learning Outcomes

Through this project, I gained practical experience in:

* Unsupervised Machine Learning
* K-Means Clustering
* Data preprocessing
* Feature scaling
* Elbow Method
* Silhouette Score
* PCA visualization
* Model serialization
* Streamlit application development
* Interactive Machine Learning dashboards

---

## 🔮 Future Enhancements

* Add interactive dataset upload.
* Add dynamic K selection.
* Display the Silhouette Score directly in Streamlit.
* Add cluster distribution charts.
* Add 3D PCA visualization.
* Add downloadable prediction results.
* Deploy the Streamlit application online.

---

## ⚠️ Note

This project is developed for **educational and demonstration purposes** to understand K-Means clustering and unsupervised machine learning.

The cluster numbers generated by K-Means are arbitrary labels and should not automatically be interpreted as specific Iris species without separate evaluation against the known labels.

---

## 👨‍💻 Author

**Darshan Bhor**

Machine Learning | Data Science | Python | Streamlit

---

## ⭐ Support

If you find this project useful, consider giving the repository a ⭐ on GitHub.

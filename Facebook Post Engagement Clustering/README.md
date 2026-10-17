# 🌟 Facebook Post Engagement Clustering

## 📌 Project Overview

**Facebook Post Engagement Clustering** is a Machine Learning project that analyzes Facebook Live post engagement data and groups posts into different behavioral segments using the **K-Means Clustering** algorithm.

The project uses engagement metrics such as reactions, comments, shares, likes, loves, wows, hahas, sads, and angrys to identify similar patterns among posts.

An interactive **Streamlit Dashboard** is provided to explore the dataset, visualize clusters, analyze engagement patterns, and predict the cluster of a new Facebook post.

---

## 🎯 Objectives

* Analyze Facebook post engagement data.
* Identify groups of posts with similar engagement behavior.
* Apply **K-Means Unsupervised Machine Learning**.
* Use data preprocessing and feature scaling for clustering.
* Visualize cluster distributions using charts.
* Apply the **Elbow Method** to analyze the suitable number of clusters.
* Use **PCA** for two-dimensional cluster visualization.
* Provide an interactive Streamlit dashboard.
* Predict the cluster group for new post engagement data.

---

## 📊 Dataset

The project uses the **Facebook Live Sellers in Thailand** dataset.

The main engagement features used for clustering are:

* `num_reactions`
* `num_comments`
* `num_shares`
* `num_likes`
* `num_loves`
* `num_wows`
* `num_hahas`
* `num_sads`
* `num_angrys`

The columns containing identifiers, dates, and unnecessary information are removed during preprocessing.

---

## 🤖 Machine Learning Approach

### 1. Data Preprocessing

The dataset is cleaned by:

* Removing unnecessary columns.
* Handling missing values using median values.
* Selecting relevant engagement features.

### 2. Log Transformation

Engagement values can have highly skewed distributions. Therefore, the project applies:

```python
np.log1p(X)
```

to reduce the effect of extreme values.

### 3. Feature Scaling

The transformed features are standardized using:

```python
StandardScaler()
```

This ensures that features with different numerical ranges contribute appropriately to the clustering process.

### 4. K-Means Clustering

The **K-Means** algorithm groups posts according to similarities in their engagement patterns.

The trained model is saved as:

```text
kmeans_model.pkl
```

### 5. PCA Visualization

**Principal Component Analysis (PCA)** is used to reduce the feature space to two dimensions so that cluster patterns can be visualized using a scatter plot.

---

## 📈 Dashboard Features

The Streamlit dashboard includes the following sections:

### 🎯 Prediction Studio

Users can enter engagement values for a new Facebook post, including:

* Reactions
* Comments
* Shares
* Likes
* Loves
* Wows
* Hahas
* Sads
* Angrys

The application applies the same preprocessing pipeline used during model training and predicts the corresponding cluster.

### 📊 Overview & Metrics

Displays important dataset information such as:

* Total Posts Evaluated
* Number of Clusters
* Average Reactions
* Average Shares
* Post Count by Cluster
* Average Reactions by Cluster

### 📉 Elbow & PCA Visuals

Provides:

* Elbow Method / Inertia Curve
* PCA 2D Cluster Visualization

These visualizations help understand the clustering structure of the dataset.

### 🔍 Raw Data Explorer

Allows users to:

* View the processed dataset.
* Filter posts according to Cluster ID.
* Explore cluster-specific records.

### ℹ️ About Us

Provides information about the project, dataset, technologies, and machine learning approach.

---

## 🗂️ Project Structure

```text
Facebook_Post_Engagement_Clustering/
│
├── K_Means_Clustering.ipynb
├── Live.csv
├── app.py
├── README.md
│
├── kmeans_model.pkl
├── scaler.pkl
├── features.pkl
└── optimal_k.pkl
```

### 📄 File Description

| File                       | Description                                                                |
| -------------------------- | -------------------------------------------------------------------------- |
| `K_Means_Clustering.ipynb` | Data analysis, preprocessing, visualization, and K-Means model development |
| `Live.csv`                 | Facebook Live engagement dataset                                           |
| `app.py`                   | Streamlit dashboard application                                            |
| `kmeans_model.pkl`         | Saved K-Means clustering model                                             |
| `scaler.pkl`               | Saved StandardScaler used during preprocessing                             |
| `features.pkl`             | Saved feature names used by the model                                      |
| `optimal_k.pkl`            | Saved optimal number of clusters                                           |
| `README.md`                | Project documentation                                                      |

---

## 🛠️ Technologies Used

* 🐍 Python
* 📊 Pandas
* 🔢 NumPy
* 🤖 Scikit-learn
* 📈 Matplotlib
* 🎨 Seaborn
* 🌐 Streamlit
* 📓 Jupyter Notebook
* 💾 Pickle

---

## 🔄 Project Workflow

```text
Facebook Live Dataset
        ↓
Data Cleaning
        ↓
Feature Selection
        ↓
Missing Value Handling
        ↓
Log Transformation
        ↓
Standard Scaling
        ↓
K-Means Clustering
        ↓
Cluster Analysis
        ↓
PCA Visualization
        ↓
Pickle Model Saving
        ↓
Streamlit Dashboard
        ↓
New Post Cluster Prediction
```

---

## 🚀 How to Run the Project

### Step 1: Open the Project Folder

Open the project folder in **VS Code** or Command Prompt.

```text
Facebook_Post_Engagement_Clustering
```

### Step 2: Install Required Libraries

Run:

```bash
pip install streamlit pandas numpy matplotlib seaborn scikit-learn
```

### Step 3: Run the Streamlit Application

Run:

```bash
streamlit run app.py
```

### Step 4: Open the Dashboard

After running the command, Streamlit will provide a local URL, usually:

```text
http://localhost:8501
```

Open this URL in your web browser.

---

## 🔮 Prediction Process

When a user enters new engagement values:

```text
New Post Engagement Data
        ↓
Log Transformation
        ↓
Saved StandardScaler
        ↓
K-Means Model
        ↓
Predicted Cluster ID
```

The dashboard then displays the predicted **Cluster ID** and the engagement characteristics of that cluster.

---

## 📌 Example Input

```text
Num Reactions : 150
Num Comments  : 25
Num Shares    : 5
Num Likes     : 120
Num Loves     : 20
Num Wows      : 5
Num Hahas     : 3
Num Sads      : 1
Num Angrys    : 1
```

The application processes these values and assigns the post to a corresponding K-Means cluster.

---

## 💡 Key Insights

The project helps identify different engagement patterns among Facebook posts by grouping posts with similar interaction characteristics.

The clustering results can be used to explore:

* High-engagement posts
* Low-engagement posts
* Similar engagement behaviors
* Reaction patterns
* Comment and share patterns
* Differences between cluster groups

---

## 🎓 Learning Outcomes

Through this project, the following concepts are demonstrated:

* Unsupervised Machine Learning
* K-Means Clustering
* Feature Engineering
* Log Transformation
* Feature Scaling
* Elbow Method
* PCA
* Data Visualization
* Model Serialization using Pickle
* Streamlit Application Development
* Interactive Machine Learning Dashboard

---

## 🔧 Future Enhancements

Possible future improvements include:

* Adding automatic optimal-cluster selection.
* Adding additional engagement visualizations.
* Adding cluster descriptions based on engagement behavior.
* Adding downloadable cluster reports.
* Adding interactive filters for individual engagement metrics.
* Deploying the Streamlit application online.

---

## 👨‍💻 Project

**Project Title:** Facebook Post Engagement Clustering

**Machine Learning Algorithm:** K-Means Clustering

**Application:** Streamlit Interactive Dashboard

**Dataset:** Facebook Live Sellers in Thailand

---

## ⭐ Conclusion

The **Facebook Post Engagement Clustering** project demonstrates how unsupervised machine learning can be used to discover meaningful patterns in social media engagement data.

By combining **K-Means clustering, data preprocessing, PCA visualization, and Streamlit**, the project provides an interactive platform for analyzing Facebook post engagement and predicting cluster assignments for new posts.

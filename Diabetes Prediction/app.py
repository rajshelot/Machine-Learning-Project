import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Diabetes Prediction Dashboard",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- MODERN VIBRANT & COLORFUL DASHBOARD STYLING WITH MEDICAL BACKGROUND ---
st.markdown("""
    <style> 
    /* Vibrant Sidebar background */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e1b4b 0%, #312e81 100%);
    }
    
    section[data-testid="stSidebar"] * {
        color: #ffffff !important;
    }

    /* Larger Shape Fill Header Banners - Center Aligned & Increased Size */
    .dashboard-title-banner {
        background: linear-gradient(90deg, #3b82f6 0%, #8b5cf6 100%);
        color: white;
        padding: 30px 40px;
        border-radius: 14px;
        box-shadow: 0 6px 20px rgba(59, 130, 246, 0.45);
        margin-bottom: 25px;
        text-align: center;
    }
    .dashboard-title-banner h1 {
        color: white !important;
        margin: 0;
        font-size: 42px !important;
        font-weight: 800;
        letter-spacing: 0.5px;
    }

    /* Glass Card Style with Dark, Highly Visible Text */
    .glass-card {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(12px);
        padding: 24px;
        border-radius: 15px;
        border: 1px solid rgba(255, 255, 255, 0.6);
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.08);
        margin-bottom: 20px;
        color: #1e293b !important;
    }
    .glass-card h3, .glass-card p, .glass-card li, .glass-card b {
        color: #1e293b !important;
    }

    /* Colorful Gradient Metric Cards */
    div.metric-card-1 {
        background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
        padding: 20px;
        border-radius: 12px;
        color: white;
        box-shadow: 0 10px 15px -3px rgba(59, 130, 246, 0.4);
        text-align: center;
        margin-bottom: 15px;
    }
    div.metric-card-2 {
        background: linear-gradient(135deg, #10b981 0%, #047857 100%);
        padding: 20px;
        border-radius: 12px;
        color: white;
        box-shadow: 0 10px 15px -3px rgba(16, 185, 129, 0.4);
        text-align: center;
        margin-bottom: 15px;
    }
    div.metric-card-3 {
        background: linear-gradient(135deg, #8b5cf6 0%, #6d28d9 100%);
        padding: 20px;
        border-radius: 12px;
        color: white;
        box-shadow: 0 10px 15px -3px rgba(139, 92, 246, 0.4);
        text-align: center;
        margin-bottom: 15px;
    }
    
    div.metric-card-1 h3, div.metric-card-2 h3, div.metric-card-3 h3 {
        margin: 0;
        font-size: 15px;
        color: #e0e7ff;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    div.metric-card-1 h2, div.metric-card-2 h2, div.metric-card-3 h2 {
        margin: 8px 0 0 0;
        font-size: 32px;
        color: #ffffff;
        font-weight: 700;
    }

    /* Colorful Containers for Charts / Dashboard Tiles */
    .dashboard-tile {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(12px);
        padding: 20px;
        border-radius: 15px;
        border: 1px solid rgba(255, 255, 255, 0.5);
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.08);
        margin-bottom: 20px;
        color: #1e293b !important;
    }

    /* Colorful Action Buttons */
    .stButton>button {
        background: linear-gradient(90deg, #ec4899 0%, #8b5cf6 100%);
        color: white;
        font-weight: bold;
        border: none;
        border-radius: 8px;
        padding: 12px 28px;
        font-size: 16px;
        box-shadow: 0 4px 12px rgba(236, 72, 153, 0.4);
        transition: 0.3s ease;
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #db2777 0%, #7c3aed 100%);
        color: #fff;
        box-shadow: 0 6px 16px rgba(219, 39, 119, 0.6);
    }
    
    /* Standard Headers with Increased Size */
    h2 {
        color: #1e293b;
        font-size: 24px !important;
        font-weight: 700;
    }
    h3 {
        color: #1e293b;
        font-size: 20px !important;
        font-weight: 700;
    }
    p, span, label {
        color: #1e293b !important;
    }
    </style>
""", unsafe_allow_html=True)

# --- CACHED DATA LOADING & MODEL TRAINING ---
@st.cache_data
def load_data():
    df = pd.read_csv('diabetes.csv')
    return df

@st.cache_resource
def train_model(df):
    X = df.drop(['Outcome'], axis=1)
    y = df['Outcome']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    
    model = DecisionTreeClassifier(criterion='gini', random_state=42)
    model.fit(X_train, y_train)
    
    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_acc = accuracy_score(y_test, model.predict(X_test))
    
    return model, X_train, X_test, y_train, y_test, train_acc, test_acc

try:
    df = load_data()
    model, X_train, X_test, y_train, y_test, train_acc, test_acc = train_model(df)
except FileNotFoundError:
    st.error("🚨 Error: 'diabetes.csv' file not found! Please place it in the same directory as app.py.")
    st.stop()

# --- SIDEBAR NAVIGATION ---
st.sidebar.title("📊 Navigation Panel")
st.sidebar.markdown("---")
app_mode = st.sidebar.radio("Choose Section", [
    "🏠 Project Overview & Prediction", 
    "📈 Executive EDA Dashboard", 
    "🤖 Model Performance",
    "ℹ️ About"
])

st.sidebar.markdown("---")
st.sidebar.info("💡 **Tip:** Type health parameters manually to evaluate risk instantly.")

# ==========================================
# 1. PROJECT OVERVIEW & LIVE PREDICTION SECTION
# ==========================================
if app_mode == "🏠 Project Overview & Prediction":
    st.markdown("""
        <div class="dashboard-title-banner">
            <h1>🩺 Diabetes Prediction & Analytics Dashboard</h1>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("Welcome to the interactive machine learning web application built to analyze diabetes risk factors and predict clinical outcomes.")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
            <div class="metric-card-1">
                <h3>Total Records</h3>
                <h2>{df.shape[0]}</h2>
            </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
            <div class="metric-card-2">
                <h3>Features Analyzed</h3>
                <h2>{df.shape[1] - 1}</h2>
            </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
            <div class="metric-card-3">
                <h3>Model Test Accuracy</h3>
                <h2>{test_acc * 100:.2f}%</h2>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("🎯 Real-Time Diabetes Risk Predictor")
    st.markdown("Type the patient vitals below to evaluate diabetes risk instantly using the trained Decision Tree model.")

    with st.form("prediction_form"):
        p_col1, p_col2, p_col3 = st.columns(3)
        
        with p_col1:
            pregnancies = st.text_input("Pregnancies", value="1")
            glucose = st.text_input("Glucose Level", value="120")
            bp = st.text_input("Blood Pressure", value="70")
            
        with p_col2:
            skin_thickness = st.text_input("Skin Thickness", value="20")
            insulin = st.text_input("Insulin Level", value="80")
            bmi = st.text_input("BMI", value="25.0")
            
        with p_col3:
            dpf = st.text_input("Diabetes Pedigree Function", value="0.5")
            age = st.text_input("Age", value="33")

        submit_button = st.form_submit_button(label="🚀 Run Diagnostic Prediction")

    if submit_button:
        try:
            input_data = np.array([[
                float(pregnancies), float(glucose), float(bp), 
                float(skin_thickness), float(insulin), float(bmi), 
                float(dpf), float(age)
            ]])
            prediction = model.predict(input_data)
            prediction_proba = model.predict_proba(input_data)

            st.markdown("---")
            if prediction[0] == 1:
                st.error(f"⚠️ **High Risk Detected:** The model predicts the patient is **Diabetic** (Confidence: {prediction_proba[0][1]*100:.2f}%)")
            else:
                st.success(f"✅ **Low Risk Detected:** The model predicts the patient is **Non-Diabetic** (Confidence: {prediction_proba[0][0]*100:.2f}%)")
        except ValueError:
            st.error("🚨 Please enter valid numeric values for all health parameters.")

    with st.expander("🔍 Preview Raw Dataset"):
        st.dataframe(df.head(10), use_container_width=True)

# ==========================================
# 2. EXECUTIVE EDA DASHBOARD SECTION
# ==========================================
elif app_mode == "📈 Executive EDA Dashboard":
    st.markdown("""
        <div class="dashboard-title-banner">
            <h1>📊 Executive EDA Dashboard</h1>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("Visual analytics highlighting key clinical parameters affecting diabetes diagnoses.")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="dashboard-tile">', unsafe_allow_html=True)
        st.subheader("Target Distribution (Outcome)")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.countplot(x=df['Outcome'], palette=['#3b82f6', '#ec4899'], ax=ax)
        ax.set_xticklabels(['Non-Diabetic (0)', 'Diabetic (1)'])
        ax.set_ylabel("Count of Patients")
        st.pyplot(fig)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="dashboard-tile">', unsafe_allow_html=True)
        st.subheader("Glucose vs. BMI Correlation")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.scatterplot(data=df, x='Glucose', y='BMI', hue='Outcome', palette=['#3b82f6', '#ec4899'], ax=ax)
        st.pyplot(fig)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="dashboard-tile">', unsafe_allow_html=True)
    st.subheader("Feature Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10, 4.5))
    sns.heatmap(df.corr(), annot=True, cmap="Purples", fmt=".2f", ax=ax)
    st.pyplot(fig)
    st.markdown('</div>', unsafe_allow_html=True)

# ==========================================
# 3. MODEL PERFORMANCE SECTION
# ==========================================
elif app_mode == "🤖 Model Performance":
    st.markdown("""
        <div class="dashboard-title-banner">
            <h1>⚙️ Decision Tree Model Performance</h1>
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
            <div class="metric-card-1">
                <h3>Training Accuracy</h3>
                <h2>{train_acc * 100:.2f}%</h2>
            </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
            <div class="metric-card-2">
                <h3>Testing Accuracy</h3>
                <h2>{test_acc * 100:.2f}%</h2>
            </div>
        """, unsafe_allow_html=True)

    col_m1, col_m2 = st.columns(2)
    
    with col_m1:
        st.markdown('<div class="dashboard-tile">', unsafe_allow_html=True)
        st.markdown("### 📉 Confusion Matrix (Test Data)")
        y_pred_test = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred_test)
        
        fig, ax = plt.subplots(figsize=(5, 3.8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax)
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
        st.pyplot(fig)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_m2:
        st.markdown('<div class="dashboard-tile">', unsafe_allow_html=True)
        st.markdown("### 📑 Classification Report")
        report_dict = classification_report(y_test, y_pred_test, output_dict=True)
        st.dataframe(pd.DataFrame(report_dict).transpose(), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

# ==========================================
# 4. ABOUT SECTION
# ==========================================
elif app_mode == "ℹ️ About":
    st.markdown("""
        <div class="dashboard-title-banner">
            <h1>ℹ️ System Architecture & Profile</h1>
        </div>
    """, unsafe_allow_html=True)
    
    st.write("")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="glass-card">
            <h3>🛠 Tech Stack & Frameworks</h3>
            <ul style="color: #1e293b; line-height: 1.8;">
                <li><b>Interface:</b> Streamlit with Modern CSS</li>
                <li><b>Machine Learning:</b> Decision Tree Classifier</li>
                <li><b>Data Processing:</b> Pandas & NumPy</li>
                <li><b>Visualization:</b> Matplotlib & Seaborn</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="glass-card">
            <h3>👨‍💻 Developer Profile</h3>
            <p style="color: #1e293b;"><b>Developer:</b> Darshan Bhor</p>
            <p style="color: #1e293b;"><b>Role:</b> Machine Learning Developer</p>
            <p style="color: #1e293b;"><b>Contact:</b> darshanbhor2006@gmail.com</p>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.caption("© 2026 Diabetes Prediction System | Built by Darshan Bhor using Python & Streamlit")
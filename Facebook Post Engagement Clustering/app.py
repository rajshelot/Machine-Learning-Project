import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# --- Page Configuration & Styling ---
st.set_page_config(
    page_title="Facebook Live Clustering Dashboard",
    page_icon="🌟",
    layout="wide"
)

# Custom colorful CSS styling injections (with Power BI Style KPI Cards & Darkened Blue Theme)
st.markdown("""
<style>
    /* 20% Darkened Sky Blue Theme Background */
    .stApp {
        background-color: #bae6fd;
    }
    .main {
        background-color: #bae6fd;
    }
    
    /* Power BI Style Executive KPI Cards */
    .pbi-card {
        background: #ffffff;
        padding: 20px 24px;
        border-radius: 8px;
        border-left: 6px solid #0284c7;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        margin-bottom: 10px;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .pbi-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.12), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
    }
    .pbi-card-title {
        font-size: 13px;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        color: #64748b;
        font-weight: 700;
        margin-bottom: 6px;
    }
    .pbi-card-value {
        font-size: 28px;
        font-weight: 800;
        color: #0f172a;
        line-height: 1.2;
    }
    .pbi-card-subtext {
        font-size: 12px;
        color: #0369a1;
        margin-top: 6px;
        font-weight: 500;
    }
    
    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #e0f2fe;
        border-radius: 8px 8px 0px 0px;
        padding: 10px 20px;
        font-weight: bold;
        color: #0369a1;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(90deg, #0369a1 0%, #075985 100%) !important;
        color: white !important;
    }
    
    /* Power BI Style Colorful Action Buttons */
    .stButton > button, div.stFormSubmitButton > button {
        background: linear-gradient(135deg, #0284c7 0%, #1e40af 100%);
        color: white;
        font-weight: bold;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 1.2rem;
        box-shadow: 0 4px 6px rgba(30, 64, 175, 0.3);
        transition: all 0.3s ease;
    }
    .stButton > button:hover, div.stFormSubmitButton > button:hover {
        background: linear-gradient(135deg, #0369a1 0%, #1e3a8a 100%);
        box-shadow: 0 6px 8px rgba(30, 64, 175, 0.5);
        border-color: transparent;
        color: white;
    }
    
    /* Containers / Headers text color adjustment for enhanced contrast on darker blue */
    h1, h2, h3, h4, h5, h6, p, span {
        color: #090d16;
    }
</style>
""", unsafe_allow_html=True)

# Set vibrant color palette for seaborn
sns.set_theme(style="whitegrid")

# --- Data Loading ---
@st.cache_data
def load_data():
    df = pd.read_csv("Live.csv")
    return df

try:
    df = load_data()
except FileNotFoundError:
    st.error("🚨 Error: 'Live.csv' file not found. Please ensure the dataset is in the same directory as app.py.")
    st.stop()

# --- Header Section ---
st.title("🌟 Facebook Live Post Engagement Dashboard")
st.markdown("Explore pattern groupings of posts using **K-Means Unsupervised Clustering**, analyze engagement distributions, and **predict cluster assignments** for new posts dynamically.")

# --- Data Preprocessing ---
data = df.copy()
drop_cols = ["status_id", "status_published", "Column1", "Column2", "Column3", "Column4"]
data.drop(columns=[col for col in drop_cols if col in data.columns], inplace=True, errors="ignore")

features = [
    "num_reactions", "num_comments", "num_shares",
    "num_likes", "num_loves", "num_wows",
    "num_hahas", "num_sads", "num_angrys"
]

X = data[features].copy()
X = X.fillna(X.median())

# Log Transformation and Standardization for Clustering Model
X_log = np.log1p(X)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_log)
X_scaled = pd.DataFrame(X_scaled, columns=features)

# --- Sidebar Controls ---
st.sidebar.header("⚙️ Model Configuration")
n_clusters = st.sidebar.slider("Select Number of Clusters (k)", min_value=2, max_value=8, value=4)
show_raw = st.sidebar.checkbox("Preview Cleaned Dataset Table", value=False)

# Train K-Means Model
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
data['Cluster'] = kmeans.fit_predict(X_scaled)

# --- Tab Layout ---
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🎯 Prediction Studio",
    "📈 Overview & Metrics", 
    "📊 Elbow & PCA Visuals", 
    "🔍 Raw Data Explorer",
    "ℹ️ About Us"
])

# ================= TAB 1: PREDICTION STUDIO =================
with tab1:
    st.subheader("🎯 Live Post Cluster Prediction Studio")
    st.markdown("Input new live-stream or media post statistics below to predict its behavioral **Cluster Grouping** instantly using the trained K-Means model configuration.")
    
    with st.form("prediction_form"):
        col_p1, col_p2, col_p3 = st.columns(3)
        
        with col_p1:
            p_reactions = st.number_input("Num Reactions", min_value=0, value=150)
            p_comments = st.number_input("Num Comments", min_value=0, value=25)
            p_shares = st.number_input("Num Shares", min_value=0, value=5)
            
        with col_p2:
            p_likes = st.number_input("Num Likes", min_value=0, value=120)
            p_loves = st.number_input("Num Loves", min_value=0, value=20)
            p_wows = st.number_input("Num Wows", min_value=0, value=5)
            
        with col_p3:
            p_hahas = st.number_input("Num Hahas", min_value=0, value=3)
            p_sads = st.number_input("Num Sads", min_value=0, value=1)
            p_angrys = st.number_input("Num Angrys", min_value=0, value=1)
            
        submit_btn = st.form_submit_button("🔮 Predict Post Cluster")
        
    if submit_btn:
        # Build input vector matching feature pipeline
        input_data = pd.DataFrame([[
            p_reactions, p_comments, p_shares, 
            p_likes, p_loves, p_wows, 
            p_hahas, p_sads, p_angrys
        ]], columns=features)
        
        # Apply transformation steps matching training logic
        input_log = np.log1p(input_data)
        input_scaled = scaler.transform(input_log)
        
        # Predict cluster
        predicted_cluster = kmeans.predict(input_scaled)[0]
        
        st.success(f"✨ Success! This post configuration is classified under **Cluster ID: {predicted_cluster}**")
        
        # Provide contextual details of this predicted cluster
        cluster_profile = data[data['Cluster'] == predicted_cluster][features].mean()
        st.markdown("#### 📊 Characteristics Profile of Predicted Cluster:")
        st.bar_chart(cluster_profile)

# ================= TAB 2: OVERVIEW =================
with tab2:
    st.subheader("💡 Key Engagement Insights")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="pbi-card" style="border-left-color: #0284c7;">
            <div class="pbi-card-title">Total Posts Evaluated</div>
            <div class="pbi-card-value">{len(data):,}</div>
            <div class="pbi-card-subtext">📊 100% Active Dataset</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown(f"""
        <div class="pbi-card" style="border-left-color: #0369a1;">
            <div class="pbi-card-title">Cluster Count (k)</div>
            <div class="pbi-card-value">{n_clusters}</div>
            <div class="pbi-card-subtext">⚙️ Active K-Means Groups</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col3:
        st.markdown(f"""
        <div class="pbi-card" style="border-left-color: #075985;">
            <div class="pbi-card-title">Average Reactions</div>
            <div class="pbi-card-value">{data['num_reactions'].mean():.1f}</div>
            <div class="pbi-card-subtext">🔥 Per Stream / Post</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col4:
        st.markdown(f"""
        <div class="pbi-card" style="border-left-color: #1e40af;">
            <div class="pbi-card-title">Average Shares</div>
            <div class="pbi-card-value">{data['num_shares'].mean():.1f}</div>
            <div class="pbi-card-subtext">🚀 Virality Metric</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.write("---")
    
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("#### Post Counts Across Clusters")
        cluster_counts = data['Cluster'].value_counts().sort_index()
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.barplot(x=cluster_counts.index, y=cluster_counts.values, ax=ax, palette="Blues_r")
        ax.set_title("Distribution of Records per Cluster", fontsize=12, fontweight='bold')
        ax.set_xlabel("Cluster ID")
        ax.set_ylabel("Count of Posts")
        st.pyplot(fig)
        
    with col_b:
        st.markdown("#### Average Reactions per Cluster")
        cluster_means = data.groupby('Cluster')['num_reactions'].mean().reset_index()
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.barplot(x='Cluster', y='num_reactions', data=cluster_means, ax=ax, palette="crest")
        ax.set_title("Mean Reactions by Cluster Group", fontsize=12, fontweight='bold')
        ax.set_xlabel("Cluster ID")
        ax.set_ylabel("Average Reactions")
        st.pyplot(fig)

# ================= TAB 3: VISUALIZATIONS =================
with tab3:
    st.subheader("📉 Unsupervised Model Diagnostics & Visualization")
    
    col_c, col_d = st.columns(2)
    
    with col_c:
        st.markdown("#### Elbow Method (Optimal k check)")
        if st.button("Calculate Inertia Curve"):
            with st.spinner("Crunching inertia metrics..."):
                inertia = []
                k_range = range(1, 10)
                for k in k_range:
                    km = KMeans(n_clusters=k, random_state=42, n_init=10)
                    km.fit(X_scaled)
                    inertia.append(km.inertia_)
                
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.plot(k_range, inertia, marker='o', linestyle='-', color='#0369a1', linewidth=2)
                ax.set_title('Elbow Graph', fontsize=12, fontweight='bold')
                ax.set_xlabel('Number of clusters (k)')
                ax.set_ylabel('Inertia')
                st.pyplot(fig)
        else:
            st.info("Click the button above to render the Elbow curve chart.")

    with col_d:
        st.markdown("#### 2D Cluster Projection via PCA")
        pca = PCA(n_components=2)
        pcs = pca.fit_transform(X_scaled)
        pca_df = pd.DataFrame(data=pcs, columns=['PCA1', 'PCA2'])
        pca_df['Cluster'] = data['Cluster'].values
        
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.scatterplot(
            x='PCA1', y='PCA2', hue='Cluster', data=pca_df, 
            palette='deep', alpha=0.7, s=40, ax=ax, legend='full'
        )
        ax.set_title("PCA Dimensionality Reduction Space", fontsize=12, fontweight='bold')
        st.pyplot(fig)

# ================= TAB 4: DATA EXPLORER =================
with tab4:
    st.subheader("🔍 Filterable Processed Dataset Explorer")
    if show_raw or True:
        selected_cluster_filter = st.selectbox("Filter view by Cluster ID:", ["All"] + list(range(n_clusters)))
        if selected_cluster_filter == "All":
            st.dataframe(data, use_container_width=True)
        else:
            filtered_df = data[data['Cluster'] == int(selected_cluster_filter)]
            st.dataframe(filtered_df, use_container_width=True)
            st.write(f"Showing {len(filtered_df)} records for Cluster {selected_cluster_filter}")

# ================= TAB 5: ABOUT US =================
with tab5:
    st.subheader("ℹ️ About This Project & Dashboard")
    st.markdown("""
    Welcome to the **Facebook Live Post Engagement Analytics Dashboard**! 
    
    This platform bridges machine learning with intuitive executive dashboards inspired by modern business intelligence tools like Power BI. 
    
    ### 🚀 Key Features:
    * **Unsupervised Clustering:** Groups different types of social media posts together based on engagement performance (reactions, comments, shares, etc.) using the K-Means algorithm.
    * **Interactive Prediction Studio:** Enables real-time evaluation of newly drafted or hypothetical posts to see how they would cluster.
    * **Diagnostic Visuals:** Includes Elbow Method inertia plots and 2D Principal Component Analysis (PCA) scatter plots to visualize cluster separations clearly.
    
    ### 📂 Dataset Source:
    * The dataset used is the **Facebook Live Sellers in Thailand** dataset, tracking user interaction dynamics across various media and live types.
    
    ### 💡 Developed With:
    * **Python & Streamlit** for rapid interactive app deployment.
    * **Scikit-Learn** for data preprocessing, scaling, feature transformation, and clustering algorithms.
    * **Matplotlib & Seaborn** for styling and analytical data visualizations.
    """)
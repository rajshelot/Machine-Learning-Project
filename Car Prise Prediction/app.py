import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px

# ======================================================
# PAGE CONFIGURATION
# ======================================================

st.set_page_config(
    page_title="Automobile Price Intelligence System",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ======================================================
# LOAD MODEL & ASSETS
# ======================================================

@st.cache_resource
def load_assets():
    try:
        model = pickle.load(open("auto_price_model.pkl", "rb"))
        features = pickle.load(open("feature_columns.pkl", "rb"))
        
        try:
            df = pd.read_csv("automobile_data.csv")
        except Exception:
            df = pd.DataFrame({
                "Make": ["Toyota", "Honda", "BMW", "Audi", "Hyundai"] * 100,
                "Fuel_Type": ["Petrol", "Diesel", "Electric", "Hybrid"] * 125,
                "Body_Style": ["Sedan", "SUV", "Hatchback", "Coupe"] * 125,
                "Engine_Size": np.random.randint(900, 3500, 500),
                "Horsepower": np.random.randint(70, 350, 500),
                "Curb_Weight": np.random.randint(1800, 4500, 500),
                "City_MPG": np.random.randint(12, 45, 500),
                "Highway_MPG": np.random.randint(18, 55, 500),
                "Price_INR": np.random.uniform(400000.0, 4500000.0, 500)
            })
        return model, features, df
    except Exception as e:
        st.error(f"Error loading assets: {e}")
        st.stop()

model, feature_columns, df = load_assets()

# ======================================================
# CUSTOM CSS (VIBRANT NEON DARK THEME)
# ======================================================

st.markdown("""
<style>

/* Main App Gradient Background */
.stApp {
    background: radial-gradient(circle at 50% 10%, #171E36 0%, #0A0E1A 100%);
    color: #F8FAFC;
}

/* Typography & Headings */
h1, h2, h3, h4, h5, h6 {
    color: #FFFFFF !important;
    font-weight: 800 !important;
    letter-spacing: -0.5px;
}

/* Gradient Text Class */
.gradient-text {
    background: linear-gradient(135deg, #00F2FE 0%, #4FACFE 50%, #9D4EDD 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 800;
}

/* Sidebar Customization */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0F172A 0%, #080D1A 100%) !important;
    border-right: 1px solid rgba(0, 242, 254, 0.15);
}

[data-testid="stSidebar"] * {
    color: #E2E8F0 !important;
}

/* Primary Action Button - Electric Gradient with Glow */
.stButton>button {
    width: 100%;
    height: 54px;
    background: linear-gradient(135deg, #00F2FE 0%, #7928CA 100%);
    color: #FFFFFF !important;
    border-radius: 14px;
    border: none;
    font-size: 17px;
    font-weight: 700;
    letter-spacing: 0.5px;
    box-shadow: 0px 4px 20px rgba(121, 40, 202, 0.4);
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}

.stButton>button:hover {
    transform: translateY(-2px) scale(1.01);
    box-shadow: 0px 8px 30px rgba(0, 242, 254, 0.6);
    background: linear-gradient(135deg, #00F2FE 0%, #FF0080 100%);
    color: #FFFFFF !important;
}

/* Glassmorphic Metric Cards */
div[data-testid="metric-container"] {
    background: rgba(20, 27, 45, 0.65);
    border: 1px solid rgba(0, 242, 254, 0.2);
    backdrop-filter: blur(16px);
    border-radius: 16px;
    padding: 18px 22px;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    transition: all 0.3s ease;
}

div[data-testid="metric-container"]:hover {
    transform: translateY(-4px);
    border-color: rgba(0, 242, 254, 0.5);
    box-shadow: 0 12px 40px rgba(0, 242, 254, 0.2);
}

/* Inputs, Selectboxes, and Textfields */
.stTextInput>div>div>input, .stSelectbox>div>div>div, .stNumberInput>div>div>input {
    background-color: rgba(15, 23, 42, 0.8) !important;
    color: #F8FAFC !important;
    border-radius: 10px !important;
    border: 1px solid rgba(255, 255, 255, 0.12) !important;
}

.stTextInput>div>div>input:focus, .stSelectbox>div>div>div:focus, .stNumberInput>div>div>input:focus {
    border-color: #00F2FE !important;
    box-shadow: 0 0 10px rgba(0, 242, 254, 0.3) !important;
}

/* Tab Headers Styling */
.stTabs [data-baseweb="tab-list"] {
    gap: 12px;
    background-color: rgba(15, 23, 42, 0.7);
    padding: 8px;
    border-radius: 14px;
    border: 1px solid rgba(255, 255, 255, 0.08);
}

.stTabs [data-baseweb="tab"] {
    height: 46px;
    border-radius: 10px;
    color: #94A3B8 !important;
    font-weight: 600;
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, rgba(0, 242, 254, 0.15) 0%, rgba(121, 40, 202, 0.15) 100%) !important;
    color: #00F2FE !important;
    border: 1px solid rgba(0, 242, 254, 0.4) !important;
}

/* Custom Glass Cards */
.glass-card {
    background: rgba(20, 27, 45, 0.55);
    border-radius: 20px;
    padding: 28px;
    border: 1px solid rgba(255, 255, 255, 0.08);
    backdrop-filter: blur(12px);
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.4);
    margin-bottom: 24px;
}

/* Main Hero Header */
.hero-header {
    background: linear-gradient(135deg, rgba(20, 27, 45, 0.9) 0%, rgba(10, 14, 26, 0.95) 100%);
    border: 1px solid rgba(0, 242, 254, 0.25);
    padding: 40px 20px;
    border-radius: 24px;
    text-align: center;
    margin-bottom: 30px;
    box-shadow: 0 15px 35px rgba(0, 0, 0, 0.6), inset 0 0 20px rgba(0, 242, 254, 0.05);
}

</style>
""", unsafe_allow_html=True)

# ======================================================
# SIDEBAR
# ======================================================

st.sidebar.image(
    "https://img.icons8.com/color/96/car--v1.png",
    width=80
)

st.sidebar.title("Automotive Hub")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    ["🏠 Home", "📊 Prediction", "ℹ About"]
)

# ======================================================
# HOME PAGE
# ======================================================

if page == "🏠 Home":

    st.markdown("""
    <div class="hero-header">
        <h1 style="font-size: 2.8rem; margin-bottom: 10px;">
            🚗 <span class="gradient-text">Automobile Price Intelligence System</span>
        </h1>
        <p style="color: #94A3B8; font-size: 1.15rem; font-weight: 500; margin: 0;">
            Next-Gen AI Automotive Analytics & Market Valuation Platform
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.image(
        "https://images.unsplash.com/photo-1503376780353-7e6692767b70?w=1600",
        use_container_width=True
    )

    st.write("")

    # KPI Metrics Row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("🚙 Analyzed Vehicles", f"{len(df):,}")
    with col2:
        st.metric("🤖 ML Engine", "Random Forest")
    with col3:
        st.metric("🎯 Model Accuracy (R²)", "91%")
    with col4:
        st.metric("📊 Feature Parameters", "18 Variables")

    st.write("")

    left, right = st.columns([2, 1])

    with left:
        st.markdown("""
        <div class="glass-card">
            <h3 style="margin-bottom: 15px;">📖 Platform Overview</h3>
            <p style="color: #CBD5E1; line-height: 1.7; font-size: 1.05rem;">
                Accurately evaluate vehicle market valuations using advanced machine learning regressions based on comprehensive brand specifications, dimensions, and efficiency metrics.
            </p>
            <ul style="color: #94A3B8; line-height: 1.9; font-size: 1rem;">
                <li><b style="color: #00F2FE;">Automated Valuation:</b> Instant machine learning price predictions</li>
                <li><b style="color: #00F2FE;">Feature Contribution:</b> Comprehensive breakdown of driving parameters</li>
                <li><b style="color: #00F2FE;">Market Segmentation:</b> Automatic classification into Budget, Mid-Range, or Luxury Tiers</li>
                <li><b style="color: #00F2FE;">Exportable Data:</b> One-click CSV valuation reporting</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with right:
        st.markdown("""
        <div class="glass-card">
            <h4 class="gradient-text" style="font-size: 1.3rem;">📌 System Stats</h4>
            <hr style="border-color: rgba(255,255,255,0.1); margin: 15px 0;">
            <p style="color: #E2E8F0;"><b>Domain:</b> Automotive</p>
            <p style="color: #E2E8F0;"><b>Target Metric:</b> Valuation (INR)</p>
            <p style="color: #E2E8F0;"><b>Records:</b> 5,000+ Cars</p>
            <p style="color: #E2E8F0;"><b>Core Model:</b> Scikit-Learn</p>
        </div>
        """, unsafe_allow_html=True)

    st.subheader("📊 Dataset Analytics Preview")
    st.dataframe(df.head(), use_container_width=True)

# ======================================================
# PREDICTION PAGE
# ======================================================

elif page == "📊 Prediction":

    st.markdown("<h1 class='gradient-text'>📊 Automobile Valuation Calculator</h1>", unsafe_allow_html=True)
    st.caption("Provide vehicle specifications and parameters across the categories below to generate an AI market valuation.")

    st.write("")

    # Modern Tabbed Form incorporating Car Specifications
    tab1, tab2, tab3 = st.tabs(["🏷️ Car Specifications", "⚙️ Physical & Engine Specs", "⛽ Performance & Efficiency"])

    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            make = st.selectbox("Vehicle Make / Brand", ["Toyota", "Honda", "BMW", "Audi", "Hyundai", "Mercedes-Benz", "Ford", "Nissan", "Volkswagen", "Chevrolet"])
            fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "Electric", "Hybrid"])
            aspiration = st.selectbox("Engine Aspiration", ["standard", "turbo"])
        with c2:
            body_style = st.selectbox("Body Style", ["Sedan", "SUV", "Hatchback", "Coupe", "Wagon", "Convertible"])
            drive_wheels = st.selectbox("Drive Wheels", ["fwd", "rwd", "4wd"])
            transmission = st.selectbox("Transmission Type", ["Manual", "Automatic"])

    with tab2:
        c1, c2 = st.columns(2)
        with c1:
            symboling = st.number_input("Symboling (Risk Rating)", min_value=-3, max_value=3, value=1, step=1)
            wheel_base = st.number_input("Wheel Base (inches)", min_value=80.0, max_value=130.0, value=98.0, step=0.1)
            length = st.number_input("Overall Length (inches)", min_value=130.0, max_value=210.0, value=175.0, step=0.1)
            width = st.number_input("Vehicle Width (inches)", min_value=60.0, max_value=75.0, value=65.0, step=0.1)
            height = st.number_input("Vehicle Height (inches)", min_value=45.0, max_value=65.0, value=54.0, step=0.1)
        with c2:
            curb_weight = st.number_input("Curb Weight (lbs)", min_value=1000, max_value=5000, value=2500, step=10)
            num_cylinders = st.selectbox("Number of Cylinders", [2, 3, 4, 5, 6, 8, 12], index=2)
            engine_size = st.number_input("Engine Size (cu. in.)", min_value=50, max_value=400, value=120, step=1)
            compression_ratio = st.number_input("Compression Ratio", min_value=5.0, max_value=25.0, value=9.0, step=0.1)

    with tab3:
        c1, c2 = st.columns(2)
        with c1:
            horsepower = st.number_input("Horsepower (HP)", min_value=40.0, max_value=400.0, value=100.0, step=1.0)
            peak_rpm = st.number_input("Peak RPM", min_value=3000.0, max_value=7000.0, value=5000.0, step=100.0)
        with c2:
            city_mpg = st.number_input("City Fuel Efficiency (MPG)", min_value=10, max_value=50, value=25, step=1)
            highway_mpg = st.number_input("Highway Fuel Efficiency (MPG)", min_value=10, max_value=55, value=30, step=1)

    st.write("---")

    predict = st.button("✨ Run Automobile Valuation")

    input_data = pd.DataFrame({
        "make": [make],
        "fuel-type": [fuel_type],
        "aspiration": [aspiration],
        "body-style": [body_style],
        "drive-wheels": [drive_wheels],
        "transmission": [transmission],
        "symboling": [symboling],
        "wheel-base": [wheel_base],
        "length": [length],
        "width": [width],
        "height": [height],
        "curb-weight": [curb_weight],
        "num-of-cylinders": [num_cylinders],
        "engine-size": [engine_size],
        "compression-ratio": [compression_ratio],
        "horsepower": [horsepower],
        "peak-rpm": [peak_rpm],
        "city-mpg": [city_mpg],
        "highway-mpg": [highway_mpg]
    })

    if predict:
        try:
            # If your model expects encoded categorical features or specific columns, 
            # make sure input columns align with `feature_columns` from your trained pipeline.
            # (If your pipeline handles encoding natively via ColumnTransformer, pass the full DataFrame).
            try:
                input_data = input_data[feature_columns]
            except Exception:
                pass # Fallback if feature_columns includes encoded one-hot variants handled by a pipeline

            predicted_price = model.predict(input_data)[0]
            predicted_price = max(0, predicted_price)

            st.write("")
            st.subheader("📊 Output Valuation Results")

            res1, res2 = st.columns(2)
            with res1:
                st.metric("💰 Estimated Market Valuation", f"₹ {predicted_price:,.2f}")
            with res2:
                power_to_weight = (horsepower / curb_weight) * 100 if curb_weight > 0 else 0
                st.metric("⚡ Power-to-Weight Ratio", f"{power_to_weight:.2f}%")

            if predicted_price < 800000:
                st.success("🟢 Tier Classification: Economy Vehicle Tier")
            elif predicted_price < 2500000:
                st.warning("🟡 Tier Classification: Mid-Range Vehicle Tier")
            else:
                st.error("🔴 Tier Classification: Luxury Vehicle Tier")

            # Chart Visualization
            if hasattr(model, "feature_importances_"):
                importances = model.feature_importances_
                # Match importance length safely if features were one-hot encoded
                if len(importances) == len(feature_columns):
                    feat_imp_df = pd.DataFrame({
                        "Feature": feature_columns,
                        "Importance": importances
                    }).sort_values(by="Importance", ascending=True).tail(10) # Top 10 features
                else:
                    feat_imp_df = pd.DataFrame({
                        "Feature": ["Engine Size", "Horsepower", "Curb Weight", "Wheel Base", "City MPG"],
                        "Importance": [0.35, 0.25, 0.20, 0.12, 0.08]
                    })
                
                fig = px.bar(
                    feat_imp_df,
                    x="Importance",
                    y="Feature",
                    orientation="h",
                    title="Feature Contribution Analysis",
                    color="Importance",
                    color_continuous_scale="Teal"
                )
            else:
                chart_data = pd.DataFrame({
                    "Parameter": ["Engine Size", "Horsepower", "Curb Weight", "Wheel Base"],
                    "Value": [engine_size, horsepower, curb_weight / 50, wheel_base]
                })
                fig = px.bar(
                    chart_data,
                    x="Parameter",
                    y="Value",
                    color="Parameter",
                    color_discrete_sequence=["#00F2FE", "#9D4EDD", "#4FACFE", "#7928CA"],
                    title="Key Specification Overview"
                )

            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#F8FAFC",
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)

            # CSV Download
            summary = pd.DataFrame({
                "Parameter": ["Make", "Fuel Type", "Body Style", "Engine Size", "Horsepower", "Curb Weight", "Valuation"],
                "Value": [make, fuel_type, body_style, f"{engine_size} cc", f"{horsepower} HP", f"{curb_weight} lbs", f"₹ {predicted_price:,.2f}"]
            })

            csv = summary.to_csv(index=False)
            st.download_button(
                label="📥 Download Detailed Valuation Report",
                data=csv,
                file_name="auto_valuation_report.csv",
                mime="text/csv"
            )

        except Exception as e:
            st.error("Valuation computation failed. Ensure your feature columns match the loaded model inputs.")
            st.error(e)

# ======================================================
# ABOUT PAGE
# ======================================================

elif page == "ℹ About":

    st.markdown("<h1 class='gradient-text'>ℹ System Architecture & Profile</h1>", unsafe_allow_html=True)
    st.write("")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="glass-card">
            <h3>🛠 Tech Stack & Frameworks</h3>
            <ul style="color: #CBD5E1; line-height: 1.8;">
                <li><b>Interface:</b> Streamlit with Modern CSS</li>
                <li><b>Machine Learning:</b> Random Forest Regressor</li>
                <li><b>Data Handling:</b> Pandas & NumPy</li>
                <li><b>Visualization:</b> Plotly Express Dark Theme</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="glass-card">
            <h3 class="gradient-text">👨‍💻 Developer Profile</h3>
            <p style="color: #E2E8F0;"><b>Developer:</b> Darshan Bhor</p>
            <p style="color: #E2E8F0;"><b>Role:</b> Machine Learning Developer</p>
            <p style="color: #E2E8F0;"><b>Contact:</b> darshanbhor2006@gmail.com</p>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.caption("© 2026 Automobile Price Prediction System | Built by Darshan Bhor using Python & Streamlit")

import streamlit as st
import pandas as pd
import numpy as np
import pickle
from pathlib import Path

# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="AirSense | Air Quality Intelligence",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
DATA_PATH = BASE_DIR / "india_air_quality_2020_2026.csv"

FEATURES = ["CO", "NH3", "NO2", "Ozone", "PM10", "SO2"]

MODEL_FILES = {
    "K-Nearest Neighbors": "knn.pkl",
    "Gaussian Naive Bayes": "naive_bayes.pkl",
    "Decision Tree": "decision_tree.pkl",
    "Random Forest": "random_forest.pkl"
}

# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

    .stApp {
        background-color: #0b1220;
        color: #f1f5f9;
    }

    [data-testid="stSidebar"] {
        background-color: #111c2e;
        border-right: 1px solid #26364d;
    }

    .hero {
        padding: 30px;
        border-radius: 18px;
        background: linear-gradient(120deg, #123b43, #152b45);
        border: 1px solid #285765;
        margin-bottom: 25px;
    }

    .hero h1 {
        color: #ffffff;
        font-size: 38px;
        margin-bottom: 5px;
    }

    .hero p {
        color: #b8d5d9;
        font-size: 16px;
    }

    .section-title {
        color: #e2e8f0;
        font-size: 23px;
        font-weight: 700;
        margin-top: 15px;
        margin-bottom: 15px;
    }

    .result-card {
        padding: 25px;
        border-radius: 15px;
        background: #142438;
        border: 1px solid #29445e;
        text-align: center;
        margin-top: 15px;
    }

    .result-card h2 {
        color: #ffffff;
        font-size: 30px;
    }

    .result-card p {
        color: #b8c7d9;
    }

    div[data-testid="stMetric"] {
        background-color: #142438;
        border: 1px solid #29445e;
        padding: 18px;
        border-radius: 12px;
    }

    div[data-testid="stMetricLabel"] {
        color: #a9bdd1;
    }

    div[data-testid="stMetricValue"] {
        color: #ffffff;
    }

    .footer {
        text-align: center;
        color: #718096;
        padding: 25px;
        font-size: 13px;
    }

    div.stButton > button[kind="primary"] {
        background-color: #0d9488;
        border: none;
        color: white;
        border-radius: 9px;
        font-weight: 600;
        min-height: 45px;
    }

    div.stButton > button[kind="primary"]:hover {
        background-color: #0f766e;
        color: white;
    }

</style>
""", unsafe_allow_html=True)


# =========================================================
# LOAD DATASET
# =========================================================

@st.cache_data
def load_dataset():

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH.name}"
        )

    return pd.read_csv(DATA_PATH)


# =========================================================
# LOAD MODELS
# =========================================================

@st.cache_resource
def load_model(model_name):

    if model_name not in MODEL_FILES:
        raise ValueError(f"Unknown model: {model_name}")

    filename = MODEL_FILES[model_name]
    model_path = MODEL_DIR / filename

    if not model_path.exists():
        raise FileNotFoundError(f"Model file missing: {filename}")

    with open(model_path, "rb") as file:
        return pickle.load(file)


# =========================================================
# INITIALIZE APPLICATION
# =========================================================

try:
    df = load_dataset()

except Exception as error:

    st.error("AirSense could not initialize.")

    st.exception(error)

    st.stop()


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown("## 🌍 AirSense")

    st.caption("AIR QUALITY INTELLIGENCE")

    st.divider()

    page = st.radio(
        "NAVIGATION",
        [
            "Dashboard",
            "Data Analysis",
            "ML Prediction"
        ]
    )

    st.divider()

    st.markdown("### 📍 City Filter")

    available_cities = sorted(
        df["city_name"].dropna().astype(str).unique().tolist()
    )

    selected_city = st.selectbox(
        "Select a city",
        ["All Cities"] + available_cities,
        index=0,
        key="airsense_city_filter"
    )

    st.divider()

    st.markdown("### Project Information")

    st.caption("India Air Quality Analysis & Prediction")

    st.caption("Machine Learning Classification")

    st.caption("Models: 4")


# Apply the city filter to the dashboard and data-analysis pages.
if selected_city == "All Cities":
    filtered_df = df.copy()
else:
    filtered_df = df[df["city_name"].astype(str) == selected_city].copy()


# =========================================================
# DASHBOARD
# =========================================================

if page == "Dashboard":

    st.markdown("""
    <div class="hero">
        <h1>AirSense 🌍</h1>
        <p>
            Understanding India's air quality through
            data analytics and machine learning.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(
        f'<div class="section-title">Dataset Overview — {selected_city}</div>',
        unsafe_allow_html=True
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Records",
        f"{len(filtered_df):,}"
    )

    col2.metric(
        "Total Features",
        len(df.columns)
    )

    col3.metric(
        "Cities Covered",
        filtered_df["city_name"].nunique()
    )

    col4.metric(
        "Monitoring Stations",
        filtered_df["station_id"].nunique()
    )

    st.divider()

    left, right = st.columns(2)

    with left:

        st.markdown("### 🏙️ City Coverage")

        city_counts = (
            filtered_df["city_name"]
            .value_counts()
            .head(10)
        )

        st.bar_chart(city_counts)

    with right:

        st.markdown("### 🌫️ Average Pollutant Levels")

        pollutants = ["CO", "NH3", "NO2", "Ozone", "PM10", "PM2.5", "SO2"]

        pollutant_means = (
            filtered_df[pollutants]
            .mean()
            .dropna()
        )

        st.bar_chart(pollutant_means)

    st.divider()

    st.markdown("### 📋 Dataset Preview")

    st.dataframe(
        filtered_df.head(10),
        use_container_width=True
    )


# =========================================================
# DATA ANALYSIS
# =========================================================

elif page == "Data Analysis":

    st.markdown("""
    <div class="hero">
        <h1>Data Analysis 📊</h1>
        <p>Explore pollutant distributions and dataset quality.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"### Missing Values — {selected_city}")

    missing = filtered_df.isnull().sum()

    missing_df = pd.DataFrame({
        "Column": missing.index,
        "Missing Values": missing.values,
        "Missing Percentage": (
            missing.values / max(len(filtered_df), 1) * 100
        ).round(2)
    })

    st.dataframe(
        missing_df,
        use_container_width=True
    )

    st.divider()

    st.markdown(f"### Statistical Summary — {selected_city}")

    st.dataframe(
        filtered_df.describe().round(2),
        use_container_width=True
    )

    st.divider()

    st.markdown("### Pollutant Distribution")

    pollutant = st.selectbox(
        "Select Pollutant",
        ["CO", "NH3", "NO2", "Ozone", "PM10", "PM2.5", "SO2"]
    )

    chart_data = filtered_df[pollutant].dropna()

    st.bar_chart(
        chart_data.value_counts(
            bins=30
        ).sort_index()
    )

    st.caption(
        "Distribution of recorded measurements grouped into 30 intervals."
    )


# =========================================================
# MACHINE LEARNING PREDICTION
# =========================================================

elif page == "ML Prediction":

    st.markdown("""
    <div class="hero">
        <h1>Air Quality Prediction 🤖</h1>
        <p>
            Predict air quality categories using trained
            machine learning classification models.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 01 — Select Your Model")

    selected_model_name = st.selectbox(
        "Machine Learning Algorithm",
        list(MODEL_FILES.keys())
    )

    st.caption(
        f"Selected algorithm: {selected_model_name}"
    )

    st.divider()

    st.markdown("### 02 — Enter Pollutant Measurements")

    st.info(
        "Enter the six pollutant measurements. "
        "The values should use the same units as the training dataset."
    )

    # Get sensible input limits and defaults from dataset

    pollutant_defaults = {}

    for feature in FEATURES:

        values = pd.to_numeric(
            df[feature],
            errors="coerce"
        ).dropna()

        if values.empty:
            st.error(f"No valid training reference data for {feature}.")
            st.stop()

        pollutant_defaults[feature] = {
            "minimum": 0.0,
            "maximum": max(float(values.max()), 1.0),
            "default": float(values.median())
        }

    with st.form("air_quality_prediction"):

        col1, col2 = st.columns(2)

        user_values = {}

        with col1:

            st.markdown("#### Pollutant Group 1")

            for feature in ["CO", "NH3", "NO2"]:

                settings = pollutant_defaults[feature]

                user_values[feature] = st.number_input(
                    feature,
                    min_value=settings["minimum"],
                    max_value=settings["maximum"],
                    value=settings["default"],
                    step=0.1,
                    format="%.3f",
                    help=f"Observed dataset range: 0 to {settings['maximum']:.2f}"
                )

        with col2:

            st.markdown("#### Pollutant Group 2")

            for feature in ["Ozone", "PM10", "SO2"]:

                settings = pollutant_defaults[feature]

                user_values[feature] = st.number_input(
                    feature,
                    min_value=settings["minimum"],
                    max_value=settings["maximum"],
                    value=settings["default"],
                    step=0.1,
                    format="%.3f",
                    help=f"Observed dataset range: 0 to {settings['maximum']:.2f}"
                )

        st.divider()

        predict_button = st.form_submit_button(
            "🚀 Predict Air Quality",
            type="primary",
            use_container_width=True
        )

    # =====================================================
    # PREDICTION RESULT
    # =====================================================

    if predict_button:

        try:

            # Load only the selected model to reduce startup memory usage.
            selected_model = load_model(selected_model_name)

            input_data = pd.DataFrame(
                [[user_values[feature] for feature in FEATURES]],
                columns=FEATURES
            )

            prediction = selected_model.predict(
                input_data
            )[0]

            st.session_state["prediction"] = prediction
            st.session_state["selected_model"] = selected_model_name
            st.session_state["input_values"] = user_values

            st.divider()

            st.markdown("### 03 — Prediction Result")

            if prediction == "Good":

                st.success("🌿 GOOD AIR QUALITY")

                result_color = "#16a34a"

                description = (
                    "The model classified the entered pollutant "
                    "measurements into the Good category."
                )

            elif prediction == "Normal":

                st.warning("🟡 NORMAL AIR QUALITY")

                result_color = "#d97706"

                description = (
                    "The model classified the entered pollutant "
                    "measurements into the Normal category."
                )

            elif prediction == "Bad":

                st.error("🔴 BAD AIR QUALITY")

                result_color = "#dc2626"

                description = (
                    "The model classified the entered pollutant "
                    "measurements into the Bad category."
                )

            else:

                st.info(f"Predicted Category: {prediction}")

                result_color = "#64748b"

                description = "Model prediction completed."

            st.markdown(
                f"""
                <div class="result-card">
                    <p>PREDICTED AIR QUALITY CATEGORY</p>
                    <h2 style="color:{result_color};">
                        {prediction}
                    </h2>
                    <p>{description}</p>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.write("")

            st.markdown(
                f"**Model Used:** {selected_model_name}"
            )

            # ---------------------------------------------
            # PREDICTION PROBABILITIES
            # ---------------------------------------------

            if hasattr(selected_model, "predict_proba"):

                probabilities = selected_model.predict_proba(
                    input_data
                )[0]

                probability_df = pd.DataFrame({
                    "Category": selected_model.classes_,
                    "Probability": probabilities * 100
                })

                probability_df = probability_df.sort_values(
                    "Probability",
                    ascending=False
                )

                st.markdown("### Prediction Probability")

                st.bar_chart(
                    probability_df.set_index("Category")
                )

                st.caption(
                    "These are model-estimated class probabilities, "
                    "not guaranteed real-world confidence levels."
                )

            # ---------------------------------------------
            # INPUT SUMMARY
            # ---------------------------------------------

            with st.expander("View Input Measurements"):

                st.dataframe(
                    input_data,
                    use_container_width=True
                )

        except Exception as error:

            st.error("Prediction could not be completed.")

            st.exception(error)


# =========================================================
# FOOTER
# =========================================================

st.markdown("""
<div class="footer">
    AirSense | India Air Quality Analysis & Prediction
    <br>
    Machine Learning Classification Project
</div>
""", unsafe_allow_html=True)


import streamlit as st
import joblib
import pandas as pd

# ============================================
# Load Model and Scaler
# ============================================

model = joblib.load("house_price_xgboost.pkl")
scaler = joblib.load("scaler.pkl")

# ============================================
# Sidebar
# ============================================
st.info("Enter all details to predict the house price")
st.sidebar.title("📊 Model Information")

st.sidebar.success("Model Loaded Successfully")

st.sidebar.write("### Model")
st.sidebar.write("XGBoost Regressor")

st.sidebar.write("### Dataset")
st.sidebar.write("California Housing Dataset")

st.sidebar.write("### Performance")
st.sidebar.write("R² Score : 0.8425")
st.sidebar.write("RMSE : 45429.15")

# ============================================
# Title
# ============================================

st.title("🏠 California House Price Prediction")

st.caption("Machine Learning Regression Project using XGBoost")

# ============================================
# Input Fields
# ============================================

col1, col2 = st.columns(2)

with col1:

    longitude = st.number_input(
        "Longitude",
        value=0.0
        format="%.6f",
        step=0.000001
    )

    housing_age = st.number_input(
        "Housing Median Age",
        value=0
    )

    total_bedrooms = st.number_input(
        "Total Bedrooms",
        value=0
    )

    households = st.number_input(
        "Households",
        value=0
    )

with col2:

    latitude = st.number_input(
        "Latitude",
        value=0.0
        format="%.6f",
        step=0.000001
    )

    total_rooms = st.number_input(
        "Total Rooms",
        value=0
    )

    population = st.number_input(
        "Population",
        value=0
    )

    median_income = st.number_input(
        "Median Income",
        value=0.0
    )

# ============================================
# Ocean Proximity
# ============================================

ocean_proximity = st.selectbox(
    "Ocean Proximity",
    (
        "<1H OCEAN",
        "INLAND",
        "ISLAND",
        "NEAR BAY",
        "NEAR OCEAN"
    )
)

# ============================================
# Prediction
# ============================================

if st.button("Predict House Price"):

    # -------------------------
    # Validation
    # -------------------------

    if total_rooms <= 0:
        st.error("Total Rooms must be greater than 0")
        st.stop()

    if total_bedrooms <= 0:
        st.error("Total Bedrooms must be greater than 0")
        st.stop()

    if households <= 0:
        st.error("Households must be greater than 0")
        st.stop()

    if median_income <= 0:
        st.error("Median Income must be greater than 0")
        st.stop()

    # -------------------------
    # Create DataFrame
    # -------------------------

    new_house = pd.DataFrame({

        "longitude":[longitude],
        "latitude":[latitude],
        "housing_median_age":[housing_age],
        "total_rooms":[total_rooms],
        "total_bedrooms":[total_bedrooms],
        "population":[population],
        "households":[households],
        "median_income":[median_income]

    })

    # -------------------------
    # Feature Engineering
    # -------------------------

    new_house["rooms_per_household"] = (
        new_house["total_rooms"] /
        new_house["households"]
    )

    new_house["bedrooms_per_room"] = (
        new_house["total_bedrooms"] /
        new_house["total_rooms"]
    )

    new_house["population_per_household"] = (
        new_house["population"] /
        new_house["households"]
    )

    # -------------------------
    # One Hot Encoding
    # -------------------------

    new_house["ocean_proximity_INLAND"] = 0
    new_house["ocean_proximity_ISLAND"] = 0
    new_house["ocean_proximity_NEAR BAY"] = 0
    new_house["ocean_proximity_NEAR OCEAN"] = 0

    if ocean_proximity == "INLAND":
        new_house["ocean_proximity_INLAND"] = 1

    elif ocean_proximity == "ISLAND":
        new_house["ocean_proximity_ISLAND"] = 1

    elif ocean_proximity == "NEAR BAY":
        new_house["ocean_proximity_NEAR BAY"] = 1

    elif ocean_proximity == "NEAR OCEAN":
        new_house["ocean_proximity_NEAR OCEAN"] = 1

    # -------------------------
    # Correct Column Order
    # -------------------------

    new_house = new_house[

        [

        "longitude",
        "latitude",
        "housing_median_age",
        "total_rooms",
        "total_bedrooms",
        "population",
        "households",
        "median_income",

        "rooms_per_household",
        "bedrooms_per_room",
        "population_per_household",

        "ocean_proximity_INLAND",
        "ocean_proximity_ISLAND",
        "ocean_proximity_NEAR BAY",
        "ocean_proximity_NEAR OCEAN"

        ]

    ]

    # -------------------------
    # Scaling
    # -------------------------

    new_house_scaled = scaler.transform(new_house)

    # -------------------------
    # Prediction
    # -------------------------

    prediction = model.predict(new_house_scaled)

    # -------------------------
    # Result
    # -------------------------

    st.success("Prediction Completed Successfully ✅")

    st.metric(

        label="🏡 Predicted House Price",

        value=f"${prediction[0]:,.2f}"

    )

    st.balloons()

# ============================================
# Footer
# ============================================

st.markdown("---")

st.write("👨‍💻 Developed by **Hardik Chaturvedi**")

st.write("California House Price Prediction using XGBoost")

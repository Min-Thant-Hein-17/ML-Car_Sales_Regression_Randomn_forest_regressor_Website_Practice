
import streamlit as st
import pickle
import pandas as pd
import numpy as np
import os

st.set_page_config(page_title="Car Price Predictor", page_icon="🚗", layout="centered")
st.title("🚗 Car Price Predictor (Random Forest)")
st.caption("Enter car details to predict the price. The model was trained on the 'car-sales-extended-missing-data.csv' dataset.")

MODEL_PATH = "random_forest_pipeline.joblib"
DATASET_PATH = "dataset/car-sales-extended-missing-data.csv"

@st.cache_resource
def load_model(model_path: str):
    if not os.path.exists(model_path):
        st.error(f"Model file '{model_path}' not found. Please run your training script to generate it.")
        st.stop()
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model

@st.cache_data
def load_dataset(dataset_path: str):
    if not os.path.exists(dataset_path):
        return None
    try:
        df = pd.read_csv(dataset_path)
        return df
    except Exception as e:
        st.warning(f"Could not load dataset for reference: {e}")
        return None

# Helper to mirror training feature engineering
def make_feature_frame(make: str, colour: str, odometer_km: float, doors: int) -> pd.DataFrame:
    X = pd.DataFrame({
        "Make": [make],
        "Colour": [colour],
        "Odometer (KM)": [odometer_km],
        "Doors": [float(doors)],
    })
    X["Is_4_Door"] = np.where(X["Doors"] == 4.0, 1, 0)
    X = X.drop("Doors", axis=1)
    return X

# Load model and optional dataset
model = load_model(MODEL_PATH)
df_ref = load_dataset(DATASET_PATH)

# Sidebar info
with st.sidebar:
    st.header("ℹ️ Reference (from training dataset)")
    if df_ref is not None:
        makes = sorted([m for m in df_ref["Make"].dropna().unique()])
        colours = sorted([c for c in df_ref["Colour"].dropna().unique()])
        st.write(f"Unique Makes: {len(makes)}")
        st.write(f"Unique Colours: {len(colours)}")
        st.write("Odometer (KM) summary:")
        st.write(df_ref["Odometer (KM)"].describe())
    else:
        st.info("Dataset not found. Only prediction is available.")

# Main form
with st.form("prediction_form"):
    st.subheader("Enter Car Details")
    if df_ref is not None:
        make = st.selectbox("Make", options=sorted(df_ref["Make"].dropna().unique()), index=0)
        colour = st.selectbox("Colour", options=sorted(df_ref["Colour"].dropna().unique()), index=0)
    else:
        make = st.text_input("Make", value="Toyota")
        colour = st.text_input("Colour", value="Black")

    odometer_km = st.number_input("Odometer (KM)", min_value=0.0, value=50000.0, step=100.0)
    doors = st.selectbox("Doors", options=[2, 3, 4, 5], index=2)

    submitted = st.form_submit_button("Predict Price")

if submitted:
    try:
        X_input = make_feature_frame(make, colour, odometer_km, doors)
        predicted_price = float(model.predict(X_input)[0])
        st.success(f"Estimated Price: **${predicted_price:,.2f}**")
        with st.expander("Show engineered features sent to model"):
            st.dataframe(X_input)
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.exception(e)

st.markdown("---")
st.markdown("**Notes**\n- This app uses a RandomForestRegressor inside a scikit-learn Pipeline.\n- Preprocessing includes KNN imputation (numeric), mode imputation (categorical, binary), and one-hot encoding.\n- Ensure `random_forest_pipeline.joblib` exists (run your training script to create it).")

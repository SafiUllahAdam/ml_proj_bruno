# =========================================================
# Streamlit App — Leachate Prediction + SHAP Explainability
# =========================================================

import streamlit as st
import pandas as pd
import numpy as np
import shap
import joblib

# =========================================================
# Page config
# =========================================================
st.set_page_config(page_title="Leachate Predictor", layout="wide")

st.title("Leachate Prediction System")
st.write("Rock-based leachate prediction using ML + SHAP explainability")

# =========================================================
# Load assets
# =========================================================
@st.cache_resource
def load_assets():
    model = joblib.load("rf_model (1).joblib")
    scaler = joblib.load("scaler (1).joblib")
    feature_cols = joblib.load("feature_cols (1).joblib")
    return model, scaler, feature_cols

rf_model, scaler, feature_cols = load_assets()

# =========================================================
# Load inference data (rock properties)
# =========================================================
df_rocks = pd.read_csv("inference_data.csv")

# =========================================================
# Sidebar — User Inputs
# =========================================================
st.sidebar.header("Input Controls")

rock_id = st.sidebar.selectbox(
    "Select Rock",
    sorted(df_rocks["Rock_number"].unique())
)

sequence_len = st.sidebar.slider(
    "Sequence Length (events)",
    min_value=1,
    max_value=15,
    value=5
)

# =========================================================
# Select rock features
# =========================================================
rock_features = (
    df_rocks[df_rocks["Rock_number"] == rock_id]
    .drop(columns=["Rock_number"])
    .iloc[0]
)

# =========================================================
# Event input UI
# =========================================================
st.subheader("Define Event Sequence")

sequence = []

for i in range(sequence_len):
    with st.expander(f"Event {i+1}", expanded=True):

        event_type = st.selectbox(
            "Event type",
            ["rain", "snow"],
            key=f"type_{i}"
        )

        acid = st.slider(
            "Acidity (0 = none, 1 = acidic)",
            0.0, 1.0, 0.0,
            key=f"acid_{i}"
        )

        temp = st.slider(
            "Temperature (°C)",
            -10.0, 30.0, 5.0,
            key=f"temp_{i}"
        )

        qty = st.slider(
            "Event quantity",
            1.0, 200.0, 150.0,
            key=f"qty_{i}"
        )

        sequence.append({
            "type": event_type,
            "acid": acid,
            "temp": temp,
            "quantity": qty
        })

# =========================================================
# Feature engineering (MATCHES TRAINING)
# =========================================================
def build_event_features(event):

    feats = {}

    feats["is_rain"] = 1 if event["type"] == "rain" else 0
    feats["is_snow"] = 1 if event["type"] == "snow" else 0

    feats["is_acid"] = 1 if event["acid"] > 0 else 0

    feats["acid_rain"] = 1 if feats["is_rain"] and feats["is_acid"] else 0
    feats["acid_snow"] = 1 if feats["is_snow"] and feats["is_acid"] else 0

    feats["event_quantity"] = event["quantity"]
    feats["Event_quantity"] = event["quantity"]

    feats["Temp"] = event["temp"]
    feats["temperature"] = event["temp"]

    feats["event_intensity"] = event["quantity"] * event["temp"]
    feats["snow_melt_potential"] = feats["is_snow"] * event["temp"]
    feats["acid_intensity"] = feats["is_acid"] * event["quantity"]

    return feats

# =========================================================
# SHAP explanation logic (SAFE)
# =========================================================
def shap_explanation(x_row, pred, shap_vals, top_k=3):

    feature_names = x_row.index
    contrib = shap_vals
    abs_vals = np.abs(contrib)
    order = np.argsort(abs_vals)[::-1]

    qty = x_row.get("event_quantity", x_row.get("Event_quantity", 0))
    acid_val = x_row.get("acid_intensity", 0)

    risk = "HIGH-RISK" if pred >= threshold else "SAFE"

    reasons = []
    for i in order:
        if len(reasons) >= top_k:
            break

        name = feature_names[i]
        val = contrib[i]

        if risk == "HIGH-RISK" and val <= 0:
            continue
        if risk == "SAFE" and val >= 0:
            continue

        if name == "event_intensity":
            reasons.append("A strong weather event increased the leachate.")
        elif name == "event_quantity":
            reasons.append("Higher precipitation volume increased runoff.")
        elif name == "acid_intensity":
            reasons.append("Acidic conditions increased material dissolution.")
        elif name.startswith("K_"):
            reasons.append("Potassium levels influenced leachate chemistry.")
        elif name.startswith("Mg_"):
            reasons.append("Magnesium content affected drainage behaviour.")
        else:
            reasons.append("Rock chemistry contributed to the outcome.")

    return risk, list(dict.fromkeys(reasons))

# =========================================================
# Run Prediction
# =========================================================
if st.button("Run Prediction"):

    explainer = shap.TreeExplainer(rf_model)

    st.subheader("Prediction Results")

    preds = []

    for i, event in enumerate(sequence):

        event_feats = build_event_features(event)

        x = pd.concat(
            [rock_features, pd.Series(event_feats)],
            axis=0
        ).to_frame().T

        # Ensure full feature parity
        for col in feature_cols:
            if col not in x.columns:
                x[col] = 0

        x = x[feature_cols]
        x_scaled = scaler.transform(x)

        pred = rf_model.predict(x_scaled)[0]
        preds.append(pred)

        shap_vals = explainer.shap_values(x)[0]

    # Threshold
    threshold = np.mean(preds) + np.std(preds)

    # Display results
    for i, pred in enumerate(preds):

        st.markdown(f"### Event {i+1}")

        x = pd.concat(
            [rock_features, pd.Series(build_event_features(sequence[i]))],
            axis=0
        ).to_frame().T

        for col in feature_cols:
            if col not in x.columns:
                x[col] = 0

        x = x[feature_cols]

        shap_vals = explainer.shap_values(x)[0]

        risk, reasons = shap_explanation(x.iloc[0], pred, shap_vals)

        st.write(f"**Predicted Leachate:** {pred:.2f}")
        st.write(f"**Risk Level:** {risk}")

        for r in reasons:
            st.write("•", r)



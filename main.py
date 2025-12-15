import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap

# =========================================================
# Page setup
# =========================================================
st.set_page_config(page_title="Leachate Prediction System", layout="centered")
st.title("Leachate Prediction from Rock & Event Sequence")

st.write(
    """
This application predicts **leachate volume after each event** using:
- Rock chemical analysis  
- A sequence of weather events  
- SHAP-based human-readable explanations
"""
)

# =========================================================
# Load trained artifacts
# =========================================================
rf_model = joblib.load("rf_model (1).joblib")
scaler = joblib.load("scaler (1).joblib")
feature_cols = joblib.load("feature_cols (1).joblib")

# Load preprocessed training data (for rock chemistry lookup)
df = pd.read_csv("inference_data.csv")

# =========================================================
# Sidebar — Rock selection
# =========================================================
st.sidebar.header("1️⃣ Select Rock")

rock_id = st.sidebar.selectbox(
    "Rock number",
    sorted(df["Rock_number"].unique())
)

rock_row = df[df["Rock_number"] == rock_id].iloc[0]

rock_features = rock_row[
    [c for c in feature_cols if c.endswith("_rock")]
]

# =========================================================
# Sidebar — Sequence input
# =========================================================
st.sidebar.header("2️⃣ Enter Sequence S")

st.sidebar.write(
    "Format per line: **event, acidity, temperature**\n\n"
    "`rain,0.6,12`\n"
    "`snow,0.2,-1`"
)

sequence_text = st.sidebar.text_area(
    "Sequence S",
    value="rain,0.6,12\nsnow,0.2,-1",
    height=150
)

# =========================================================
# Parse sequence
# =========================================================
sequence = []

for line in sequence_text.split("\n"):
    if line.strip():
        e, a, t = line.split(",")
        sequence.append({
            "Type_event": e.strip(),
            "Acid": float(a),
            "Temp": float(t)
        })

# =========================================================
# Feature builder (MATCHES TRAINING)
# =========================================================
def build_event_features(event):
    is_rain = int(event["Type_event"] == "rain")
    is_snow = int(event["Type_event"] == "snow")
    is_acid = int(event["Acid"] > 0)

    acid_rain = int(is_rain and is_acid)
    acid_snow = int(is_snow and is_acid)

    event_quantity = 150  # fixed as in training
    temperature = event["Temp"]

    return {
        "Event_quantity": event_quantity,
        "Temp": temperature,
        "is_rain": is_rain,
        "is_snow": is_snow,
        "is_acid": is_acid,
        "acid_rain": acid_rain,
        "acid_snow": acid_snow,
        "event_quantity": event_quantity,
        "temperature": temperature,
        "event_intensity": event_quantity * temperature,
        "snow_melt_potential": is_snow * temperature,
        "acid_intensity": is_acid * event_quantity
    }

# =========================================================
# SHAP explanation (Bruno-style)
# =========================================================
def shap_explanation(x_row, pred, shap_values, threshold=100, top_k=3):

    risk = "HIGH" if pred >= threshold else "LOW"
    order = np.argsort(np.abs(shap_values))[::-1]

    def acidity_label(v):
        if v < 0.1: return "very low acidity"
        if v < 0.3: return "low acidity"
        if v < 0.7: return "moderate acidity"
        return "very high acidity"

    explanations = []

    for i in order:
        if len(explanations) >= top_k:
            break

        name = feature_cols[i]
        val = shap_values[i]
        sign = np.sign(val)

        acid_val = x_row["acid_intensity"]
        qty = x_row["event_quantity"]

        if name == "event_intensity":
            explanations.append(
                "A strong precipitation event increased leachate."
                if sign > 0 else
                "A weak event helped keep leachate low."
            )

        elif name == "acid_intensity":
            explanations.append(
                f"The event had {acidity_label(acid_val)}, increasing leachate."
                if sign > 0 else
                f"{acidity_label(acid_val)} helped limit leachate."
            )

        elif name.startswith("K_"):
            explanations.append(
                "Higher potassium increased leachate."
                if sign > 0 else
                "Lower potassium helped control leachate."
            )

        elif name.startswith("Carbonate"):
            explanations.append(
                "Higher carbonate increased leachate."
                if sign > 0 else
                "Lower carbonate reduced leachate."
            )

        else:
            explanations.append(
                "Water chemistry increased leachate."
                if sign > 0 else
                "Stable chemistry kept leachate low."
            )

    return risk, list(dict.fromkeys(explanations))

# =========================================================
# Run prediction
# =========================================================
if st.button("Run Prediction"):

    explainer = shap.TreeExplainer(rf_model)   # ✅ outside loop

    st.subheader("Prediction Results")

    for i, event in enumerate(sequence):

        event_feats = build_event_features(event)

        x = pd.concat(
            [rock_features, pd.Series(event_feats)],
            axis=0
        ).to_frame().T

        # Ensure inference data matches training features
        for col in feature_cols:
            if col not in x.columns:
                x[col] = 0

        x = x[feature_cols]

        # Prediction uses scaled data
        x_scaled = scaler.transform(x)
        pred = rf_model.predict(x_scaled)[0]

        # SHAP uses unscaled data
        shap_vals = explainer.shap_values(x)[0]

        risk, reasons = shap_explanation(
            x.iloc[0], pred, shap_vals
        )


        st.markdown(f"### Event {i+1}")
        st.write(f"**Predicted Leachate:** {pred:.2f}")
        st.write(f"**Risk Level:** {risk}")

        st.write("**Explanation:**")
        for r in reasons:
            st.write("•", r)




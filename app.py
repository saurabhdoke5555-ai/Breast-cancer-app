

import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt


st.set_page_config(page_title="Breast Cancer Diagnosis Predictor", page_icon="🩺", layout="wide")

MODELS_DIR = "models"

# Load trained artifacts 

@st.cache_resource
def load_artifacts():
    model_files = {
        "Logistic Regression": "logistic_regression.pkl",
        "Support Vector Machine": "support_vector_machine.pkl",
        "Random Forest (Tuned)": "random_forest_tuned.pkl",
        "Random Forest": "random_forest.pkl",
        "K-Nearest Neighbors": "k-nearest_neighbors.pkl",
        "Decision Tree": "decision_tree.pkl",
        "Naive Bayes": "naive_bayes.pkl",
    }
    models = {}
    for name, fname in model_files.items():
        try:
            models[name] = joblib.load(f"{MODELS_DIR}/{fname}")
        except FileNotFoundError:
            pass  # model wasn't saved; skip gracefully

    scaler = joblib.load(f"{MODELS_DIR}/scaler.pkl")
    with open(f"{MODELS_DIR}/metadata.json") as f:
        metadata = json.load(f)
    return models, scaler, metadata


try:
    models, scaler, metadata = load_artifacts()
except FileNotFoundError:
    st.error(
        "Model files not found. Please run the Jupyter notebook "
        "(FA2_CaseStudy_BreastCancer.ipynb) first — it trains the models and saves "
        "them into the 'models/' folder that this app reads from."
    )
    st.stop()

feature_names = metadata["feature_names"]
feature_means = metadata["feature_means"]
feature_stds = metadata["feature_stds"]
feature_min = metadata["feature_min"]
feature_max = metadata["feature_max"]
target_names = metadata["target_names"]          # ['malignant', 'benign']
top_features = metadata["top_importance_features"]  # 10 most predictive features

# Header

st.title("🩺 Breast Cancer Diagnosis Predictor")


# Sidebar — model selection & mode

st.sidebar.header("⚙️ Settings")
model_name = st.sidebar.selectbox("Choose a model", list(models.keys()), index=0)
model = models[model_name]

mode = st.sidebar.radio(
    "Input mode",
    ["Quick (top 10 features)", "Advanced (all 30 features)", "Load a sample patient"],
    index=0,
)

st.sidebar.markdown("---")
st.sidebar.caption(
    "All models were trained on the Breast Cancer Wisconsin (Diagnostic) dataset "
    "(569 samples, 30 features)"
)

# Sample patients 

SAMPLES = {
    "Typical benign-leaning profile": {f: feature_means[f] - 0.6 * feature_stds[f] for f in feature_names},
    "Typical malignant-leaning profile": {f: feature_means[f] + 0.8 * feature_stds[f] for f in feature_names},
}

# Build the input form

st.subheader("1️⃣ Enter Patient / Sample Measurements")

user_values = dict(feature_means)  # start every feature at the dataset mean

if mode == "Load a sample patient":
    sample_choice = st.selectbox("Pick a sample profile", list(SAMPLES.keys()))
    user_values.update(SAMPLES[sample_choice])
    st.info("Sample values loaded below. Switch to Advanced mode to fine-tune them.")
    cols = st.columns(3)
    for i, feat in enumerate(feature_names):
        cols[i % 3].metric(feat, f"{user_values[feat]:.3f}")

elif mode == "Quick (top 10 features)":
    st.caption(
        "These are the 10 features the tuned Random Forest found most predictive. "
        "All other features are held at the dataset average."
    )
    cols = st.columns(2)
    for i, feat in enumerate(top_features):
        lo, hi = float(feature_min[feat]), float(feature_max[feat])
        default = float(feature_means[feat])
        step = (hi - lo) / 200 if hi > lo else 0.01
        user_values[feat] = cols[i % 2].slider(
            feat, min_value=lo, max_value=hi, value=default, step=step
        )

else:  # Advanced mode — every feature
    st.caption("Full control over all 30 measurements, grouped by type.")
    groups = {
        "Mean values": [f for f in feature_names if f.startswith("mean")],
        "Standard error values": [f for f in feature_names if f.endswith("error")],
        "Worst (largest) values": [f for f in feature_names if f.startswith("worst")],
    }
    for group_name, feats in groups.items():
        with st.expander(group_name, expanded=(group_name == "Mean values")):
            cols = st.columns(2)
            for i, feat in enumerate(feats):
                lo, hi = float(feature_min[feat]), float(feature_max[feat])
                default = float(feature_means[feat])
                step = (hi - lo) / 200 if hi > lo else 0.01
                user_values[feat] = cols[i % 2].number_input(
                    feat, min_value=lo, max_value=hi, value=default, step=step, format="%.4f"
                )

# Predict

st.subheader("2️⃣ Prediction")

if st.button("🔍 Predict Diagnosis", type="primary"):
    input_df = pd.DataFrame([user_values])[feature_names]  # enforce correct column order
    input_scaled = scaler.transform(input_df)

    prediction = model.predict(input_scaled)[0]
    label = target_names[prediction]

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(input_scaled)[0]
        confidence = proba[prediction]
    else:
        proba = None
        confidence = None

    col1, col2 = st.columns([1, 1])
    with col1:
        if label == "malignant":
            st.error(f"### Prediction: **Malignant** ⚠️")
        else:
            st.success(f"### Prediction: **Benign** ✅")
        if confidence is not None:
            st.metric("Model confidence", f"{confidence * 100:.1f}%")
        st.caption(f"Model used: **{model_name}**")

    with col2:
        if proba is not None:
            fig, ax = plt.subplots(figsize=(4, 3))
            ax.bar(target_names, proba, color=["#e74c3c", "#2ecc71"])
            ax.set_ylim(0, 1)
            ax.set_ylabel("Predicted probability")
            ax.set_title("Class probabilities")
            for i, p in enumerate(proba):
                ax.text(i, p + 0.02, f"{p:.2f}", ha="center")
            st.pyplot(fig)

    with st.expander("Compare entered values against dataset averages"):
        compare_feats = top_features
        compare_df = pd.DataFrame({
            "Entered value": [user_values[f] for f in compare_feats],
            "Dataset average": [feature_means[f] for f in compare_feats],
        }, index=compare_feats)
        st.bar_chart(compare_df)


else:
    st.info("Set the measurements above and click **Predict Diagnosis** to see the result.")

# Footer / about

with st.expander("ℹ️ About this app & the models"):
    st.markdown(
        """
        - **Dataset:** Breast Cancer Wisconsin (Diagnostic), 569 samples, 30 numeric features
          (`sklearn.datasets.load_breast_cancer`).
        - **Models available:** Logistic Regression, Decision Tree, Random Forest (plain and
          `GridSearchCV`-tuned), Support Vector Machine, K-Nearest Neighbors, Naive Bayes.
        - **Preprocessing:** All inputs are standardized with the same `StandardScaler`
          fitted on the training data in the notebook, before being passed to the model.
        - **Source:** Trained in `FA2_CaseStudy_BreastCancer.ipynb`;
        """
    )

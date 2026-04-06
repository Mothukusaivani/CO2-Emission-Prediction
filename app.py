import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.express as px

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="CO₂ ML Intelligence Dashboard",
    page_icon="🌍",
    layout="wide"
)

# ---------------- CUSTOM STYLING ----------------
st.markdown("""
<style>
.main { background-color: #f0f2f6; }
h1 { color: #1f77b4; text-align: center; }
h2, h3, h4 { color: #1f77b4; }
.stButton>button { background-color: #1f77b4; color: white; }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# LOAD MODELS
# -------------------------------------------------
@st.cache_resource
def load_models():
    models = {
        "Linear Regression": joblib.load("saved_models/linear_regression.pkl"),
        "XGBoost": joblib.load("saved_models/xgboost.pkl"),
        "MLP Regressor": joblib.load("saved_models/mlp_regressor.pkl"),
        "LightGBM": joblib.load("saved_models/lightgbm.pkl"),
        "CatBoost": joblib.load("saved_models/catboost.pkl")
    }

    scaler = joblib.load("saved_models/scaler.pkl")
    feature_columns = joblib.load("saved_models/feature_columns.pkl")

    return models, scaler, feature_columns


# -------------------------------------------------
# LOAD DATA FILES
# -------------------------------------------------
@st.cache_data
def load_data_files():
    metrics_df = pd.read_csv("saved_models/model_comparison_results.csv")
    test_df = pd.read_csv("saved_models/test_predictions.csv")

    metrics_df.columns = metrics_df.columns.str.strip().str.lower()
    test_df.columns = test_df.columns.str.strip().str.lower()

    return metrics_df, test_df


models, scaler, feature_columns = load_models()
metrics_df, test_df = load_data_files()

# -------------------------------------------------
# HEADER
# -------------------------------------------------
st.title("🌍 CO₂ Emission ML Intelligence Dashboard")

# -------------------------------------------------
# MODEL SELECTION
# -------------------------------------------------
selected_model_name = st.selectbox("Select Model", list(models.keys()))
model = models[selected_model_name]

# -------------------------------------------------
# MODEL PERFORMANCE
# -------------------------------------------------
st.subheader("📊 Model Performance Overview")

selected_metrics = metrics_df[metrics_df["model"] == selected_model_name]

col1, col2, col3 = st.columns(3)

r2 = selected_metrics["r2"].values[0] if "r2" in selected_metrics else np.nan
rmse = selected_metrics["rmse"].values[0] if "rmse" in selected_metrics else np.nan
# mae = selected_metrics["mae"].values[0] if "mae" in selected_metrics else np.nan

col1.metric("R² Score", round(r2, 4))
col2.metric("RMSE", round(rmse, 2))
# col3.metric("MAE", round(mae, 2))

fig_compare = px.bar(
    metrics_df,
    x="model",
    y="r2",
    title="R² Comparison Across Models",
    text="r2"
)

st.plotly_chart(fig_compare, use_container_width=True)

# -------------------------------------------------
# ACTUAL VS PREDICTED
# -------------------------------------------------
st.subheader("📈 Actual vs Predicted")

if "model" in test_df.columns:

    model_test = test_df[test_df["model"] == selected_model_name].copy()

    fig_actual_pred = px.scatter(
        model_test,
        x="actual",
        y="predicted",
        title="Actual vs Predicted CO₂",
        opacity=0.6
    )

    fig_actual_pred.add_shape(
        type="line",
        x0=model_test["actual"].min(),
        y0=model_test["actual"].min(),
        x1=model_test["actual"].max(),
        y1=model_test["actual"].max(),
        line=dict(dash="dash")
    )

    st.plotly_chart(fig_actual_pred, use_container_width=True)

# -------------------------------------------------
# ERROR DISTRIBUTION
# -------------------------------------------------
st.subheader("📉 Prediction Error Distribution")

if "actual" in test_df.columns and "predicted" in test_df.columns:

    model_test["error"] = model_test["actual"] - model_test["predicted"]

    fig_error = px.histogram(
        model_test,
        x="error",
        nbins=50,
        title="Prediction Error Distribution"
    )
    st.plotly_chart(fig_error, use_container_width=True)

# -------------------------------------------------
# FEATURE IMPORTANCE
# -------------------------------------------------
st.subheader("🧠 Feature Importance")

importance = None

if hasattr(model, "feature_importances_"):
    importance = model.feature_importances_

elif hasattr(model, "coef_"):
    importance = np.abs(model.coef_)

if importance is not None:

    imp_df = pd.DataFrame({
        "Feature": feature_columns,
        "Importance": importance
    }).sort_values(by="Importance", ascending=False)

    fig_importance = px.bar(
        imp_df.head(15),
        x="Importance",
        y="Feature",
        orientation="h",
        title="Top Feature Importance"
    )

    st.plotly_chart(fig_importance, use_container_width=True)

else:
    st.info("Feature importance not available for this model.")

# -------------------------------------------------
# CUSTOM PREDICTION
# -------------------------------------------------
st.subheader("🔮 Custom CO₂ Prediction")

with st.expander("Enter Feature Values"):

    input_values = {}

    # TEXT INPUT (no dropdown)
    input_values["country"] = st.text_input("Country", "India")

    input_values["year"] = st.number_input("Year", value=2022)
    input_values["population"] = st.number_input("Population", value=0.0)
    input_values["gdp"] = st.number_input("GDP", value=0.0)
    input_values["cement_co2"] = st.number_input("Cement CO2", value=0.0)
    input_values["coal_co2"] = st.number_input("Coal CO2", value=0.0)
    input_values["oil_co2"] = st.number_input("Oil CO2", value=0.0)
    input_values["gas_co2"] = st.number_input("Gas CO2", value=0.0)

    if st.button("Predict CO₂ Emission"):

        # Create dataframe with model features
        input_full = pd.DataFrame(columns=feature_columns)
        input_full.loc[0] = 0

        for key, value in input_values.items():

            if key == "country":

                country_col = "country_" + value.lower().replace(" ", "_")

                if country_col in input_full.columns:
                    input_full[country_col] = 1

            else:
                if key in input_full.columns:
                    input_full[key] = value

        # Scaling if required
        if selected_model_name in ["Linear Regression", "MLP Regressor"]:
            input_final = scaler.transform(input_full)
        else:
            input_final = input_full.values

        prediction = model.predict(input_final)[0]

        st.success(f"Predicted CO₂ Emission: {prediction:.2f}")

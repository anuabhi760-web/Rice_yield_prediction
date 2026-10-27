"""
STREAMLIT DASHBOARD — Rice Yield Prediction
----------------------------------------------------------------------
Run: pip install streamlit plotly
     streamlit run app.py

Prerequisite: run 01_train_final_models.py and 02_train_state_space_model.py
first (this app loads their saved outputs — it does NOT retrain anything,
so it launches instantly).

Layout:
  - Model comparison metrics (Objective 3 / RQ2)
  - District-level history chart: actual yield, RF/XGBoost predictions,
    and SSM forecast with 95% confidence band (Objective 5 / RQ3)
  - Global feature importance via SHAP (Objective 4 / RQ4)
  - "What-if" interactive prediction panel — lets a user adjust
    rainfall/temperature/soil inputs and see XGBoost's predicted yield
    update live. This is the closest thing to your proposal's planned
    AI chatbot (Objective 6) without the LLM integration itself —
    frame it in your presentation as "the underlying prediction engine
    the chatbot would call."
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib  # only for random_forest.pkl, which 01_train_final_models.py
               # now saves with joblib's compression to keep file size
               # under GitHub's 25MB browser-upload limit
import os
import plotly.graph_objects as go

st.set_page_config(page_title="Rice Yield Prediction Dashboard", layout="wide")

# The chatbot lives in chatbot.py (same folder). If that file is missing the
# dashboard still works and the chat page explains what to do.
try:
    from chatbot import RiceKnowledge, chat, llm_config
    CHATBOT_AVAILABLE, CHATBOT_ERROR = True, ""
except Exception as _e:  # noqa
    CHATBOT_AVAILABLE, CHATBOT_ERROR = False, f"{type(_e).__name__}: {_e}"


def show_plotly(fig):
    """st.plotly_chart without tripping over Streamlit's use_container_width deprecation."""
    try:
        st.plotly_chart(fig, use_container_width=True)
    except TypeError:
        st.plotly_chart(fig)

# ------------------------------------------------------------------
# All paths anchored to THIS script's location, not the current working
# directory — Streamlit Cloud (and most deploy environments) don't
# guarantee cwd == this folder, so "../data/..." style relative paths
# break unpredictably. This pattern works regardless of how/where the
# app is launched from.
# ------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")


# ------------------------------------------------------------------
# Load everything once (cached — the app should feel instant)
# ------------------------------------------------------------------
def _find_file(names, required=True):
    """Look next to app.py, then in ./data, for the first existing file in `names`."""
    names = [names] if isinstance(names, str) else names
    for folder in (BASE_DIR, DATA_DIR):
        for n in names:
            p = os.path.join(folder, n)
            if os.path.isfile(p):
                return p
    if required:
        raise FileNotFoundError(names[0] if len(names) == 1 else " or ".join(names))
    return None


@st.cache_data
def load_data():
    full_history = pd.read_csv(_find_file(["rice_model_ready_full.csv", "rice_dataset_final.csv",
                                           "rice_model_ready_baseline.csv"]))
    test_predictions = pd.read_csv(_find_file("test_predictions.csv"))
    model_summary = pd.read_csv(_find_file("model_summary.csv"))
    shap_importance = pd.read_csv(_find_file("shap_global_importance.csv"), index_col=0)
    ssm_forecasts = pd.read_csv(_find_file("ssm_forecasts.csv"))
    ssm_diagnostics = pd.read_csv(_find_file("ssm_diagnostics.csv"))
    return full_history, test_predictions, model_summary, shap_importance, ssm_forecasts, ssm_diagnostics


@st.cache_resource
def load_models():
    # random_forest*.pkl are saved with joblib compression (keeps them under GitHub's
    # 25MB browser-upload limit), so they must be loaded with joblib.
    rf = joblib.load(_find_file("random_forest.pkl"))
    with open(_find_file("xgboost.pkl"), "rb") as f:
        xgb_model = pickle.load(f)
    with open(_find_file("feature_cols.pkl"), "rb") as f:
        feature_cols = pickle.load(f)
    # "Production" models refit on ALL years (01_train_final_models.py). Trees can't predict above
    # what they saw in training, so the 1966-2005 evaluation models under-predict next season.
    rf_prod = xgb_prod = None
    p = _find_file("random_forest_full.pkl", required=False)
    if p:
        rf_prod = joblib.load(p)
    p = _find_file("xgboost_full.pkl", required=False)
    if p:
        with open(p, "rb") as f:
            xgb_prod = pickle.load(f)
    return rf, xgb_model, feature_cols, rf_prod, xgb_prod


try:
    full_history, test_predictions, model_summary, shap_importance, ssm_forecasts, ssm_diagnostics = load_data()
    rf_model, xgb_model, feature_cols, rf_prod, xgb_prod = load_models()
except FileNotFoundError as e:
    st.error(f"Missing file: {e}. Run 01_train_final_models.py and "
             "02_train_state_space_model.py in this folder before launching the dashboard.")
    st.stop()

# Models used for NEXT-season forecasts (chatbot + what-if panel)
_have_prod = rf_prod is not None and xgb_prod is not None
forecast_models = {"Random Forest": rf_prod if rf_prod is not None else rf_model,
                   "XGBoost": xgb_prod if xgb_prod is not None else xgb_model}
xgb_forecast = forecast_models["XGBoost"]
MODELS_NOTE = "" if _have_prod else (
    "⚠️ *These estimates come from the evaluation models trained on 1966–2005 only, which tend to under-predict "
    "today's higher yields. Run `01_train_final_models.py` to create the all-years production models "
    "(`random_forest_full.pkl`, `xgboost_full.pkl`).*")

TARGET = "RICE YIELD (Kg per ha)"

# ------------------------------------------------------------------
# Header + model comparison
# ------------------------------------------------------------------
st.title("🌾 Rice Yield Prediction — District-Level Dashboard")
st.caption("Crop Yield Prediction Based on Climate, Soil and Environmental Factors — "
           "JAIN Deemed-to-be University")

# ------------------------------------------------------------------
# Chatbot page (second view; the dashboard below is unchanged)
# ------------------------------------------------------------------
MAX_LLM_CALLS_PER_SESSION = 30       # protects your API credit on a public app


def _secret(name):
    """Read a key from Streamlit secrets or environment variables — never hard-code keys in the repo."""
    try:
        val = st.secrets[name]
    except Exception:
        val = None
    return val or os.environ.get(name)


@st.cache_resource
def get_knowledge():
    return RiceKnowledge.from_dir(BASE_DIR)


def _queue_prompt(text):
    st.session_state["pending_prompt"] = text


def _clear_chat():
    st.session_state["chat"], st.session_state["chat_ctx"] = [], {}


def render_chatbot_page():
    st.subheader("💬 Rice Yield Assistant")
    st.caption("Ask about district yields, next-season outlooks, trends, what drives yield, or model accuracy. "
               "Answers come from this project's own data and models.")
    if not CHATBOT_AVAILABLE:
        st.error(f"chatbot.py could not be loaded ({CHATBOT_ERROR}). Make sure chatbot.py is uploaded next to app.py.")
        return
    try:
        kb = get_knowledge()
    except Exception as ex:
        st.error(f"The chatbot could not load its data: {ex}")
        return

    ss = st.session_state
    ss.setdefault("chat", [])
    ss.setdefault("chat_ctx", {})
    ss.setdefault("llm_calls", 0)

    cfg = llm_config(_secret)
    with st.sidebar.expander("🔑 Optional: AI-written replies"):
        st.caption("Without a key the chatbot still answers from the data (in English). With a key it also understands "
                   "Hindi / Telugu / Bengali / Tamil questions and words its replies more naturally. "
                   "A key pasted here is kept for this browser session only.")
        pasted = st.text_input("Anthropic API key", type="password")
        if pasted:
            cfg = {"provider": "anthropic", "key": pasted.strip(),
                   "model": _secret("ANTHROPIC_MODEL") or "claude-haiku-5-5"}
    left = max(0, MAX_LLM_CALLS_PER_SESSION - ss["llm_calls"])
    if cfg:
        st.sidebar.success(f"AI wording: ON ({cfg['provider']}) — {left} calls left this session")
    else:
        st.sidebar.info("AI wording: OFF — built-in answers only")
    st.sidebar.button("🗑️ Clear chat", on_click=_clear_chat)

    if not ss["chat"]:
        st.info("👋 Hi! Pick an example or type your own question below.")
        examples = ["What will rice yield be in Patna this Kharif?", "Top 5 districts in Punjab",
                    "Compare Patna and Gaya", "What factors drive rice yield?",
                    "How accurate are the models?", "Advice for farmers in Raichur"]
        cols = st.columns(2)
        for i, ex in enumerate(examples):
            cols[i % 2].button(ex, key=f"ex_{i}", on_click=_queue_prompt, args=(ex,))

    prompt = st.chat_input("Ask about rice yield…")
    pending = ss.pop("pending_prompt", None)
    prompt = prompt or pending
    if prompt:
        history = [{"role": m["role"], "content": m["content"]} for m in ss["chat"][-6:]]
        ss["chat"].append({"role": "user", "content": prompt})
        try:
            with st.spinner("Looking that up…"):
                ans = chat(kb, prompt, ss["chat_ctx"], forecast_models, feature_cols, llm_cfg=cfg,
                           history=history, calls_left=left, models_note=MODELS_NOTE)
            ss["llm_calls"] += ans.llm_calls
            ss["chat"].append({"role": "assistant", "content": ans.text, "table": ans.table, "chart": ans.chart,
                               "suggestions": ans.suggestions, "llm_used": ans.llm_used, "llm_error": ans.llm_error})
        except Exception as ex:                       # never let one bad question crash the page
            ss["chat"].append({"role": "assistant",
                               "content": f"Sorry — something went wrong answering that ({type(ex).__name__}). "
                                          "Try rephrasing, e.g. *yield in Patna in 2020*."})

    last_assistant = max((i for i, m in enumerate(ss["chat"]) if m["role"] == "assistant"), default=-1)
    for m in ss["chat"]:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            if m.get("table") is not None:
                st.dataframe(m["table"], hide_index=True)
            if m.get("chart") is not None:
                st.line_chart(m["chart"])
            if m.get("llm_used"):
                st.caption("✨ Reworded by AI using only this project's data")
            if m.get("llm_error"):
                st.caption(f"⚠️ AI wording unavailable ({m['llm_error']}) — showing the built-in answer.")
    if last_assistant >= 0 and ss["chat"][last_assistant].get("suggestions"):
        st.caption("You could also ask:")
        for j, sug in enumerate(ss["chat"][last_assistant]["suggestions"][:3]):
            st.button(sug, key=f"sug_{last_assistant}_{j}", on_click=_queue_prompt, args=(sug,))


view = st.sidebar.radio("View", ["📈 Dashboard", "💬 Chatbot"])
if view == "💬 Chatbot":
    render_chatbot_page()
    st.stop()


st.subheader("Model Comparison (temporal split: train 1966-2005, test 2006-2025)")
cols = st.columns(len(model_summary))
for col, (_, row) in zip(cols, model_summary.iterrows()):
    col.metric(row["model"], f"R² = {row['test_r2']:.3f}", f"RMSE {row['test_rmse']:.0f} kg/ha")

with st.expander("Why don't the models clearly beat 'same as last year'?"):
    st.write("""
    A naive baseline (predict = last year's yield) scores R² ≈ 0.90 over 2006–2025 because rice yield is strongly
    autocorrelated. Most of that score comes from 2016–2025: in this dataset those years look unusually smooth
    (possibly extrapolated rather than observed), which makes "same as last year" almost perfect. The table below
    splits the scores by period — 2006–2015, where the data still shows real year-to-year swings, is the fairer
    comparison. (Columns appear after re-running `01_train_final_models.py`.)
    """)
    _cols = [c for c in ["model", "test_r2", "r2_2006_2015", "r2_2016_2025", "test_rmse"] if c in model_summary.columns]
    st.dataframe(model_summary[_cols], hide_index=True)

st.divider()

# ------------------------------------------------------------------
# District selector
# ------------------------------------------------------------------
st.subheader("District-Level Forecast Explorer")
states = sorted(full_history["State Name"].unique())
col1, col2 = st.columns(2)
selected_state = col1.selectbox("State", states)
districts_in_state = sorted(full_history.loc[full_history["State Name"] == selected_state, "Dist Name"].unique())
selected_district = col2.selectbox("District", districts_in_state)

dist_history = full_history[full_history["Dist Name"] == selected_district].sort_values("Year")
dist_test_pred = test_predictions[test_predictions["Dist Name"] == selected_district].sort_values("Year")
dist_ssm = ssm_forecasts[ssm_forecasts["Dist Name"] == selected_district].sort_values("Year")

# ------------------------------------------------------------------
# History + forecast chart
# ------------------------------------------------------------------
fig = go.Figure()

fig.add_trace(go.Scatter(x=dist_history["Year"], y=dist_history[TARGET],
                          mode="lines+markers", name="Actual Yield",
                          line=dict(color="black", width=2)))

if not dist_test_pred.empty:
    fig.add_trace(go.Scatter(x=dist_test_pred["Year"], y=dist_test_pred["rf_pred"],
                              mode="lines", name="Random Forest Prediction",
                              line=dict(color="#2ca25f", dash="dot")))
    fig.add_trace(go.Scatter(x=dist_test_pred["Year"], y=dist_test_pred["xgb_pred"],
                              mode="lines", name="XGBoost Prediction",
                              line=dict(color="#e34a33", dash="dot")))

if not dist_ssm.empty:
    fig.add_trace(go.Scatter(x=dist_ssm["Year"], y=dist_ssm["ssm_forecast"],
                              mode="lines", name="SSM Forecast (trend + climate)",
                              line=dict(color="#3182bd", width=2)))
    fig.add_trace(go.Scatter(
        x=pd.concat([dist_ssm["Year"], dist_ssm["Year"][::-1]]),
        y=pd.concat([dist_ssm["ssm_ci_upper"], dist_ssm["ssm_ci_lower"][::-1]]),
        fill="toself", fillcolor="rgba(49,130,189,0.15)", line=dict(color="rgba(0,0,0,0)"),
        name="SSM 95% Confidence Interval", showlegend=True,
    ))
else:
    st.info(f"No SSM forecast available for {selected_district} "
            "(likely too few years of history — see ssm_diagnostics.csv)")

fig.add_vline(x=2005.5, line_dash="dash", line_color="gray",
              annotation_text="Train / Test split")
fig.update_layout(title=f"{selected_district}, {selected_state} — Yield History & Forecasts",
                   xaxis_title="Year", yaxis_title="Yield (kg/ha)", height=500,
                   legend=dict(orientation="h", yanchor="bottom", y=1.02))
show_plotly(fig)

# SSM calibration note for this district
if not dist_ssm.empty:
    diag_row = ssm_diagnostics[ssm_diagnostics["district"] == selected_district]
    if not diag_row.empty and diag_row.iloc[0]["status"] == "ok":
        coverage = diag_row.iloc[0]["ci_coverage"]
        st.caption(f"SSM 95% CI coverage for this district: {coverage:.0%} "
                    "(fraction of actual test-year values falling inside the shaded band — "
                    "close to 95% means the uncertainty estimate is trustworthy)")

st.divider()

# ------------------------------------------------------------------
# Feature importance (SHAP)
# ------------------------------------------------------------------
st.subheader("What Drives Yield Predictions? (SHAP Feature Importance — XGBoost)")
top_n = st.slider("Number of top features to show", 5, 20, 10)
top_features = shap_importance.head(top_n).sort_values("mean_abs_shap")
fig2 = go.Figure(go.Bar(x=top_features["mean_abs_shap"], y=top_features.index,
                         orientation="h", marker_color="#756bb1"))
fig2.update_layout(title="Mean |SHAP value| — higher means bigger influence on predicted yield",
                    xaxis_title="Mean |SHAP value|", height=400 + top_n * 15)
show_plotly(fig2)

st.divider()

# ------------------------------------------------------------------
# What-if prediction panel
# ------------------------------------------------------------------
st.subheader("What-If Yield Predictor")
st.caption("Adjust climate/soil inputs below to see XGBoost's predicted yield update live — "
           "the same prediction engine the chatbot uses for next-season outlooks.")

# start from this district's most recent row as the baseline, let user override key drivers
baseline_full_row = dist_history.iloc[-1]
baseline_row = baseline_full_row[feature_cols].copy()
baseline_actual_yield = baseline_full_row[TARGET]
baseline_year = int(baseline_full_row["Year"])

wcol1, wcol2, wcol3 = st.columns(3)
rain = wcol1.slider("Annual Rainfall (mm)", 200, 3500, int(baseline_row.get("RAIN_ANNUAL_mm", 1000)))
temp = wcol2.slider("Annual Temperature (°C)", 22.0, 34.0, float(baseline_row.get("TEMP_ANNUAL_C", 28.0)), 0.1)
soil_ph = wcol3.slider("Soil pH", 4.5, 9.0, float(baseline_row.get("SOIL_PHH2O", 6.5)), 0.1)

input_row = baseline_row.copy()
input_row["RAIN_ANNUAL_mm"] = rain
input_row["TEMP_ANNUAL_C"] = temp
input_row["SOIL_PHH2O"] = soil_ph
input_df = pd.DataFrame([input_row])[feature_cols]

predicted_yield = xgb_forecast.predict(input_df)[0]
st.metric("Predicted Rice Yield", f"{predicted_yield:.0f} kg/ha",
          f"{predicted_yield - baseline_actual_yield:.0f} vs. {baseline_year} actual ({baseline_actual_yield:.0f} kg/ha)")

st.caption("⚠️ Data quality note: soil features for districts outside the original 67 with "
           "confirmed ISRIC coverage currently use an interim state-median imputation. "
           "See README.md for details and the re-fetch script (03_fetch_missing_soil.py).")

# yield_spread_forecast_app.py
# -------------------------------------------------------------
# Yield Spread Forecasting with AR/VAR and Recession Probability
# -------------------------------------------------------------
# What this app does
# - Downloads 10Y and 2Y Treasury yields (FRED: DGS10, DGS2)
# - Computes 10y–2y term spread
# - Fits either an AR(p) model to the spread OR a VAR model with macro covariates
# - Forecasts the spread, with confidence intervals
# - Estimates recession probability using a logistic model (USREC as target)
# - Highlights turning points where probability crosses a user-defined threshold
# - Interactive charts and controls designed for recruiter-friendly demo
# -------------------------------------------------------------

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
from pandas_datareader import data as pdr
from datetime import date, timedelta
import statsmodels.api as sm
from statsmodels.tsa.api import VAR
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tools.tools import add_constant
import plotly.graph_objects as go
import plotly.express as px

# =============================
# ---------- UI SETUP ---------
# =============================
st.set_page_config(page_title="Yield Spread Forecasting (AR/VAR)", layout="wide")

HERO_TITLE = "📈 Yield Spread Forecasting — AR/VAR + Recession Probability"
HERO_SUB = "CFA L2 Concepts: Term Structure, Econometrics, Business Cycle Signals"

st.markdown(
    """
    <style>
    :root { --bg:#0d1117; --panel:#161b22; --border:#30363d; --text:#e6edf3; --muted:#9db1d6; }
    html, body, [data-testid="stAppViewContainer"] { background: var(--bg); color: var(--text); }
    [data-testid="stSidebar"] { background: #0c1220; }
    .big { font-size: 1.6rem; font-weight:700; }
    .muted { color: var(--muted); }
    .metric { background: var(--panel); padding: 12px 16px; border:1px solid var(--border); border-radius: 12px; }
    .card { background: var(--panel); padding: 16px; border:1px solid var(--border); border-radius: 16px; }
    .hr { height:1px; background:var(--border); margin: 1rem 0; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(f"<div class='big'>{HERO_TITLE}</div>", unsafe_allow_html=True)
st.markdown(f"<div class='muted'>{HERO_SUB}</div>", unsafe_allow_html=True)
st.markdown("<div class='hr'></div>", unsafe_allow_html=True)

# =============================
# --------- HELPERS -----------
# =============================
@st.cache_data(show_spinner=False)
def load_fred(start: str, end: str) -> pd.DataFrame:
    """Download daily 10Y, 2Y, US recession indicator, unemployment, CPI (YoY), Industrial Production (YoY)."""
    series = {
        "DGS10": "DGS10",   # 10-Year Treasury Constant Maturity Rate (% per annum)
        "DGS2": "DGS2",     # 2-Year Treasury Constant Maturity Rate
        "USREC": "USREC",   # NBER-based Recession Indicators for the US (Monthly, 0/1)
        "UNRATE": "UNRATE", # Unemployment rate (Monthly, %)
        "CPIAUCSL": "CPIAUCSL", # CPI (Monthly, index)
        "INDPRO": "INDPRO", # Industrial Production Index (Monthly)
    }
    df = pd.DataFrame()
    for col, code in series.items():
        s = pdr.DataReader(code, "fred", start, end)
        s.rename(columns={code: col}, inplace=True)
        df = s if df.empty else df.join(s, how="outer")

    # Transformations
    # Forward-fill to align daily freq, interpolate where reasonable
    df = df.sort_index().ffill()

    # Compute term spread
    df["SPREAD"] = df["DGS10"] - df["DGS2"]

    # Convert monthly series to YoY where useful
    # CPI YoY %
    df["CPI_YoY"] = (
        df["CPIAUCSL"].pct_change(12) * 100
    )
    # Industrial Production YoY %
    df["INDPRO_YoY"] = (
        df["INDPRO"].pct_change(12) * 100
    )

    # Trim to available SPREAD data
    df = df.dropna(subset=["SPREAD"]).copy()
    return df


def make_train(df: pd.DataFrame, freq: str = "M"):
    """Resample to desired frequency and prepare modeling frames."""
    if freq == "D":
        d = df.copy()
    elif freq == "W":
        d = df.resample("W-FRI").mean()
    else:
        d = df.resample("M").mean()

    # Align features/target
    model_df = d[["SPREAD", "UNRATE", "CPI_YoY", "INDPRO_YoY", "USREC"]].dropna()
    return model_df


def fit_ar(model_df: pd.DataFrame, max_lags: int = 12):
    """Auto-select AR order by AIC and fit."""
    y = model_df["SPREAD"].dropna()
    best_aic, best_p, best_model = np.inf, None, None
    for p in range(1, max_lags + 1):
        try:
            m = AutoReg(y, lags=p, old_names=False).fit()
            if m.aic < best_aic:
                best_aic, best_p, best_model = m.aic, p, m
        except Exception:
            pass
    return best_model, best_p, best_aic


def fit_var(model_df: pd.DataFrame, max_lags: int = 6):
    """VAR on [SPREAD, UNRATE, CPI_YoY, INDPRO_YoY]."""
    X = model_df[["SPREAD", "UNRATE", "CPI_YoY", "INDPRO_YoY"]].dropna()
    var = VAR(X)
    sel = var.select_order(maxlags=max_lags)
    p = sel.aic or sel.bic or sel.fpe or sel.hqic
    p = int(p) if p is not None else 2
    res = var.fit(maxlags=p)
    return res, p


def forecast_ar(model, steps: int = 12):
    fc = model.get_forecast(steps=steps)
    mean = fc.predicted_mean
    conf = fc.conf_int(alpha=0.05)  # 95% CI
    return mean, conf


def forecast_var(model, steps: int = 12):
    fc = model.forecast_interval(model.endog, steps=steps, alpha=0.05)
    mean = pd.DataFrame(fc[0], columns=model.names)
    lower = pd.DataFrame(fc[1], columns=model.names)
    upper = pd.DataFrame(fc[2], columns=model.names)
    return mean, lower, upper


def fit_recession_logit(model_df: pd.DataFrame, lag_months: int = 6):
    """Logistic regression: USREC ~ lagged spread + macro. Target is whether recession occurs within next lag_months.
    This approximates turning-point probability.
    """
    d = model_df.copy()
    # Define target: recession in next L months (max of USREC ahead)
    d["REC_FWD"] = d["USREC"].rolling(window=lag_months, min_periods=1).max().shift(-lag_months)
    d = d.dropna()

    # Features (you can swap/add more): current spread + macro
    X = d[["SPREAD", "UNRATE", "CPI_YoY", "INDPRO_YoY"]]
    X = add_constant(X, has_constant="add")
    y = d["REC_FWD"]

    logit = sm.Logit(y, X).fit(disp=False)

    # In-sample probabilities
    d["REC_PROB"] = logit.predict(X)

    return logit, d


def make_spread_chart(hist: pd.Series, fc_mean: pd.Series = None, fc_lower=None, fc_upper=None, title="Term Spread (10Y–2Y)"):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist.index, y=hist.values, mode="lines", name="Historical"))
    if fc_mean is not None:
        fig.add_trace(go.Scatter(x=fc_mean.index, y=fc_mean.values, mode="lines", name="Forecast"))
        if fc_lower is not None and fc_upper is not None:
            fig.add_trace(go.Scatter(x=fc_mean.index, y=fc_upper.values, mode="lines", name="Upper 95%", line=dict(dash="dot")))
            fig.add_trace(go.Scatter(x=fc_mean.index, y=fc_lower.values, mode="lines", name="Lower 95%", line=dict(dash="dot"), fill="tonexty", fillcolor="rgba(200,200,255,0.2)"))
    fig.update_layout(template="plotly_dark", height=420, title=title, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    return fig


def make_prob_chart(prob_df: pd.DataFrame, thresh: float = 0.5):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=prob_df.index, y=prob_df["REC_PROB"], mode="lines", name="Recession Probability"))
    fig.add_hline(y=thresh, line_dash="dash", annotation_text=f"Threshold {thresh:.0%}")
    fig.update_layout(template="plotly_dark", height=300, title="Recession Probability (Next-Period Window)")
    return fig


def turning_points(prob_df: pd.DataFrame, thresh: float = 0.5):
    s = prob_df["REC_PROB"].copy()
    cross_up = (s.shift(1) < thresh) & (s >= thresh)
    cross_dn = (s.shift(1) >= thresh) & (s < thresh)
    ups = s[cross_up]
    dns = s[cross_dn]
    return ups.index.to_list(), dns.index.to_list()

# =============================
# -------- SIDEBAR -------------
# =============================
st.sidebar.header("⚙️ Controls")
start = st.sidebar.date_input("Start date", value=date(1990,1,1))
end = st.sidebar.date_input("End date", value=date.today())
freq = st.sidebar.selectbox("Frequency", ["M", "W", "D"], index=0, help="Resample data: Monthly (default), Weekly, or Daily")
model_type = st.sidebar.radio("Model Type", ["AR", "VAR"], index=0)
max_lags = st.sidebar.slider("Max lags (order selection)", 2, 18, 12)
steps = st.sidebar.slider("Forecast horizon (months)", 3, 24, 12)
recess_lag = st.sidebar.slider("Recession look-ahead window (months)", 3, 18, 12)
prob_thresh = st.sidebar.slider("Turn-point threshold", 0.1, 0.9, 0.5, 0.05)

with st.sidebar.expander("ℹ️ Notes"):
    st.write(
        """
        **Data:** FRED (DGS10, DGS2, USREC, UNRATE, CPIAUCSL, INDPRO).\
        **Models:** AR(p) auto-selected by AIC for SPREAD; VAR with macro covariates auto-lag by AIC.\
        **Recession Probability:** Logistic regression predicting recession within a forward window using SPREAD + macro.
        """
    )

# =============================
# --------- MAIN FLOW ---------
# =============================

# 1) Load data
with st.spinner("Downloading FRED data…"):
    df = load_fred(start.isoformat(), end.isoformat())

model_df = make_train(df, freq=freq)

# 2) Fit chosen model
col1, col2, col3 = st.columns([2,1,1])
with col1:
    st.markdown("### 📊 Data Preview")
    st.dataframe(model_df.tail(12))

with col2:
    st.markdown("### 🔧 Model Fit")
    if model_type == "AR":
        ar_model, p_sel, aic = fit_ar(model_df, max_lags=max_lags)
        st.metric("AR order (AIC)", p_sel)
        st.caption(f"AIC: {aic:.2f}")
    else:
        var_model, p_sel = fit_var(model_df, max_lags=min(max_lags, 12))
        st.metric("VAR order (AIC)", p_sel)

with col3:
    st.markdown("### 🧭 Horizon")
    st.metric("Forecast months", steps)
    st.metric("Recession window", recess_lag)

# 3) Forecast spread
hist_spread = model_df["SPREAD"].copy()

if model_type == "AR":
    mean, conf = forecast_ar(ar_model, steps=steps)
    fc_index = pd.date_range(hist_spread.index[-1] + pd.tseries.offsets.MonthEnd(0), periods=steps, freq="M" if freq=="M" else freq)
    fc_mean = pd.Series(mean.values, index=fc_index, name="Forecast")
    fc_lower = pd.Series(conf.iloc[:, 0].values, index=fc_index, name="Lower95")
    fc_upper = pd.Series(conf.iloc[:, 1].values, index=fc_index, name="Upper95")
else:
    mean, lower, upper = forecast_var(var_model, steps=steps)
    fc_index = pd.date_range(hist_spread.index[-1] + pd.tseries.offsets.MonthEnd(0), periods=steps, freq="M" if freq=="M" else freq)
    fc_mean = pd.Series(mean["SPREAD"].values, index=fc_index, name="Forecast")
    fc_lower = pd.Series(lower["SPREAD"].values, index=fc_index, name="Lower95")
    fc_upper = pd.Series(upper["SPREAD"].values, index=fc_index, name="Upper95")

# Chart 1: Spread + forecast
st.markdown("\n")
st.plotly_chart(make_spread_chart(hist_spread, fc_mean, fc_lower, fc_upper), use_container_width=True)

# 4) Recession probability model
logit_model, prob_df = fit_recession_logit(model_df, lag_months=recess_lag)

# Heatmap-style line (prob) + threshold crossings
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.subheader("Recession Probability & Turning Points")
st.plotly_chart(make_prob_chart(prob_df, thresh=prob_thresh), use_container_width=True)

ups, downs = turning_points(prob_df, thresh=prob_thresh)

c1, c2 = st.columns(2)
with c1:
    st.markdown("**Crossed ABOVE threshold** (risk rising)")
    if ups:
        st.write(pd.Series(ups, name="Date").dt.strftime("%Y-%m-%d").to_list())
    else:
        st.write("– None in-sample –")
with c2:
    st.markdown("**Crossed BELOW threshold** (risk easing)")
    if downs:
        st.write(pd.Series(downs, name="Date").dt.strftime("%Y-%m-%d").to_list())
    else:
        st.write("– None in-sample –")

st.markdown("</div>", unsafe_allow_html=True)

# 5) Model diagnostics
with st.expander("🔍 Model Diagnostics"):
    if model_type == "AR":
        st.write(ar_model.summary().as_text())
    else:
        st.write(var_model.summary())

with st.expander("🧪 Logistic (Recession) Model Summary"):
    st.text(logit_model.summary2().as_text())

# 6) Downloadables
def to_csv_download(df: pd.DataFrame):
    return df.to_csv(index=True).encode("utf-8")

c1, c2, c3 = st.columns(3)
with c1:
    st.download_button("⬇️ Download model dataset (CSV)", data=to_csv_download(model_df), file_name="yield_spread_model_dataset.csv", mime="text/csv")
with c2:
    fc_df = pd.DataFrame({"Forecast": fc_mean, "Lower95": fc_lower, "Upper95": fc_upper})
    st.download_button("⬇️ Download forecast (CSV)", data=to_csv_download(fc_df), file_name="spread_forecast.csv", mime="text/csv")
with c3:
    st.download_button("⬇️ Download recession probs (CSV)", data=to_csv_download(prob_df[["REC_PROB"]]), file_name="recession_probabilities.csv", mime="text/csv")

# 7) Footnote for recruiters
st.markdown("<div class='hr'></div>", unsafe_allow_html=True)
st.caption(
    "This demo highlights CFA L2 concepts: term structure (10Y–2Y), AR/VAR modeling, and a simple turning-point probability model using macro covariates."
)

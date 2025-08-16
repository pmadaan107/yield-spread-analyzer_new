# app.py
"""
Bond Scenario Pricer – Simple Website (Streamlit)
-------------------------------------------------
Enter a bond's coupon, maturity, payment frequency, and optional OAS (spread),
then analyze how changes in the **government yield curve** (flatten/steepen/parallel)
affect the bond **price today** and the **price at a chosen year before maturity**.

This app supports:
- Live gov't curves (Bank of Canada / US Treasury) or manual/example points
- Scenario engine: parallel shift, steepen, flatten (short vs long end)
- Pricing off the zero curve + OAS
- Duration/convexity
- Horizon (year t) price before maturity, with two modes:
  A) Carry/Roll-down (today's curve rolled to horizon)
  B) Scenario-at-horizon (apply your chosen curve shift at horizon)
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
import requests
from dataclasses import dataclass
from datetime import date, timedelta

st.set_page_config(page_title="Bond Scenario Pricer", layout="wide")
st.title("💹 Bond Scenario Pricer – Government Curve Impact")

# =============================
# Utilities
# =============================

def build_zero_curve(tenors: np.ndarray, rates: np.ndarray):
    order = np.argsort(tenors)
    x = np.asarray(tenors, float)[order]
    y = np.asarray(rates, float)[order]
    cs = CubicSpline(x, y, bc_type="natural")
    def z(t):
        t = np.asarray(t, float)
        return cs(np.clip(t, x.min(), x.max()))
    return z


def apply_two_point_shift(zf, short_bps: float, long_bps: float, pivot: float = 5.0):
    s = short_bps / 10000.0
    l = long_bps / 10000.0
    def z(t):
        t = np.asarray(t, float)
        return zf(t) + np.where(t <= pivot, s, l)
    return z


def year_fractions(n, freq):
    return np.arange(1, n + 1) / freq

@dataclass
class Bond:
    face: float
    coupon: float   # annual coupon rate (e.g., 0.05 = 5%)
    freq: int       # payments per year
    maturity: float # years
    oas_bps: float = 0.0

@dataclass
class Results:
    price: float
    macaulay: float
    mod_dur: float
    conv: float


def price_bond_zero(bond: Bond, zf) -> Results:
    n = int(round(bond.maturity * bond.freq))
    t = year_fractions(n, bond.freq)
    c = bond.coupon * bond.face / bond.freq
    cf = np.full(n, c)
    cf[-1] += bond.face
    oas = bond.oas_bps / 10000.0
    z = zf(t) + oas
    df = np.exp(-z * t)
    pv = cf * df
    price = float(pv.sum())
    w = pv / price
    macaulay = float((w * t).sum())
    # Effective mod duration & convexity via 1 bp parallel move
    eps = 1e-4
    def price_shift(dy):
        df_s = np.exp(-(z + dy) * t)
        return float((cf * df_s).sum())
    p_up = price_shift(eps)
    p_dn = price_shift(-eps)
    mod_dur = (p_dn - p_up) / (2 * eps * price)
    conv = (p_up + p_dn - 2 * price) / (price * eps ** 2)
    return Results(price, macaulay, mod_dur, conv)


def solve_flat_ytm(price: float, bond: Bond, guess: float = 0.04) -> float:
    n = int(round(bond.maturity * bond.freq))
    c = bond.coupon * bond.face / bond.freq
    y = guess
    for _ in range(50):
        r = y / bond.freq
        k = np.arange(1, n + 1)
        df = (1 + r) ** (-k)
        pv = c * df.sum() + bond.face * df[-1]
        ddf = -(k / bond.freq) * (1 + r) ** (-(k + 1))
        dp = c * ddf.sum() + bond.face * ddf[-1]
        diff = pv - price
        if abs(diff) < 1e-10:
            break
        y -= diff / (dp if dp != 0 else 1e-12)
        y = max(-0.99, y)
    return float(y)

# =============================
# Data sources (Gov't curves)
# =============================

@st.cache_data(ttl=3600)
def fetch_boc_latest():
    url = "https://www.bankofcanada.ca/valet/observations/group/bond_yields/json?recent=1"
    r = requests.get(url, timeout=10)
    r.raise_for_status()
    obs = r.json()["observations"][0]
    # Map: series -> tenor years
    mp = {"V39056": 2, "V39059": 3, "V39057": 5, "V39060": 7, "V39058": 10, "V39062": 30}
    ten, rt = [], []
    for code, yr in mp.items():
        if code in obs and "v" in obs[code]:
            ten.append(float(yr))
            rt.append(float(obs[code]["v"]) / 100.0)
    if not ten:
        raise ValueError("BoC data missing")
    o = np.argsort(ten)
    return np.array(ten)[o], np.array(rt)[o]

@st.cache_data(ttl=3600)
def fetch_treasury_latest():
    today = date.today(); start = today - timedelta(days=14)
    base = "https://api.fiscaldata.treasury.gov/services/api/fiscal_service/v1/accounting/od/daily_treasury_yield_curve"
    params = {"filter": f"record_date:gte:{start.isoformat()}", "sort": "-record_date", "page[number]": 1, "page[size]": 1}
    r = requests.get(base, params=params, timeout=10)
    r.raise_for_status()
    rec = r.json()["data"][0]
    fields = {"1_yr":1, "2_yr":2, "3_yr":3, "5_yr":5, "7_yr":7, "10_yr":10, "20_yr":20, "30_yr":30}
    ten, rt = [], []
    for f, yr in fields.items():
        key = f"yield_curve_{f}"
        v = rec.get(key)
        if v not in (None, ""):
            ten.append(float(yr)); rt.append(float(v)/100.0)
    if not ten:
        raise ValueError("Treasury data missing")
    o = np.argsort(ten)
    return np.array(ten)[o], np.array(rt)[o]

# =============================
# Sidebar controls
# =============================

with st.sidebar:
    st.header("⚙️ Inputs")
    market = st.selectbox("Market", ["Canada", "United States"], index=0)
    source = st.radio("Curve Data", ["Live (Govt API)", "Example", "Manual"], index=0)

    if source == "Manual":
        st.caption("Enter tenor (years) and zero rate (%). Keep tenors increasing.")
        manual = st.data_editor(pd.DataFrame({
            "Tenor (yrs)": [1,2,3,5,7,10,20,30],
            "Zero Rate (%)": [4.7,4.6,4.5,4.3,4.2,4.1,4.0,4.0]
        }), num_rows="dynamic", use_container_width=True)
        tenors = manual["Tenor (yrs)"].to_numpy(float)
        rates  = manual["Zero Rate (%)"].to_numpy(float) / 100.0
    else:
        EX_TEN = np.array([1,2,3,5,7,10,20,30])
        EX_CA  = np.array([0.047,0.046,0.045,0.043,0.042,0.041,0.040,0.040])
        EX_US  = np.array([0.048,0.047,0.046,0.044,0.043,0.042,0.041,0.041])
        tenors = EX_TEN
        rates  = EX_CA if market == "Canada" else EX_US
        if source == "Live (Govt API)":
            try:
                tenors, rates = fetch_boc_latest() if market=="Canada" else fetch_treasury_latest()
                st.success("Fetched latest government curve.")
            except Exception as e:
                st.warning(f"Live fetch failed: {e}. Using example points.")

    st.divider()
    st.subheader("Bond")
    face   = st.number_input("Face Value", 0.0, 1e9, 100.0, step=1.0)
    coup   = st.number_input("Coupon % (annual)", 0.0, 100.0, 5.0, step=0.1) / 100.0
    freq   = st.selectbox("Payments per Year (n)", [1,2,4], index=1)
    mat    = st.number_input("Maturity (years)", 0.25, 100.0, 10.0, step=0.25)
    oasbps = st.number_input("OAS / Spread (bps)", -1000.0, 2000.0, 0.0, step=5.0)

    st.divider()
    st.subheader("Scenario")
    scenario = st.selectbox("Type", ["None","Parallel","Steepen","Flatten"], index=2)
    mag = st.slider("Magnitude (bps)", -300, 300, 50, step=5)
    pivot = st.slider("Pivot (years)", 1, 15, 5)

    st.divider()
    st.subheader("Horizon Pricing")
    hor = st.number_input("Horizon year before maturity (t)", 0.0, 99.0, 3.0, step=0.5)

# =============================
# Build curves & scenario
# =============================

z_base = build_zero_curve(tenors, rates)

short_bps, long_bps = 0.0, 0.0
if scenario == "Parallel":
    short_bps = long_bps = mag
elif scenario == "Steepen":
    short_bps, long_bps = -mag, mag
elif scenario == "Flatten":
    short_bps, long_bps = mag, -mag
z_shift = apply_two_point_shift(z_base, short_bps, long_bps, pivot=float(pivot))

# =============================
# Plots – curves
# =============================

col_curve, col_pts = st.columns([2,1])
with col_curve:
    st.subheader("Government Zero Curves")
    grid = np.linspace(tenors.min(), tenors.max(), 200)
    fig, ax = plt.subplots(figsize=(7,4))
    ax.plot(grid, z_base(grid)*100, label="Base")
    ax.plot(grid, z_shift(grid)*100, linestyle="--", label="Scenario")
    ax.scatter(tenors, rates*100, marker="o", label="Points")
    ax.set_xlabel("Maturity (years)"); ax.set_ylabel("Zero Rate (%)"); ax.grid(True, alpha=0.3); ax.legend()
    st.pyplot(fig, use_container_width=True)

with col_pts:
    st.subheader("Key Tenors (%)")
    kt = np.array([1,2,3,5,7,10])
    dfk = pd.DataFrame({
        "Tenor": kt,
        "Base": np.round(z_base(kt)*100,3),
        "Scenario": np.round(z_shift(kt)*100,3),
        "Δ (bps)": np.round((z_shift(kt)-z_base(kt))*10000,1)
    })
    st.dataframe(dfk, hide_index=True, use_container_width=True)

# =============================
# Pricing today
# =============================

bond = Bond(face=face, coupon=coup, freq=int(freq), maturity=mat, oas_bps=oasbps)
res_base   = price_bond_zero(bond, z_base)
res_shift  = price_bond_zero(bond, z_shift)

st.subheader("Price Today (off Gov't Curve + OAS)")
ptab = pd.DataFrame({
    "Metric": ["Price (Base)", "Price (Scenario)", "Δ Price", "Δ %", "Macaulay (y)", "ModDur", "Convexity"],
    "Value": [
        round(res_base.price, 4),
        round(res_shift.price, 4),
        round(res_shift.price - res_base.price, 4),
        round((res_shift.price/res_base.price - 1)*100, 4),
        round(res_base.macaulay, 4),
        round(res_base.mod_dur, 6),
        round(res_base.conv, 6),
    ]
})
st.dataframe(ptab, hide_index=True, use_container_width=True)

# =============================
# Horizon price at year t < maturity
# =============================

st.subheader("Price at a Particular Year Before Maturity")
if hor <= 0:
    st.info("Set a positive horizon year t (e.g., 3.0)")
else:
    if hor >= mat:
        st.warning("Horizon must be strictly before maturity.")
    else:
        # Remaining maturity and cash flows after t
        rem_maturity = mat - hor
        # Roll-down (carry-only) uses the same zero curve but evaluated at remaining maturities
        # To compute **theoretical** horizon price P(t), discount remaining cash flows from t using the same curve
        # then divide by discount factor to t (i.e., forward price from today): P_fwd(0->t) = PV0(remaining)/DF0(t)
        n_total = int(round(mat * bond.freq))
        pay_times = year_fractions(n_total, bond.freq)  # times from 0
        # Select payments strictly after t
        after_mask = pay_times > hor + 1e-10
        t_after = pay_times[after_mask]
        # Cash flows
        c = bond.coupon * bond.face / bond.freq
        cf = np.full_like(t_after, c)
        if len(cf) > 0:
            cf[-1] += bond.face if abs(t_after[-1] - mat) < 1e-8 else 0.0
        # Present value at 0 of remaining CFs using base curve (carry-only)
        oas = bond.oas_bps / 10000.0
        z0 = z_base(t_after) + oas
        pv0_remaining = float(np.sum(cf * np.exp(-z0 * t_after)))
        df0_t = float(np.exp(-(z_base(hor) + oas) * hor))
        price_at_t_carry = pv0_remaining / df0_t

        # Scenario-at-horizon: apply scenario curve for discounting from t onward
        zS = z_shift(t_after) + oas
        pv0_remaining_s = float(np.sum(cf * np.exp(-zS * t_after)))
        df0_t_s = float(np.exp(-(z_shift(hor) + oas) * hor))
        price_at_t_scenario = pv0_remaining_s / df0_t_s

        htab = pd.DataFrame({
            "Metric": [
                "Horizon t (years)",
                "Remaining maturity (years)",
                "Price at t – Carry/Roll-down",
                "Price at t – Scenario-at-horizon",
                "Δ vs Carry"
            ],
            "Value": [
                hor,
                round(rem_maturity, 4),
                round(price_at_t_carry, 4),
                round(price_at_t_scenario, 4),
                round(price_at_t_scenario - price_at_t_carry, 4)
            ]
        })
        st.dataframe(htab, hide_index=True, use_container_width=True)
        st.caption("Carry/Roll-down uses today's base curve to value the future price implied today. Scenario-at-horizon applies your chosen curve shift at time t before discounting.")

# =============================
# Help
# =============================
with st.expander("ℹ️ Notes"):
    st.markdown("""
- **Steepen/Flatten**: Short end (≤ pivot) vs long end (> pivot) are moved by ±Magnitude (bps).
- **OAS**: Constant spread added to the zero curve for the bond; we hold it fixed across scenarios.
- **Horizon Price**: Forward price implied today. Carry = no extra shock; Scenario-at-horizon = apply your curve shift at t.
""")

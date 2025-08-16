# app.py
"""
Yield Curve Explorer & Bond Pricing (Deloitte-Ready)
---------------------------------------------------
An interactive Streamlit app to:
- Visualize zero-coupon yield curves
- Simulate steepening/flattening shifts
- Price bonds off the curve (plus optional credit spread)
- Report Macaulay duration, modified duration, and convexity

Author: Preet Madaan
"""

import math
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt

st.set_page_config(page_title="Yield Curve Explorer & Bond Pricer", layout="wide")
st.title("📈 Yield Curve Explorer & Bond Pricer")

# -------------------------------
# Utility & Finance Math
# -------------------------------

def as_bps(x: float) -> float:
    """Convert percent to basis points for display where needed."""
    return x * 10000.0


def year_fractions(n_periods: int, freq: int) -> np.ndarray:
    return np.arange(1, n_periods + 1) / freq


def build_zero_curve(tenors: np.ndarray, rates: np.ndarray):
    """Return a callable zero-rate function z(t) using cubic spline (monotone-ish via natural BC)."""
    # Guard: ensure strictly increasing tenors
    order = np.argsort(tenors)
    tenors = tenors[order]
    rates = rates[order]
    # Natural cubic spline in annualized terms
    cs = CubicSpline(tenors, rates, bc_type="natural")

    def z(t: float | np.ndarray) -> np.ndarray:
        t = np.clip(np.asarray(t, dtype=float), tenors.min(), tenors.max())
        return cs(t)

    return z


def apply_curve_shift(z_func, short_shift_bps: float, long_shift_bps: float, pivot: float = 5.0):
    """Return a shifted zero-curve callable. Short end (<= pivot) gets short_shift; long end (> pivot) gets long_shift."""
    short_shift = short_shift_bps / 10000.0
    long_shift = long_shift_bps / 10000.0

    def z_shifted(t):
        t_arr = np.asarray(t, dtype=float)
        shift = np.where(t_arr <= pivot, short_shift, long_shift)
        return z_func(t_arr) + shift

    return z_shifted


@dataclass
class BondSpec:
    face: float = 100.0
    coupon_rate: float = 0.03  # annual coupon rate (e.g., 0.03 = 3%)
    freq: int = 2  # payments per year
    maturity_years: float = 5.0
    oas_bps: float = 0.0  # spread over zero curve, in bps


@dataclass
class BondResults:
    price: float
    macaulay_duration: float
    modified_duration: float
    convexity: float


def price_bond_from_zero_curve(spec: BondSpec, z_func) -> BondResults:
    """Price fixed coupon bond from a zero curve + OAS (in bps). Returns price and risk measures."""
    # cash flows
    n = int(round(spec.maturity_years * spec.freq))
    times = year_fractions(n, spec.freq)
    cpn = spec.coupon_rate * spec.face / spec.freq
    cash_flows = np.full(n, cpn)
    cash_flows[-1] += spec.face

    # discount using continuously compounded zero rates with OAS
    oas = spec.oas_bps / 10000.0
    z_rates = z_func(times) + oas
    # PV = CF * exp(-z(t)*t)
    df = np.exp(-z_rates * times)
    pv_cashflows = cash_flows * df
    price = float(pv_cashflows.sum())

    # Macaulay duration (years)
    weights = pv_cashflows / price
    macaulay = float(np.sum(weights * times))

    # Modified duration using effective (small parallel shift)
    eps = 1e-4  # 1 bps in decimal (0.0001)

    def price_with_parallel(shift):
        df_shift = np.exp(-(z_rates + shift) * times)
        return float(np.sum(cash_flows * df_shift))

    p_up = price_with_parallel(eps)
    p_dn = price_with_parallel(-eps)
    dP = p_dn - p_up  # price rises when yields down
    mod_duration = (dP / (2 * eps)) / price

    # Convexity (effective)
    convexity = (p_up + p_dn - 2 * price) / (price * (eps ** 2))

    return BondResults(price=price, macaulay_duration=macaulay, modified_duration=mod_duration, convexity=convexity)


# -------------------------------
# Example Curves (Clean + Stable)
# -------------------------------

EXAMPLE_TENORS = np.array([0.5, 1, 2, 3, 5, 7, 10, 20, 30])
# Example Canada-ish zero rates (decimal). Adjust to taste; realistic but static.
EXAMPLE_CA_RATES = np.array([0.045, 0.046, 0.044, 0.043, 0.041, 0.040, 0.039, 0.038, 0.038])
# Example US-ish zero rates
EXAMPLE_US_RATES = np.array([0.047, 0.048, 0.046, 0.045, 0.043, 0.042, 0.041, 0.040, 0.040])

# -------------------------------
# Sidebar Controls
# -------------------------------

with st.sidebar:
    st.header("⚙️ Controls")
    market = st.selectbox("Market", ["Canada", "United States"], index=0)

    data_mode = st.radio(
        "Yield Curve Data",
        ["Example (built-in)", "Manual (enter points)"]
    )

    if data_mode == "Manual (enter points)":
        st.caption("Enter tenor (years) and zero rate (%) rows. Keep tenors increasing.")
        manual_df = st.data_editor(
            pd.DataFrame({
                "Tenor (yrs)": [0.5, 1, 2, 3, 5, 7, 10, 20, 30],
                "Zero Rate (%)": [4.5, 4.6, 4.4, 4.3, 4.1, 4.0, 3.9, 3.8, 3.8],
            }),
            num_rows="dynamic",
            use_container_width=True,
        )
        tenors = manual_df["Tenor (yrs)"].to_numpy(float)
        rates = manual_df["Zero Rate (%)"].to_numpy(float) / 100.0
    else:
        tenors = EXAMPLE_TENORS
        rates = EXAMPLE_CA_RATES if market == "Canada" else EXAMPLE_US_RATES

    st.divider()
    st.subheader("Scenario Shifts (bps)")
    short_shift_bps = st.slider("Short end shift (≤ 5y)", -300, 300, 0, 5)
    long_shift_bps = st.slider("Long end shift (> 5y)", -300, 300, 0, 5)

    st.divider()
    st.subheader("Bond Inputs")
    st.caption("Add bonds to price off the (shifted) curve. Coupon = annual %; OAS = spread in bps.")
    default_bonds = pd.DataFrame({
        "Face": [100, 100, 100],
        "Coupon %": [3.0, 0.0, 5.0],
        "Freq": [2, 2, 2],
        "Maturity (yrs)": [2.0, 5.0, 10.0],
        "OAS (bps)": [0.0, 25.0, 75.0],
    })
    bonds_df = st.data_editor(default_bonds, num_rows="dynamic", use_container_width=True)

# -------------------------------
# Build Curves
# -------------------------------

try:
    z_base = build_zero_curve(tenors, rates)
    z_shift = apply_curve_shift(z_base, short_shift_bps, long_shift_bps, pivot=5.0)
except Exception as e:
    st.error(f"Curve construction error: {e}")
    st.stop()


# -------------------------------
# Plots: Yield Curves
# -------------------------------

curve_grid = np.linspace(min(tenors), max(tenors), 200)
base_vals = z_base(curve_grid)
shift_vals = z_shift(curve_grid)

col_curve, col_table = st.columns([2, 1])
with col_curve:
    st.subheader("Yield Curve (Zero Rates)")
    fig1, ax1 = plt.subplots(figsize=(7, 4))
    ax1.plot(curve_grid, base_vals * 100, label="Base Curve")
    ax1.plot(curve_grid, shift_vals * 100, linestyle="--", label="Shifted Curve")
    ax1.scatter(tenors, rates * 100, label="Input Points", marker="o")
    ax1.set_xlabel("Maturity (years)")
    ax1.set_ylabel("Zero Rate (%)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    st.pyplot(fig1, use_container_width=True)

with col_table:
    st.subheader("Key Points (%)")
    key_tenors = np.array([0.5, 1, 2, 3, 5, 7, 10])
    key = pd.DataFrame({
        "Tenor": key_tenors,
        "Base": np.round(z_base(key_tenors) * 100, 3),
        "Shifted": np.round(z_shift(key_tenors) * 100, 3),
        "Δ (bps)": np.round(as_bps(z_shift(key_tenors) - z_base(key_tenors)), 1),
    })
    st.dataframe(key, hide_index=True, use_container_width=True)

# -------------------------------
# Bond Pricing & Risk
# -------------------------------

st.subheader("Bond Pricing & Risk (off Shifted Curve)")

results = []
for _, row in bonds_df.iterrows():
    try:
        spec = BondSpec(
            face=float(row["Face"]),
            coupon_rate=float(row["Coupon %"]) / 100.0,
            freq=int(row["Freq"]),
            maturity_years=float(row["Maturity (yrs)"]),
            oas_bps=float(row["OAS (bps)"]),
        )
        res = price_bond_from_zero_curve(spec, z_shift)
        results.append({
            "Face": spec.face,
            "Coupon %": round(spec.coupon_rate * 100, 4),
            "Freq": spec.freq,
            "Maturity (yrs)": spec.maturity_years,
            "OAS (bps)": spec.oas_bps,
            "Price": round(res.price, 4),
            "Macaulay Dur (y)": round(res.macaulay_duration, 4),
            "Mod Dur": round(res.modified_duration, 6),
            "Convexity": round(res.convexity, 6),
        })
    except Exception as e:
        results.append({"Error": str(e)})

if results:
    out_df = pd.DataFrame(results)
    st.dataframe(out_df, use_container_width=True)

    # Sensitivity chart for selected bond
    idx = st.selectbox("Select a bond for sensitivity chart (by row index)", list(range(len(out_df))))
    if isinstance(idx, int) and idx < len(out_df) and "Price" in out_df.columns:
        # Parallel shift sensitivity (-150 to +150 bps)
        spec_row = bonds_df.iloc[idx]
        spec = BondSpec(
            face=float(spec_row["Face"]),
            coupon_rate=float(spec_row["Coupon %"]) / 100.0,
            freq=int(spec_row["Freq"]),
            maturity_years=float(spec_row["Maturity (yrs)"]),
            oas_bps=float(spec_row["OAS (bps)"]),
        )
        shifts = np.arange(-150, 155, 5) / 10000.0
        prices = []
        for s in shifts:
            def z_tmp(t):
                return z_shift(t) + s
            prices.append(price_bond_from_zero_curve(spec, z_tmp).price)
        fig2, ax2 = plt.subplots(figsize=(7, 4))
        ax2.plot(shifts * 10000, prices)
        ax2.set_xlabel("Parallel Shift (bps)")
        ax2.set_ylabel("Price")
        ax2.grid(True, alpha=0.3)
        st.pyplot(fig2, use_container_width=True)

# -------------------------------
# Explanations (client-facing)
# -------------------------------
with st.expander("ℹ️ What do these metrics mean? (client-ready)"):
    st.markdown(
        """
- **Zero curve**: interest rates for default-free zero-coupon bonds at different maturities.
- **Steepening / flattening**: different moves at short vs long maturities; we model this with two sliders.
- **Price**: present value of cash flows discounted using the zero curve plus a bond-specific spread (OAS).
- **Macaulay duration**: weighted-average time to cash flows (years).
- **Modified duration**: % price change per 1% (100 bps) yield move (local slope).
- **Convexity**: curvature of the price–yield relationship; higher convexity cushions large moves.
        """
    )

st.caption("Tip: Export screenshots of the curve and tables for your Deloitte-style slide deck.")

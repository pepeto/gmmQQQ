# Streamlit app: QQQ full-history (Stooq CSV) → original features → GMM → dark Plotly SCATTER (log Y)
# Chart colors by continuous bull probability (0..1), taller figure, view starts at 2020.
# Comments in English; no try/except; no .iloc.

import numpy as np
import pandas as pd
import streamlit as st
from pandas_datareader.data import DataReader
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
import talib
import plotly.graph_objects as go

st.set_page_config(page_title="QQQ GMM Regimes", layout="wide")
st.title("QQQ Regime Inference with GMM (Stooq)")

# ---------- Sidebar: parameters ----------
st.sidebar.header("GMM Parameters")
n_components = 2
params = {
    "cov": "tied",
    "tol": 0.0027360784448994865,
    "n_init": 3,
    "init_params": "kmeans"
}
st.sidebar.write(params)

# ---------- 1) Fetch both sources (for info) but USE CSV for modeling because it's more recent ----------
@st.cache_data(show_spinner=False)
def fetch_stooq_both():
    symbol_reader = "QQQ.US"   # DataReader symbol
    symbol_csv    = "qqq.us"   # CSV endpoint symbol (lowercase at stooq)
    start_early   = pd.Timestamp("1900-01-01")

    # DataReader (for comparison/report only)
    dr_df = DataReader(symbol_reader, "stooq", start=start_early).sort_index()

    # CSV (used for modeling/plotting)
    csv_url = f"https://stooq.com/q/d/l/?s={symbol_csv}&i=d"
    csv_df = pd.read_csv(csv_url, parse_dates=["Date"]).set_index("Date").sort_index()

    def _prepare(df: pd.DataFrame) -> pd.DataFrame:
        cols = {c: c.title() for c in df.columns}
        df = df.rename(columns=cols)
        for c in ["Open", "High", "Low", "Close", "Volume"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        return df.dropna(subset=["Open", "High", "Low", "Close"])

    return _prepare(dr_df), _prepare(csv_df)

dr_df, csv_df = fetch_stooq_both()

# Choose CSV explicitly (more recent)
price_df = csv_df.copy()
chosen = "CSV (stooq.com)"

# ---------- 2) Coverage checks (RAW) ----------
st.subheader("Coverage Check – RAW")
col1, col2, col3 = st.columns(3)
with col1:
    st.write(f"**DataReader range**: {dr_df.index.min().date()} → {dr_df.index.max().date()} (rows={dr_df.shape[0]})")
with col2:
    st.write(f"**CSV range**: {csv_df.index.min().date()} → {csv_df.index.max().date()} (rows={csv_df.shape[0]})")
with col3:
    st.write(f"**Chosen source**: {chosen}")

first_raw_date = price_df.index.min()
last_raw_date  = price_df.index.max()
st.write(f"**Used raw range**: {first_raw_date.date()} → {last_raw_date.date()} (rows={price_df.shape[0]})")

# Enforce inception not later than 1999 (QQQ launched 1999-03-10)
assert first_raw_date.year <= 1999, f"Unexpected start date; got {first_raw_date.date()}"

# ---------- 3) Original feature engineering (matches your GMM.py) ----------
# Features: log_returns, ROC(5), WILLR(38), n_wma = Close/WMA(51) - 1
log_returns = np.log(price_df["Close"]).diff()

high_np  = np.ascontiguousarray(price_df["High"].to_numpy(dtype=np.float64))
low_np   = np.ascontiguousarray(price_df["Low"].to_numpy(dtype=np.float64))
close_np = np.ascontiguousarray(price_df["Close"].to_numpy(dtype=np.float64))

roc   = pd.Series(talib.ROC(close_np, timeperiod=5), index=price_df.index)
wma   = pd.Series(talib.WMA(close_np, timeperiod=51), index=price_df.index)
willr = pd.Series(talib.WILLR(high_np, low_np, close_np, timeperiod=38), index=price_df.index)

n_wma = (price_df["Close"] / wma) - 1

feature_df = pd.DataFrame({
    "log_returns": log_returns,
    "roc": roc,
    "willr": willr,
    "n_wma": n_wma
}).dropna()

aligned_index = pd.DatetimeIndex(feature_df.index)
close_aligned = price_df.loc[aligned_index, "Close"]

# ---------- 4) Scale features and fit GMM over entire feature history ----------
scaler = StandardScaler()
features_scaled = scaler.fit_transform(feature_df.to_numpy())

gmm = GaussianMixture(
    n_components=n_components,
    covariance_type=params['cov'],
    tol=params['tol'],
    n_init=params['n_init'],
    init_params=params['init_params'],
    max_iter=500,
    random_state=42
)
gmm.fit(features_scaled)

# Identify bull by higher mean on log_returns
logret_idx = 0
bull_label = int(np.argmax(gmm.means_[:, logret_idx]))

# Last-point diagnostics
proba_last = gmm.predict_proba(features_scaled[-1:].copy())[0]
last_ts = aligned_index[-1]
last_close = float(close_aligned.loc[last_ts])
last_component = int(np.argmax(proba_last))
bull_probability = float(proba_last[bull_label])
is_bullish = (last_component == bull_label)

st.subheader("Coverage Check – FEATURES")
st.write(f"**Feature range**: {aligned_index[0].date()} → {aligned_index[-1].date()} (rows={aligned_index.shape[0]})")

st.subheader("Last Point")
st.write(f"**{last_ts.date()}** | Close={last_close:.2f} | bull_label={bull_label} | "
         f"bull_prob={bull_probability:.4f} | bullish={is_bullish}")

# ---------- 5) Plotly SCATTER in dark mode with log Y – colored by bull probability ----------
# Visual filter: chart starts at 2020-01-01 (training remains full-history)
plot_start = pd.Timestamp("2020-01-01")
mask_plot = aligned_index >= plot_start
index_plot = aligned_index[mask_plot]
close_plot = close_aligned.loc[index_plot]

# Continuous probabilities over the whole history, then filter for plotting
proba_all = gmm.predict_proba(features_scaled)[:, bull_label]
proba_plot = proba_all[mask_plot]

fig_prob = go.Figure()
fig_prob.add_trace(go.Scatter(
    x=index_plot,
    y=close_plot,
    mode="markers",
    name="Close (prob-colored)",
    marker=dict(
        size=5,
        opacity=0.9,
        color=proba_plot,              # continuous values in [0, 1]
        cmin=0.0, cmax=1.0,
        colorscale=[ [0.0, "red"], [0.5, "yellow"], [1.0, "green"] ],
        colorbar=dict(title="Bull probability")
    )
))

# Highlight last point
fig_prob.add_trace(go.Scatter(
    x=[last_ts],
    y=[last_close],
    mode="markers",
    name="Last point",
    marker=dict(size=12, color="rgba(0,0,0,0)", line=dict(width=2, color="white"), symbol="circle-open-dot")
))

fig_prob.update_layout(
    title=f"QQQ – Bull probability coloring (scatter, log-Y) – last: {pd.Timestamp(last_ts).date()} | bull_prob={bull_probability:.3f}",
    xaxis_title="Date",
    yaxis_title="Close (log scale)",
    legend_title="Legend",
    hovermode="x unified",
    template="plotly_dark",
    height=900   # taller figure
)
fig_prob.update_yaxes(type="log")

st.plotly_chart(fig_prob, use_container_width=True)

# Optional tail preview of plotted data
st.subheader("Tail of Price Data (Plotted Slice since 2020)")
st.dataframe(close_plot.tail(10).to_frame("Close"))

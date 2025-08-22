# Streamlit app: QQQ full-history (Stooq) → original features → GMM → dark Plotly SCATTER (log Y)
# Graph starts at 2020 and is taller; model still trains on full history.
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
st.title("QQQ Regime Inference with GMM (Stooq, Original Features)")

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

# ---------- 1) Fetch from two sources and choose earliest coverage ----------
@st.cache_data(show_spinner=False)
def fetch_stooq_full():
    symbol_reader = "QQQ.US"
    symbol_csv    = "qqq.us"
    start_early   = pd.Timestamp("1900-01-01")

    dr_df = DataReader(symbol_reader, "stooq", start=start_early).sort_index()
    csv_url = f"https://stooq.com/q/d/l/?s={symbol_csv}&i=d"
    csv_df = pd.read_csv(csv_url, parse_dates=["Date"]).set_index("Date").sort_index()

    def _prepare(df: pd.DataFrame) -> pd.DataFrame:
        cols = {c: c.title() for c in df.columns}
        df = df.rename(columns=cols)
        for c in ["Open", "High", "Low", "Close", "Volume"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        return df.dropna(subset=["Open", "High", "Low", "Close"])

    dr_df  = _prepare(dr_df)
    csv_df = _prepare(csv_df)

    min_dr, min_csv = dr_df.index.min(), csv_df.index.min()
    if min_csv < min_dr:
        price_df = csv_df.copy()
        chosen = "CSV (stooq.com)"
    else:
        price_df = dr_df.copy()
        chosen = "pandas_datareader('stooq')"

    return price_df, dr_df, csv_df, chosen

price_df, dr_df, csv_df, chosen = fetch_stooq_full()

# ---------- 2) Coverage checks (RAW) ----------
first_raw_date = price_df.index.min()
last_raw_date  = price_df.index.max()

st.subheader("Coverage Check – RAW")
col1, col2, col3 = st.columns(3)
with col1:
    st.write(f"**DataReader range**: {dr_df.index.min().date()} → {dr_df.index.max().date()} (rows={dr_df.shape[0]})")
with col2:
    st.write(f"**CSV range**: {csv_df.index.min().date()} → {csv_df.index.max().date()} (rows={csv_df.shape[0]})")
with col3:
    st.write(f"**Chosen source**: {chosen}")
st.write(f"**Used raw range**: {first_raw_date.date()} → {last_raw_date.date()} (rows={price_df.shape[0]})")

# Enforce inception year 1999 (dataset level)
assert first_raw_date.year == 1999, f"Expected first year 1999; got {first_raw_date.date()}"

# ---------- 3) Original feature engineering (matches your GMM.py) ----------
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

components_all = gmm.predict(features_scaled)

# Identify bull by higher mean on log_returns
logret_idx = 0
bull_label = int(np.argmax(gmm.means_[:, logret_idx]))
is_bull = (components_all == bull_label)
is_bear = ~is_bull

# Last-point diagnostics
proba_last = gmm.predict_proba(features_scaled[-1:].copy())[0]
last_ts = aligned_index[-1]
last_close = float(close_aligned.loc[last_ts])
last_component = int(components_all[-1])
bull_probability = float(proba_last[bull_label])
is_bullish = (last_component == bull_label)

st.subheader("Coverage Check – FEATURES")
st.write(f"**Feature range**: {aligned_index[0].date()} → {aligned_index[-1].date()} (rows={aligned_index.shape[0]})")
# st.write("**Warm-up bars**: ~51 (from WMA(51))")

st.subheader("Last Point")
st.write(f"**{last_ts.date()}** | Close={last_close:.2f} | component={last_component} | "
         f"bull_label={bull_label} | bull_prob={bull_probability:.4f} | bullish={is_bullish}")

# ---------- 5) Plotly SCATTER in dark mode with log Y ----------
# Visual filter: start chart at 2020-01-01 (training remains full-history)
plot_start = pd.Timestamp("2020-01-01")
mask_plot = aligned_index >= plot_start
index_plot = aligned_index[mask_plot]
close_plot = close_aligned.loc[index_plot]

is_bull_plot = is_bull[mask_plot]
is_bear_plot = ~is_bull_plot

y_bull_plot = close_plot.where(is_bull_plot, other=np.nan)
y_bear_plot = close_plot.where(is_bear_plot, other=np.nan)

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=index_plot, y=y_bull_plot,
    mode="markers",
    name="Uptrend (bull)",
    marker=dict(color="green", size=4, opacity=0.85)
))
fig.add_trace(go.Scatter(
    x=index_plot, y=y_bear_plot,
    mode="markers",
    name="Downtrend (bear)",
    marker=dict(color="red", size=4, opacity=0.85)
))

# Last point marker (will be inside the plot range if last date >= 2020)
fig.add_trace(go.Scatter(
    x=[last_ts], y=[last_close],
    mode="markers",
    name="Last point",
    marker=dict(size=10, color="green" if is_bullish else "red", symbol="circle-open")
))

fig.update_layout(
    title=f"QQQ regimes by GMM (scatter) – last: {pd.Timestamp(last_ts).date()} | bull_prob={bull_probability:.3f}",
    xaxis_title="Date",
    yaxis_title="Close (log scale)",
    legend_title="Regime",
    hovermode="x unified",
    template="plotly_dark",
    height=900   # <-- taller figure
)
fig.update_yaxes(type="log")

st.plotly_chart(fig, use_container_width=True)

# Optional tail preview of plotted data
st.subheader("Tail of Price Data (Plotted Slice since 2020)")
st.dataframe(close_plot.tail(10).to_frame("Close"))

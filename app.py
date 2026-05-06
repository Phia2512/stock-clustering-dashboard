

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Stock Clustering Dashboard", layout="wide")
st.title("Stock Clustering Dashboard")
st.sidebar.header("Settings")

# --- Sidebar controls ---
all_stocks = ["AAPL", "TSLA", "MSFT", "GOOGL", "AMZN", "NFLX", "META", "NVDA", "AMD", "INTC",
              "JPM", "BAC", "GS", "WMT", "TGT", "UBER", "LYFT", "BABA", "DIS", "SONY"]

stocks = st.sidebar.multiselect(
    "Pick stocks to cluster",
    all_stocks,
    default=["AAPL", "TSLA", "MSFT", "GOOGL", "AMZN", "NFLX", "META", "NVDA"]
)

if not stocks or len(stocks) < 2:
    st.warning("Please select at least 2 stocks from the sidebar!")
    st.stop()

n_clusters = st.sidebar.slider("Number of clusters", 2, 6, 3)
if len(stocks) < n_clusters:
    n_clusters = len(stocks)
    st.sidebar.warning(f"Clusters reduced to {n_clusters} to match stock count!")

# --- Download data ---
with st.spinner("Downloading stock data from Yahoo Finance..."):
    raw_data = yf.download(stocks, period="2y", auto_adjust=True)
    if isinstance(raw_data.columns, pd.MultiIndex):
        raw = raw_data["Close"]
    else:
        raw = raw_data[["Close"]]
        raw.columns = stocks
    raw = raw.dropna(how="all")
    if isinstance(raw, pd.Series):
        raw = raw.to_frame()
        raw.columns = stocks

st.success(f"Loaded data for {len(stocks)} stocks!")

returns = raw.pct_change().dropna()

features = pd.DataFrame({
    "Annual Return":  returns.mean() * 252,
    "Volatility":     returns.std() * np.sqrt(252),
    "Sharpe Ratio":   (returns.mean() / returns.std()) * np.sqrt(252)
})

scaler  = StandardScaler()
scaled  = scaler.fit_transform(features)
kmeans  = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
features["Cluster"] = kmeans.fit_predict(scaled).astype(str)
pca     = PCA(n_components=2)
coords  = pca.fit_transform(scaled)
features["PC1"]   = coords[:, 0]
features["PC2"]   = coords[:, 1]
features["Stock"] = features.index

# ================================
# SECTION 1 — Individual Stock View
# ================================
st.markdown("---")
st.subheader("Individual Stock Data")

selected_stock = st.selectbox("Select a stock to inspect", stocks)

stock_data = raw[[selected_stock]].copy()
stock_data.columns = ["Close Price"]
stock_data["Daily Return %"] = stock_data["Close Price"].pct_change() * 100
stock_data["7D Moving Avg"]  = stock_data["Close Price"].rolling(7).mean()
stock_data["30D Moving Avg"] = stock_data["Close Price"].rolling(30).mean()

col1, col2, col3 = st.columns(3)
col1.metric("Current Price",  f"${stock_data['Close Price'].iloc[-1]:.2f}")
col2.metric("Annual Return",  f"{stock_data['Daily Return %'].mean() * 252:.2f}%")
col3.metric("Volatility",     f"{stock_data['Daily Return %'].std():.2f}%")

fig_price = px.line(
    stock_data,
    y=["Close Price", "7D Moving Avg", "30D Moving Avg"],
    title=f"{selected_stock} — Price Chart (2 Years)",
    labels={"value": "Price (USD)", "index": "Date"}
)
st.plotly_chart(fig_price, use_container_width=True)

fig_ret = px.bar(
    stock_data,
    y="Daily Return %",
    title=f"{selected_stock} — Daily Returns %",
    color="Daily Return %",
    color_continuous_scale=["red", "gray", "green"]
)
st.plotly_chart(fig_ret, use_container_width=True)

with st.expander("Show raw data table"):
    st.dataframe(stock_data.tail(30))

# ================================
# SECTION 2 — Comparison Chart
# ================================
st.markdown("---")
st.subheader("Comparison Chart — All Selected Stocks")
st.caption("Shows how each stock grew or fell over 2 years, starting from the same point (100)")

normalized = (raw / raw.iloc[0]) * 100
fig_compare = px.line(
    normalized,
    title="Normalized Price Comparison (Base = 100)",
    labels={"value": "Growth (Base 100)", "index": "Date", "variable": "Stock"}
)
fig_compare.update_layout(hovermode="x unified")
st.plotly_chart(fig_compare, use_container_width=True)

# ================================
# SECTION 3 — Risk Meter
# ================================
st.markdown("---")
st.subheader("Risk Meter — How risky is each stock?")
st.caption("Based on annual volatility. Higher volatility = higher risk.")

risk_df = features[["Stock", "Volatility"]].copy()
risk_df = risk_df.sort_values("Volatility", ascending=True)

def risk_label(v):
    if v < 0.25:
        return "Low Risk"
    elif v < 0.45:
        return "Medium Risk"
    else:
        return "High Risk"

def risk_color(v):
    if v < 0.25:
        return "green"
    elif v < 0.45:
        return "orange"
    else:
        return "red"

risk_df["Risk Level"] = risk_df["Volatility"].apply(risk_label)
risk_df["Color"]      = risk_df["Volatility"].apply(risk_color)

fig_risk = px.bar(
    risk_df,
    x="Stock",
    y="Volatility",
    color="Risk Level",
    color_discrete_map={
        "Low Risk":    "green",
        "Medium Risk": "orange",
        "High Risk":   "red"
    },
    title="Risk Meter by Stock",
    labels={"Volatility": "Annual Volatility", "Stock": "Stock"}
)
fig_risk.add_hline(y=0.25, line_dash="dash", line_color="green",  annotation_text="Low / Medium boundary")
fig_risk.add_hline(y=0.45, line_dash="dash", line_color="red",    annotation_text="Medium / High boundary")
st.plotly_chart(fig_risk, use_container_width=True)

risk_col1, risk_col2, risk_col3 = st.columns(3)
low    = risk_df[risk_df["Risk Level"] == "Low Risk"]["Stock"].tolist()
medium = risk_df[risk_df["Risk Level"] == "Medium Risk"]["Stock"].tolist()
high   = risk_df[risk_df["Risk Level"] == "High Risk"]["Stock"].tolist()

risk_col1.success(f"Low Risk: {', '.join(low) if low else 'None'}")
risk_col2.warning(f"Medium Risk: {', '.join(medium) if medium else 'None'}")
risk_col3.error(f"High Risk: {', '.join(high) if high else 'None'}")

# ================================
# SECTION 4 — Buy / Sell Signal
# ================================
st.markdown("---")
st.subheader("Buy / Sell Signal — Based on Sharpe Ratio")
st.caption("Sharpe Ratio measures return vs risk. Above 1.0 = Buy, Below 0.5 = Sell, In between = Hold")

signal_df = features[["Stock", "Sharpe Ratio", "Annual Return", "Volatility"]].copy()

def signal(s):
    if s >= 1.0:
        return "BUY"
    elif s >= 0.5:
        return "HOLD"
    else:
        return "SELL"

def signal_color(s):
    if s >= 1.0:
        return "green"
    elif s >= 0.5:
        return "orange"
    else:
        return "red"

signal_df["Signal"] = signal_df["Sharpe Ratio"].apply(signal)
signal_df["Color"]  = signal_df["Sharpe Ratio"].apply(signal_color)
signal_df = signal_df.sort_values("Sharpe Ratio", ascending=False)

fig_signal = px.bar(
    signal_df,
    x="Stock",
    y="Sharpe Ratio",
    color="Signal",
    color_discrete_map={
        "BUY":  "green",
        "HOLD": "orange",
        "SELL": "red"
    },
    title="Buy / Hold / Sell Signal per Stock",
    labels={"Sharpe Ratio": "Sharpe Ratio Score"}
)
fig_signal.add_hline(y=1.0, line_dash="dash", line_color="green", annotation_text="BUY threshold")
fig_signal.add_hline(y=0.5, line_dash="dash", line_color="red",   annotation_text="SELL threshold")
st.plotly_chart(fig_signal, use_container_width=True)

st.subheader("Signal Summary")
for _, row in signal_df.iterrows():
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.write(f"**{row['Stock']}**")
    c2.write(f"Sharpe: `{row['Sharpe Ratio']:.2f}`")
    c3.write(f"Return: `{row['Annual Return']:.2%}`")
    c4.write(f"Risk: `{row['Volatility']:.2%}`")
    if row["Signal"] == "BUY":
        c5.success("BUY")
    elif row["Signal"] == "HOLD":
        c5.warning("HOLD")
    else:
        c5.error("SELL")

# ================================
# SECTION 5 — Clustering
# ================================
st.markdown("---")
st.subheader("Clustering All Selected Stocks")

fig_cluster = px.scatter(
    features, x="PC1", y="PC2",
    color="Cluster", text="Stock",
    hover_data=["Annual Return", "Volatility", "Sharpe Ratio"],
    title="KMeans Stock Clustering (2D View)"
)
fig_cluster.update_traces(textposition="top center", marker=dict(size=12))
st.plotly_chart(fig_cluster, use_container_width=True)

st.subheader("Cluster Summary")
summary = features.groupby("Cluster")[["Annual Return", "Volatility", "Sharpe Ratio"]].mean().round(3)
st.dataframe(summary)

st.markdown("---")
st.subheader(f"Where does {selected_stock} sit?")
row = features.loc[selected_stock]
c1, c2, c3, c4 = st.columns(4)
c1.metric("Cluster",        f"Group {row['Cluster']}")
c2.metric("Annual Return",  f"{row['Annual Return']:.2%}")
c3.metric("Volatility",     f"{row['Volatility']:.2%}")
c4.metric("Sharpe Ratio",   f"{row['Sharpe Ratio']:.2f}")
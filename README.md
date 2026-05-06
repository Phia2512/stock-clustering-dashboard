# Stock Market Clustering Dashboard

An AI-powered stock market analysis dashboard built with Python and Streamlit.
Groups stocks by financial behavior using KMeans clustering and provides
risk assessment and buy/sell signals.

## Features
- Live stock data download using yfinance
- KMeans clustering to group similar stocks
- Interactive 2D cluster visualization using PCA
- Stock comparison chart
- Risk meter for each stock
- Buy / Hold / Sell signals based on Sharpe Ratio
- Individual stock price charts with moving averages

## Technologies Used
- Python
- Streamlit
- Scikit-learn (KMeans, PCA, StandardScaler)
- yfinance
- Pandas
- NumPy
- Plotly

## How to Run

1. Clone this repository
2. Install dependencies
3. Run the dashboard

## Install dependencies
pip install -r requirements.txt

## Run the app
streamlit run app.py

## Dashboard Sections
- Individual Stock Data — price chart, returns, moving averages
- Comparison Chart — all stocks normalized to same base
- Risk Meter — low, medium, high risk color coded
- Buy/Hold/Sell Signal — based on Sharpe Ratio score
- Clustering View — 2D scatter plot of stock groups

## Project By
Your Name
Coimbatore, Tamil Nadu
2025

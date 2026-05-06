# Stock Clustering Dashboard

I built this project to learn how machine learning works with real world data.
The idea was simple — can we group stocks that behave similarly using an algorithm?
Turns out yes, and the results are actually pretty useful!

## What this project does

I used Python to pull real stock market data directly from Yahoo Finance
and then applied a clustering algorithm called KMeans to group stocks
based on how they actually behave — not just their price.

The dashboard shows you which stocks are risky, which ones are stable,
and whether a stock is worth buying or not based on its performance score.

## What I learned building this

- How to get real financial data using yfinance
- What volatility and Sharpe ratio actually mean
- How KMeans clustering groups data without being told the answer
- How PCA reduces data so we can visualize it in 2D
- How to build a live interactive dashboard using Streamlit

## How to run it yourself

Download the files and open a terminal in the folder

Install the libraries
pip install -r requirements.txt

Run the dashboard
streamlit run app.py

Then open your browser and go to localhost:8501

## Tech I used

Python, Streamlit, Scikit-learn, yfinance, Pandas, Plotly

## Sections in the dashboard

Stock price chart with moving averages for any stock you pick,
comparison chart showing all stocks growing from the same starting point,
a risk meter that tells you how volatile each stock is,
buy hold or sell signals based on the Sharpe ratio score,
and a 2D cluster plot showing which stocks behave similarly.

## Built by
Jophilla 

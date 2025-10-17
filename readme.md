# 📈 Stock Market Forecasting Web App

A **Streamlit-based web application** that forecasts stock prices for selected companies using **ARIMA/SARIMA** models.  
Developed in **Python**, the app fetches live stock data using **Yahoo Finance (yfinance)**, performs **EDA**, checks **stationarity**, applies **seasonal decomposition**, and provides **visual forecasts** with interactive graphs.

## 🧠 Key Features

- Fetches real-time stock data via **Yahoo Finance (yfinance)**
- Interactive visualizations with **Plotly**
- **ADF Stationarity Test** for model preparation
- **Seasonal Decomposition** of time series data
- Forecasting using **ARIMA / SARIMA models**
- Compare **Actual vs Predicted** values dynamically
- Simple and intuitive **Streamlit UI**

## ⚙️ Installation & Setup (Local Machine)
Follow these steps to run the app locally 👇
### 1. Clone the Repository
```
git clone https://github.com/ahmedalimalik99/Stock-Matket-Analysis.git
```

### 2. Create a Virtual Environment
```
python -m venv myvenv
```
### 3. Activate the Virtual Environment
Windows:
```
venv\Scripts\activate
```
Mac/Linux:
```
source venv/bin/activate
```
### 4. Install Dependencies
```
pip install -r requirements.txt
```
### 5. Run the Streamlit App
```
streamlit run stock_market_forecasting.py
```
Then open the URL shown in your terminal (usually http://localhost:8501).

📊 Usage Guide

1. Upload / Select your preferred stock symbol (e.g., AAPL, MSFT, TSLA).

2. Adjust Date Range using the sidebar filters.

3. View raw data and interactive visualizations.

4. Run stationarity tests and seasonal decomposition.

5. Choose model parameters (p, d, q, P, D, Q, seasonal period).

6. Generate forecasts and visualize actual vs predicted prices.
👨‍💻 Author

**Ahmed Ali Malik**

📍 Location : Pakistan

💼 Aspiring Data Scientist | Python Developer | Machine Learning Enthusiast
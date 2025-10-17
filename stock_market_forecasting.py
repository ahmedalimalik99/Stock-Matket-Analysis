import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import date
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller

# ---------------------------
# App header
# ---------------------------
app_name = 'Stock Market Forecasting'
st.title(app_name)
st.subheader('This app forecasts the stock price of a selected company')

st.image("https://media.istockphoto.com/id/1487894858/photo/candlestick-chart-and-data-of-financial-market.jpg?s=612x612&w=0&k=20&c=wZ6vVmbm4BV2JOePSnNNz-0aFVOJZ0P9nhdeOMGUg5I=")

# ---------------------------
# Sidebar inputs
# ---------------------------
st.sidebar.header('Select parameters')
start_date = st.sidebar.date_input('Start date', date(2020, 1, 1))
end_date = st.sidebar.date_input('End date', date(2020, 12, 31))

ticker_list = ["AAPL", "MSFT", "GOOG", "GOOGL", "META", "TSLA", "NVDA", "ADBE", "PYPL", "INTC", "CMCSA", "NFLX", "PEP"]
ticker = st.sidebar.selectbox('Select company', ticker_list)

# ---------------------------
# Fetch data
# ---------------------------
data = yf.download(ticker, start=start_date, end=end_date)

if data.empty:
    st.error("No data fetched for the selected ticker / date range. Try a different range or ticker.")
    st.stop()

# Flatten multi-index columns if present and create a proper 'Date' column
if isinstance(data.columns, pd.MultiIndex):
    new_cols = []
    for col in data.columns:
        # col is a tuple like ('Close','GOOGL') or ('Date','')
        if len(col) == 2:
            if col[1] is None or str(col[1]).strip() == "":
                new_cols.append(col[0])
            else:
                new_cols.append(f"{col[0]}_{col[1]}")
        else:
            new_cols.append("_".join([c for c in col if c]))
    data.columns = new_cols
else:
    # ensure columns are strings
    data.columns = [str(c) for c in data.columns]

# Reset index so Date becomes a column named 'Date'
data = data.reset_index()
# Ensure Date is datetime
data['Date'] = pd.to_datetime(data['Date'])

st.write('Data from', start_date, 'to', end_date)
st.write(data.tail(10))

# ---------------------------
# Visualization (all numeric columns)
# ---------------------------
st.header('Data Visualization')
st.subheader('All numeric series')

# Select numeric columns excluding Date
numeric_cols = [c for c in data.columns if c != 'Date' and pd.api.types.is_numeric_dtype(data[c])]

if not numeric_cols:
    st.error("No numeric columns found to plot.")
else:
    # melt for multi-series plotting
    try:
        melted = data.melt(id_vars='Date', value_vars=numeric_cols,
                           var_name='Price Type', value_name='Price')
        fig = px.line(melted, x='Date', y='Price', color='Price Type',
                      title=f'{ticker} - Numeric Series', width=1000, height=600)
        st.plotly_chart(fig)
    except Exception as e:
        st.error(f"Could not create multi-series plot: {e}")

# ---------------------------
# Select column for forecasting
# ---------------------------
st.subheader('Select column for forecasting (single series)')
column = st.selectbox('Select the column to be used for forecasting', numeric_cols)

# Keep only Date and chosen column for modeling
series_df = data[['Date', column]].copy()
series_df = series_df.dropna().reset_index(drop=True)  # remove NaNs

st.write("Selected series (last 10 rows):")
st.write(series_df.tail(10))

# ---------------------------
# Stationarity test (ADF)
# ---------------------------
st.header('Stationarity check (ADF test)')
adf_result = None
try:
    adf_stat, adf_pvalue, *_ = adfuller(series_df[column].values)
    adf_result = (adf_pvalue, adf_pvalue < 0.05)
    st.write(f"ADF p-value: {adf_pvalue:.6f}")
    st.write("=> Stationary:" , adf_result[1])
except Exception as e:
    st.warning(f"ADF test could not be performed: {e}")

# ---------------------------
# Seasonal decomposition (safe)
# ---------------------------
st.header('Seasonal Decomposition (additive)')
min_period = 12
if len(series_df) < 2 * min_period:
    st.info(f"Not enough data to run decomposition (need at least {2*min_period} points).")
else:
    try:
        decomposition = seasonal_decompose(series_df[column], model='additive', period=min_period, extrapolate_trend='freq')
        # Plotting with matplotlib object - show as image in Streamlit
        st.write("Decomposition plot (matplotlib)")
        fig_decomp = decomposition.plot()
        st.pyplot(fig_decomp)
    except Exception as e:
        st.warning(f"Decomposition failed: {e}")

# ---------------------------
# SARIMAX model inputs
# ---------------------------
st.header('ARIMA / SARIMA model parameters')

p = st.slider('p (AR order)', 0, 5, 2)
d = st.slider('d (difference order)', 0, 2, 1)
q = st.slider('q (MA order)', 0, 5, 2)
seasonal_P = st.slider('Seasonal P', 0, 2, 0)
seasonal_D = st.slider('Seasonal D', 0, 1, 0)
seasonal_Q = st.slider('Seasonal Q', 0, 2, 0)
seasonal_period = st.number_input('Seasonal period (e.g., 12 for monthly seasonality)', 0, 365, 12)

# Prepare the series for modeling
y = series_df[column].astype(float).dropna()

if len(y) < 10:
    st.error("Not enough data points to fit the model (need at least 10). Adjust the date range or choose another ticker.")
    st.stop()

# Fit the model (wrapped in try/except — model fitting can fail)
model = None
try:
    sarimax_order = (p, d, q)
    seasonal_order = (seasonal_P, seasonal_D, seasonal_Q, int(seasonal_period) if seasonal_period > 0 else 0)
    # if seasonal_period is 0, pass seasonal_order=(0,0,0,0)
    model_inst = sm.tsa.statespace.SARIMAX(y, order=sarimax_order,
                                           seasonal_order=seasonal_order if seasonal_period > 0 else (0,0,0,0),
                                           enforce_stationarity=False, enforce_invertibility=False)
    model = model_inst.fit(disp=False)
    st.subheader("Model summary")
    st.write(model.summary())
except Exception as e:
    st.error(f"Model fitting failed: {e}")
    st.stop()

# ---------------------------
# Forecasting
# ---------------------------
st.header('Forecasting')
forecast_period = st.number_input('Days to forecast (horizon)', 1, 365, 10)

try:
    forecast_obj = model.get_forecast(steps=int(forecast_period))
    preds = forecast_obj.predicted_mean
    # Build DataFrame for predictions with proper dates starting the day after last actual date
    start_for_preds = series_df['Date'].iloc[-1] + pd.Timedelta(days=1)
    pred_index = pd.date_range(start=start_for_preds, periods=len(preds), freq='D')
    preds_df = pd.DataFrame({'Date': pred_index, 'predicted_mean': preds.values})
    st.write("Predictions")
    st.write(preds_df)
except Exception as e:
    st.error(f"Forecasting failed: {e}")
    st.stop()

# ---------------------------
# Plot Actual vs Predicted
# ---------------------------
st.header('Actual vs Predicted')

fig = go.Figure()
fig.add_trace(go.Scatter(x=series_df['Date'], y=series_df[column], mode='lines', name='Actual'))
fig.add_trace(go.Scatter(x=preds_df['Date'], y=preds_df['predicted_mean'], mode='lines', name='Predicted'))
fig.update_layout(title=f'{ticker} - Actual vs Predicted ({column})', xaxis_title='Date', yaxis_title='Price', width=1100, height=450)
st.plotly_chart(fig)

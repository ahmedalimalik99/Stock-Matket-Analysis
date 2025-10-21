import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import date, datetime, timedelta
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
import warnings
warnings.filterwarnings('ignore')

# ---------------------------
# Page Configuration & Styling
# ---------------------------
st.set_page_config(
    page_title="AI Stock Forecasting Pro",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 5px solid #667eea;
        margin: 0.5rem 0;
    }
    .prediction-card {
        background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2c3e50;
        margin: 1.5rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #667eea;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------
# App header
# ---------------------------
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown('<div class="main-header">📈 AI Stock Forecasting Pro</div>', unsafe_allow_html=True)
    st.markdown("### Advanced Market Intelligence & Predictive Analytics")

# ---------------------------
# Sidebar with enhanced inputs
# ---------------------------
with st.sidebar:
    st.markdown("### 🎯 Configuration Panel")
    
    # Company selection with logos
    st.markdown("#### Company Selection")
    ticker_info = {
        "AAPL": {"name": "Apple Inc.", "sector": "Technology"},
        "MSFT": {"name": "Microsoft Corp", "sector": "Technology"}, 
        "GOOG": {"name": "Alphabet Inc.", "sector": "Technology"},
        "META": {"name": "Meta Platforms", "sector": "Technology"},
        "TSLA": {"name": "Tesla Inc.", "sector": "Automotive"},
        "NVDA": {"name": "NVIDIA Corp", "sector": "Technology"},
        "AMZN": {"name": "Amazon.com Inc.", "sector": "Consumer"},
        "JPM": {"name": "JPMorgan Chase", "sector": "Financial"},
        "JNJ": {"name": "Johnson & Johnson", "sector": "Healthcare"},
        "XOM": {"name": "Exxon Mobil", "sector": "Energy"}
    }
    
    selected_ticker = st.selectbox(
        "Select Company",
        options=list(ticker_info.keys()),
        format_func=lambda x: f"{x} - {ticker_info[x]['name']}"
    )
    
    # Date range with presets
    st.markdown("#### Time Period")
    date_preset = st.selectbox("Quick Select", [
        "Custom Range", 
        "Last 3 Months", 
        "Last 6 Months", 
        "Last Year", 
        "Last 2 Years",
        "Last 5 Years"
    ])
    
    today = date.today()
    if date_preset == "Last 3 Months":
        start_date = today - timedelta(days=90)
        end_date = today
    elif date_preset == "Last 6 Months":
        start_date = today - timedelta(days=180)
        end_date = today
    elif date_preset == "Last Year":
        start_date = today - timedelta(days=365)
        end_date = today
    elif date_preset == "Last 2 Years":
        start_date = today - timedelta(days=730)
        end_date = today
    elif date_preset == "Last 5 Years":
        start_date = today - timedelta(days=1825)
        end_date = today
    else:
        start_date = st.date_input('Start Date', date(2020, 1, 1))
        end_date = st.date_input('End Date', today)
    
    # Advanced settings
    with st.expander("🔧 Advanced Settings"):
        st.markdown("#### Model Parameters")
        col1, col2 = st.columns(2)
        with col1:
            p = st.slider('AR Order (p)', 0, 5, 1)
            d = st.slider('Difference Order (d)', 0, 2, 1)
            q = st.slider('MA Order (q)', 0, 5, 1)
        with col2:
            seasonal_P = st.slider('Seasonal P', 0, 2, 0)
            seasonal_D = st.slider('Seasonal D', 0, 1, 0)
            seasonal_Q = st.slider('Seasonal Q', 0, 2, 0)
            seasonal_period = st.number_input('Seasonal Period', 0, 365, 12)
        
        forecast_days = st.slider('Forecast Horizon (Days)', 7, 365, 30)

# ---------------------------
# Fetch and prepare data
# ---------------------------
@st.cache_data(ttl=3600)  # Cache for 1 hour
def fetch_stock_data(ticker, start, end):
    """Fetch stock data with error handling"""
    try:
        data = yf.download(ticker, start=start, end=end, progress=False)
        if data.empty:
            return None, "No data available for the selected period"
        
        # Clean column names
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [col[0] for col in data.columns]
        
        data = data.reset_index()
        data['Date'] = pd.to_datetime(data['Date'])
        
        # Calculate additional metrics
        data['Daily_Return'] = data['Close'].pct_change() * 100
        data['Volatility'] = data['Daily_Return'].rolling(window=20).std()
        data['SMA_20'] = data['Close'].rolling(window=20).mean()
        data['SMA_50'] = data['Close'].rolling(window=50).mean()
        
        return data, None
    except Exception as e:
        return None, f"Error fetching data: {str(e)}"

# Fetch data
data, error = fetch_stock_data(selected_ticker, start_date, end_date)

if error:
    st.error(f"❌ {error}")
    st.stop()

# ---------------------------
# Key Metrics Dashboard
# ---------------------------
st.markdown("## 📊 Market Intelligence Dashboard")

# Calculate key metrics
latest = data.iloc[-1]
previous = data.iloc[-2] if len(data) > 1 else latest

price_change = latest['Close'] - previous['Close']
price_change_pct = (price_change / previous['Close']) * 100
volume_change = latest['Volume'] - previous['Volume']

# Display metrics in columns
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
    <div class="metric-card">
        <h3>Current Price</h3>
        <h2>${latest['Close']:.2f}</h2>
        <p style="color: {'#00cc00' if price_change_pct >= 0 else '#ff4444'}">
            {price_change_pct:+.2f}% (${price_change:+.2f})
        </p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card">
        <h3>Volume</h3>
        <h2>{latest['Volume']:,.0f}</h2>
        <p style="color: {'#00cc00' if volume_change >= 0 else '#ff4444'}">
            {volume_change:+,.0f}
        </p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-card">
        <h3>Volatility</h3>
        <h2>{latest['Volatility']:.2f}%</h2>
        <p>20-day rolling std</p>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="metric-card">
        <h3>Data Points</h3>
        <h2>{len(data):,}</h2>
        <p>Trading days</p>
    </div>
    """, unsafe_allow_html=True)

# ---------------------------
# Interactive Price Chart with Technical Indicators
# ---------------------------
st.markdown("## 📈 Advanced Technical Analysis")

# Chart type selector
chart_type = st.radio("Chart Type:", ["Line", "Candlestick", "Area"], horizontal=True)

fig = make_subplots(
    rows=2, cols=1,
    shared_xaxes=True,
    vertical_spacing=0.1,
    subplot_titles=('Price Movement', 'Volume'),
    row_heights=[0.7, 0.3]
)

if chart_type == "Candlestick":
    fig.add_trace(go.Candlestick(
        x=data['Date'],
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close'],
        name="OHLC"
    ), row=1, col=1)
else:
    trace_type = go.Scatter if chart_type == "Line" else go.Scatter
    fig.add_trace(trace_type(
        x=data['Date'], y=data['Close'],
        mode='lines',
        name='Close Price',
        line=dict(width=2, color='#1f77b4')
    ), row=1, col=1)

# Add moving averages
fig.add_trace(go.Scatter(
    x=data['Date'], y=data['SMA_20'],
    mode='lines',
    name='SMA 20',
    line=dict(width=1, color='orange')
), row=1, col=1)

fig.add_trace(go.Scatter(
    x=data['Date'], y=data['SMA_50'],
    mode='lines',
    name='SMA 50',
    line=dict(width=1, color='red')
), row=1, col=1)

# Volume bars
colors = ['red' if row['Close'] < data.iloc[i-1]['Close'] else 'green' 
          for i, row in data.iterrows()]
colors[0] = 'green'  # First day

fig.add_trace(go.Bar(
    x=data['Date'],
    y=data['Volume'],
    name='Volume',
    marker_color=colors,
    opacity=0.6
), row=2, col=1)

fig.update_layout(
    height=600,
    showlegend=True,
    xaxis_rangeslider_visible=False,
    template="plotly_white"
)

st.plotly_chart(fig, use_container_width=True)

# ---------------------------
# Statistical Analysis
# ---------------------------
st.markdown("## 🔍 Statistical Insights")

col1, col2 = st.columns(2)

with col1:
    # Returns distribution
    returns_fig = px.histogram(
        data, x='Daily_Return', 
        title='Daily Returns Distribution',
        nbins=50,
        color_discrete_sequence=['#ff6b6b']
    )
    returns_fig.update_layout(showlegend=False)
    st.plotly_chart(returns_fig, use_container_width=True)

with col2:
    # Correlation heatmap
    numeric_data = data[['Open', 'High', 'Low', 'Close', 'Volume']].corr()
    corr_fig = px.imshow(
        numeric_data,
        title='Price Correlations',
        aspect='auto',
        color_continuous_scale='RdBu_r'
    )
    st.plotly_chart(corr_fig, use_container_width=True)

# ---------------------------
# Time Series Analysis
# ---------------------------
st.markdown("## ⚙️ Time Series Analysis")

analysis_col1, analysis_col2 = st.columns(2)

with analysis_col1:
    # Stationarity test
    st.markdown("#### Stationarity Check (ADF Test)")
    try:
        adf_stat, adf_pvalue, _, _, adf_critical, _ = adfuller(data['Close'].dropna())
        st.metric("ADF p-value", f"{adf_pvalue:.6f}")
        st.metric("Stationary", "✅ Yes" if adf_pvalue < 0.05 else "❌ No")
        
        # Display critical values
        st.write("Critical Values:")
        for key, value in adf_critical.items():
            st.write(f"{key}: {value:.3f}")
    except Exception as e:
        st.error(f"ADF test failed: {e}")

with analysis_col2:
    # Seasonal decomposition
    st.markdown("#### Seasonal Decomposition")
    if len(data) >= 2 * 30:  # Minimum data points
        try:
            decomposition = seasonal_decompose(
                data.set_index('Date')['Close'], 
                model='additive', 
                period=30,
                extrapolate_trend='freq'
            )
            
            decomp_fig = make_subplots(
                rows=4, cols=1,
                subplot_titles=('Original', 'Trend', 'Seasonal', 'Residual')
            )
            
            decomp_fig.add_trace(go.Scatter(x=data['Date'], y=decomposition.observed, name='Observed'), row=1, col=1)
            decomp_fig.add_trace(go.Scatter(x=data['Date'], y=decomposition.trend, name='Trend'), row=2, col=1)
            decomp_fig.add_trace(go.Scatter(x=data['Date'], y=decomposition.seasonal, name='Seasonal'), row=3, col=1)
            decomp_fig.add_trace(go.Scatter(x=data['Date'], y=decomposition.resid, name='Residual'), row=4, col=1)
            
            decomp_fig.update_layout(height=600, showlegend=False)
            st.plotly_chart(decomp_fig, use_container_width=True)
        except Exception as e:
            st.warning(f"Decomposition limited: {e}")
    else:
        st.info("Need more data for seasonal decomposition")

# ---------------------------
# Machine Learning Forecasting
# ---------------------------
st.markdown("## 🤖 AI Forecasting Engine")

# Prepare data for modeling
model_data = data[['Date', 'Close']].copy().dropna()

if len(model_data) < 30:
    st.warning("Insufficient data for reliable forecasting. Need at least 30 data points.")
else:
    # Fit SARIMAX model
    with st.spinner('Training AI model... This may take a moment'):
        try:
            y = model_data['Close'].astype(float)
            
            sarimax_order = (p, d, q)
            seasonal_order = (seasonal_P, seasonal_D, seasonal_Q, seasonal_period)
            
            model = sm.tsa.statespace.SARIMAX(
                y, 
                order=sarimax_order,
                seasonal_order=seasonal_order,
                enforce_stationarity=False, 
                enforce_invertibility=False
            )
            fitted_model = model.fit(disp=False)
            
            # Generate forecasts
            forecast = fitted_model.get_forecast(steps=forecast_days)
            forecast_mean = forecast.predicted_mean
            confidence_int = forecast.conf_int()
            
            # Create forecast dataframe
            last_date = model_data['Date'].iloc[-1]
            forecast_dates = pd.date_range(
                start=last_date + timedelta(days=1), 
                periods=forecast_days, 
                freq='D'
            )
            
            forecast_df = pd.DataFrame({
                'Date': forecast_dates,
                'Predicted': forecast_mean,
                'Lower_CI': confidence_int.iloc[:, 0],
                'Upper_CI': confidence_int.iloc[:, 1]
            })
            
            # Display forecast results
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Forecast visualization
                forecast_fig = go.Figure()
                
                # Historical data
                forecast_fig.add_trace(go.Scatter(
                    x=model_data['Date'], 
                    y=model_data['Close'],
                    mode='lines',
                    name='Historical',
                    line=dict(color='#1f77b4', width=2)
                ))
                
                # Forecast
                forecast_fig.add_trace(go.Scatter(
                    x=forecast_df['Date'], 
                    y=forecast_df['Predicted'],
                    mode='lines',
                    name='Forecast',
                    line=dict(color='#ff7f0e', width=2, dash='dash')
                ))
                
                # Confidence interval
                forecast_fig.add_trace(go.Scatter(
                    x=forecast_df['Date'].tolist() + forecast_df['Date'].tolist()[::-1],
                    y=forecast_df['Upper_CI'].tolist() + forecast_df['Lower_CI'].tolist()[::-1],
                    fill='toself',
                    fillcolor='rgba(255, 127, 14, 0.2)',
                    line=dict(color='rgba(255,255,255,0)'),
                    name='Confidence Interval'
                ))
                
                forecast_fig.update_layout(
                    title=f'{selected_ticker} Price Forecast ({forecast_days} days)',
                    xaxis_title='Date',
                    yaxis_title='Price ($)',
                    height=500,
                    template="plotly_white"
                )
                
                st.plotly_chart(forecast_fig, use_container_width=True)
            
            with col2:
                st.markdown("#### 📋 Forecast Summary")
                
                current_price = model_data['Close'].iloc[-1]
                forecast_end_price = forecast_df['Predicted'].iloc[-1]
                price_change_forecast = forecast_end_price - current_price
                pct_change_forecast = (price_change_forecast / current_price) * 100
                
                st.markdown(f"""
                <div class="prediction-card">
                    <h4>Final Forecast Price</h4>
                    <h3>${forecast_end_price:.2f}</h3>
                    <p style="color: {'#00cc00' if pct_change_forecast >= 0 else '#ff4444'}">
                        {pct_change_forecast:+.1f}%
                    </p>
                    <p>from current ${current_price:.2f}</p>
                </div>
                """, unsafe_allow_html=True)
                
                st.metric("Model AIC", f"{fitted_model.aic:.2f}")
                st.metric("Model BIC", f"{fitted_model.bic:.2f}")
                
                # Download forecast data
                csv = forecast_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Forecast",
                    data=csv,
                    file_name=f"{selected_ticker}_forecast.csv",
                    mime="text/csv"
                )
                
        except Exception as e:
            st.error(f"❌ Model training failed: {str(e)}")
            st.info("Try adjusting model parameters or select a different time period.")

# ---------------------------
# Footer
# ---------------------------
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    <p>📊 <strong>AI Stock Forecasting Pro</strong> | Built with Streamlit & Plotly</p>
    <p>⚠️ Disclaimer: This tool is for educational purposes only. Past performance does not guarantee future results.</p>
</div>
""", unsafe_allow_html=True)
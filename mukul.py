import streamlit as st
from datetime import date
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go
import pandas as pd

START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

stocks = {
    "Reliance Industries": "RELIANCE.NS",
    "Tata Consultancy Services": "TCS.NS",
    "HDFC Bank": "HDFCBANK.NS",
    "Bharti Airtel": "BHARTIARTL.NS",
    "ICICI Bank": "ICICIBANK.NS",
    "Infosys": "INFY.NS",
    "State Bank Of India": "SBIN.NS",
    "Hindustan Unilever": "HINDUNILVR.NS",
    "ITC": "ITC.NS",
    "Life Insurance Corporation": "LIC.NS",
    "HCL Technologies": "HCLTECH.NS",
    "Larsen & Toubro": "LT.NS",
    "Bajaj Finance": "BAJFINANCE.NS",
    "Sun Pharmaceutical Industries": "SUNPHARMA.NS",
    "Mahindra & Mahindra": "M&M.NS",
    "Maruti Suzuki": "MARUTI.NS",
    "Kotak Mahindra Bank": "KOTAKBANK.NS",
    "Oil And Natural Gas Corporation": "ONGC.NS",
    "Axis Bank": "AXISBANK.NS",
    "UltraTech Cement": "ULTRACEMCO.NS"
}

st.title("📈 Stock Price Prediction App")

selected_stock = st.selectbox("Select a stock for prediction:", list(stocks.keys()))
stock_symbol = stocks[selected_stock]

n_years = st.slider("Select the number of years for prediction:", 1, 5)
period = n_years * 365

@st.cache_data
def load_data(ticker):
    try:
        data = yf.download(ticker, START, TODAY)
        if data.empty:
            st.error(f"No data found for {ticker}. Please check the ticker symbol or date range.")
            return None
        data.reset_index(inplace=True)
        return data
    except Exception as e:
        st.error(f"Failed to load data for {ticker}: {e}")
        return None


data_load_state = st.text("Loading data...")
data = load_data(stock_symbol)
if data is None:
    data_load_state.text("Loading data... Failed!")
    st.stop()
data_load_state.text("Loading data... done!")

st.subheader(f"📜 Raw Data for {selected_stock}")
st.write(data.tail())

def plot_raw_data():
    if data.empty:
        st.error("No data available to plot. Please check the stock selection or date range.")
        return
    try:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name='Stock Open', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='Stock Close', line=dict(color='orange')))
        fig.layout.update(title_text=f"📊 Time Series Data for {selected_stock} Stock Prices", xaxis_rangeslider_visible=True, xaxis_title="Date", yaxis_title="Price (INR)")
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Error during raw data plotting: {e}")


# Add Candlestick Chart
def plot_candlestick():
    if data.empty:
        st.error("No data available to plot. Please check the stock selection or date range.")
        return
    try:
      fig = go.Figure(data=[go.Candlestick(
          x=data['Date'],
          open=data['Open'],
          high=data['High'],
          low=data['Low'],
          close=data['Close'],
          increasing_line_color='green',
          decreasing_line_color='red'
      )])
      fig.update_layout(
          title=f"📈 Candlestick Chart for {selected_stock}",
          xaxis_title="Date",
          yaxis_title="Price (INR)",
          xaxis_rangeslider_visible=False
      )
      st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Error during candlestick plot: {e}")


st.subheader(f"📈 Candlestick Chart for {selected_stock}")
plot_candlestick()

# Add Stock Current Price Graph
def plot_current_price():
    if data.empty:
        st.error("No data available to plot. Please check the stock selection or date range.")
        return
    try:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='Current Price', line=dict(color='purple')))
        fig.layout.update(title_text=f"📈 Current Stock Price for {selected_stock}", xaxis_rangeslider_visible=True, xaxis_title="Date", yaxis_title="Price (INR)")
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
       st.error(f"Error during current price plot: {e}")



st.subheader(f"📊 Current Stock Price for {selected_stock}")
plot_current_price()

df_train = data[['Date','Close']].copy()
df_train['y'] = df_train[['Close']].copy()['Close']
df_train['Date'] = pd.to_datetime(df_train['Date'])
df_train = df_train.rename(columns={"Date": "ds"})
df_train = df_train.drop(['Close'], axis = 1)

# Data type validation before conversion to numeric
if 'y' in df_train.columns:
      
    try:
        df_train['y'] = pd.to_numeric(df_train['y'], errors='raise')
    except TypeError as e:
        st.error(f"TypeError during numeric conversion: {e}. This error means that 'y' column could not be converted to a numeric type. Please check for non-numeric values.")
        st.stop()
    except ValueError as e:
        st.error(f"ValueError during numeric conversion: {e}. This error means that 'y' column could not be converted to numeric as the values are not convertible. Please check raw data.")
        st.stop()
    
    if df_train['y'].isnull().any():
        st.error(f"The 'y' column contains missing or null values, and the Prophet model cannot train on missing values.")
        st.stop()

    if not isinstance(df_train['y'], pd.Series):
         st.error("The 'y' column is not a Pandas Series. This indicates a problem with data slicing")
         st.stop()

else:
    st.error("The 'y' column is missing in the DataFrame. Cannot proceed with model training.")
    st.stop()


m = Prophet()
try:
    m.fit(df_train)
except Exception as e:
    st.error(f"Error during Prophet model training: {e}. Please check the data. The error was: {e}")
    st.stop()

future = m.make_future_dataframe(periods=period)
try:
    forecast = m.predict(future)
except Exception as e:
        st.error(f"Error during Prophet model prediction: {e}. Please check the data. The error was: {e}")
        st.stop()


st.subheader(f"📊 Forecast Data for {selected_stock}")
st.write(forecast.tail())

try:
    st.subheader(f"🔮 Forecast Plot for {selected_stock}")
    fig1 = plot_plotly(m, forecast)
    st.plotly_chart(fig1)
except Exception as e:
      st.error(f"Error during forecast plot: {e}")


try:
    st.subheader(f"🔍 Forecast Components for {selected_stock}")
    fig2 = m.plot_components(forecast)
    st.write(fig2)
except Exception as e:
      st.error(f"Error during forecast component plot: {e}")
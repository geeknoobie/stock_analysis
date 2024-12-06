import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import datetime as dt
import matplotlib.dates as mdates


# Load stock ticker data (replace with your CSV file)
@st.cache_data
def load_tickers():
    stock = pd.read_csv(
        '/Users/debabratapanda/PycharmProjects/projects/stock_analysis/nasdaq_screener_1733440533426.csv')
    stocks = stock.iloc[:, [0, 1]]
    return stocks


# Fetch stock data dynamically based on selected ticker
def fetch_data_with_date_limit(ticker):
    try:
        data = yf.download(ticker)
        if data.empty:
            st.error("No data available for the selected stock.")
            return None, None, None
        first_available_date = data.index[0].date()
        last_available_date = data.index[-1].date()
        return data, first_available_date, last_available_date
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None, None, None


def prepare_lstm_data(data, lookback=60):
    # Normalize the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data["Close"].values.reshape(-1, 1))

    # Check if there is enough data for the lookback
    if len(scaled_data) <= lookback:
        return None, None, None

    # Create sequences
    X, y = [], []
    for i in range(lookback, len(scaled_data)):
        X.append(scaled_data[i - lookback:i, 0])
        y.append(scaled_data[i, 0])

    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))  # Reshape for LSTM
    return X, y, scaler


def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dense(units=25))
    model.add(Dense(units=1))  # Output layer
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


def predict_stock_lstm(data, prediction_period):
    lookback = 60
    X, y, scaler = prepare_lstm_data(data, lookback)

    # Check if the data is sufficient for LSTM
    if X is None or y is None:
        st.error("Not enough data to train the model. Please select a longer date range.")
        return None

    # Build LSTM model
    model = build_lstm_model((X.shape[1], 1))

    with st.spinner("Training the LSTM model..."):
        model.fit(X, y, batch_size=32, epochs=10, verbose=0)

    # Predict future prices
    last_60_days = data["Close"].values[-lookback:]
    last_60_days_scaled = scaler.transform(last_60_days.reshape(-1, 1))
    X_pred = np.array([last_60_days_scaled])
    X_pred = np.reshape(X_pred, (X_pred.shape[0], X_pred.shape[1], 1))

    predictions = []
    for _ in range(prediction_period):
        pred = model.predict(X_pred, verbose=0)
        predictions.append(pred[0][0])
        pred_reshaped = np.array(pred).reshape(1, 1, 1)
        X_pred = np.append(X_pred[:, 1:, :], pred_reshaped, axis=1)

    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    return predictions


def plot_predictions(data, predictions):
    st.subheader("Future Price Predictions")
    st.write("this project is intended to demonstrate coding skills and is not intended to provide financial or investment advice")

    # Get the starting date for predictions
    last_date = data.index[-1]
    future_dates = [last_date + pd.Timedelta(days=i) for i in range(1, len(predictions) + 1)]

    # Create a DataFrame for predictions
    prediction_df = pd.DataFrame({
        "Date": future_dates,
        "Predicted Price": predictions.flatten()
    })

    # Display the prediction table
    st.subheader("Predicted Prices with Dates")
    st.write(prediction_df)

    # Plot the predictions
    fig, ax = plt.subplots()

    # Use the dataset's index (dates) for historical data
    ax.plot(data.index, data["Close"].values, label="Historical Prices", color="white", linestyle="-")

    # Use generated future dates for predictions
    ax.plot(
        future_dates,
        predictions,
        label="Predicted Prices",
        color="white",
        linestyle="--",
    )

    # Format x-axis to show only months
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))  # Show only month names (e.g., Jan, Feb)
    ax.xaxis.set_major_locator(mdates.MonthLocator())         # Set major ticks to months
    ax.tick_params(axis='x', rotation=0)                     # No rotation for months

    # Set labels and title
    ax.set_xlabel("Month", color="white")
    ax.set_ylabel("Price", color="white")
    ax.set_title("Stock Price Prediction", color="white")

    # Make the legend background transparent and set text color to white
    legend = ax.legend()
    legend.get_frame().set_facecolor('none')  # Transparent background
    legend.get_frame().set_edgecolor('none')  # Remove legend border
    for text in legend.get_texts():
        text.set_color("white")  # Set legend text color to white

    # Remove plot spines (box around the plot)
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Set plot background to transparent
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")

    # Update tick color to white
    ax.tick_params(colors="white")

    # Render the plot
    st.pyplot(fig)


def main():
    st.title("Stock Analysis and Prediction App")

    # Load ticker data
    stocks = load_tickers()

    # Stock Selection Dropdown
    st.subheader("Select a Stock for Analysis")
    stock_name = st.selectbox(
        "Choose a Stock",
        options=stocks['Name'] + " (" + stocks['Symbol'] + ")",
    )
    # Extract ticker
    stock_ticker = stock_name.split("(")[-1].replace(")", "").strip()

    # Fetch data to determine date limits
    data, first_available_date, last_available_date = fetch_data_with_date_limit(stock_ticker)

    if data is not None:
        # Date Range Picker
        st.subheader("Select Date Range")
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start Date", value=first_available_date, min_value=first_available_date, max_value=last_available_date)
        with col2:
            end_date = st.date_input("End Date", value=last_available_date, min_value=first_available_date, max_value=last_available_date)

        # Display Data and Visualization
        st.subheader(f"Showing data for {stock_name}")
        filtered_data = data.loc[start_date:end_date]
        st.write(filtered_data)

        # Prediction
        st.header("Stock Price Prediction with LSTM")
        prediction_period = st.slider("Prediction Period (in days)", min_value=1, max_value=365, value=30)
        if st.button("Predict"):
            predictions = predict_stock_lstm(filtered_data, prediction_period)
            plot_predictions(filtered_data, predictions)


if __name__ == "__main__":
    main()
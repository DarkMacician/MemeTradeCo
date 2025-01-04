import json
import numpy as np
import pandas as pd
import requests
from datetime import datetime, timedelta, timezone
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import ModelCheckpoint
import time
import threading

sym = 'BNBUSDT'
path = 'Output\BNBUSDT.json'
uri = "mongodb+srv://prompt:123@cluster0.admu7.mongodb.net/NeoFeed?retryWrites=true&w=majority"
client = MongoClient(uri, server_api=ServerApi('1'))
db = client['MemeTradeCO']
btc = db['BNBUSDT']

def update_missing_data(symbol, last_timestamp_str):
    """
    Update the database with missing data from last_timestamp + 1 minute to the current time.

    Args:
        symbol (str): The trading symbol (e.g., "BTCUSDT").
        last_timestamp_str (str): The last timestamp as a string in the format '%Y-%m-%d %H:%M'.

    Returns:
        None
    """
    gmt_plus_7 = timezone(timedelta(hours=7))
    # Parse the last timestamp
    last_timestamp = datetime.strptime(last_timestamp_str, '%Y-%m-%d %H:%M').replace(tzinfo=gmt_plus_7)
    #print(last_timestamp)
    current_time = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    #print(current_time)

    # Loop to fetch and insert missing data
    while last_timestamp <= current_time - timedelta(minutes=1):  # Increment to the next minute
        try:
            # Fetch data for the next minute
            new_data = fetch_latest_data(symbol, last_timestamp)
            if new_data:
                # Insert the new data into the database
                btc.insert_one(new_data)
                print(f"Inserted data: {new_data}")
            else:
                print(f"No data fetched for {last_timestamp}.")

        except Exception as e:
            print(f"Error fetching or inserting data for {last_timestamp}: {e}")
        last_timestamp += timedelta(minutes=1)

    print("Database is up-to-date.")

def is_last_timestamp_current(last_timestamp_str):
    """
    Check if the last_timestamp is the same as the current time in UTC (rounded to the nearest minute).

    Args:
        last_timestamp_str (str): The last timestamp as a string in the format '%Y-%m-%d %H:%M'.

    Returns:
        bool: True if last_timestamp matches the current time, False otherwise.
    """
    # Parse the last timestamp string to a datetime object
    last_timestamp = datetime.strptime(last_timestamp_str, '%Y-%m-%d %H:%M').replace(tzinfo=timezone.utc)

    # Get the current time in UTC, rounded to the nearest minute
    current_time = datetime.now(timezone.utc).replace(second=0, microsecond=0)

    # Check if last_timestamp matches current_time
    return last_timestamp == current_time

# Fetch data from Binance
def fetch_latest_data(symbol, last_timestamp):
    url = "https://api.binance.com/api/v3/klines"
    interval = "1m"
    params = {
        "symbol": symbol,
        "interval": interval,
        "limit": 1,
        "startTime": int((last_timestamp + timedelta(minutes=1)).timestamp() * 1000)
    }
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        if data:
            return {
                "timestamp": datetime.fromtimestamp(data[0][0] / 1000).strftime('%Y-%m-%d %H:%M'),
                "open_price": float(data[0][1]),
                "high_price": float(data[0][2]),
                "low_price": float(data[0][3]),
                "close_price": float(data[0][4]),
                "volume": float(data[0][5]),
                "symbol": symbol
            }
        else:
            return None
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data: {e}")
        return None

class CustomModelCheckpoint(ModelCheckpoint):
    def __init__(self, filepath, **kwargs):
        super().__init__(filepath, **kwargs)
        self.best_avg_loss = np.inf

    def on_epoch_end(self, epoch, logs=None):
        avg_loss = (logs["loss"] + logs["val_loss"]) / 2
        if avg_loss < self.best_avg_loss:
            self.best_avg_loss = avg_loss
            super().on_epoch_end(epoch, logs)

def update_database_and_predict(symbol=sym, current_balance=1, risk_tolerance=0.1):
    """
    Predict the next minute's price, make a trade decision, and then update the database with the latest data.
    """
    try:
        # Fetch the most recent record from the database
        last_record = btc.find_one(sort=[("timestamp", -1)])
        last_timestamp_str = last_record['timestamp']

        if not is_last_timestamp_current(last_timestamp_str):
            update_missing_data(sym, last_record['timestamp'])

        last_record = btc.find_one(sort=[("timestamp", -1)])
        last_timestamp = datetime.strptime(last_record['timestamp'], '%Y-%m-%d %H:%M')
        data_1 = list(btc.find().sort('_id', -1).limit(1440))  # Get the last 1440 records
        df = pd.DataFrame(data_1)
        df['timestamp'] = pd.to_datetime(df['timestamp']).dt.strftime('%Y-%m-%d %H:%M')
        df = df.sort_values('timestamp')

        # Scale the 'close_price' for LSTM input
        scaler = MinMaxScaler(feature_range=(0, 1))
        df['close_price_scaled'] = scaler.fit_transform(df[['close_price']])

        # Prepare the recent data sequence for prediction
        sequence_length = 3
        recent_data = df['close_price_scaled'].iloc[-sequence_length:].values.reshape(1, sequence_length, 1)

        # Predict the next price
        predicted_scaled_price = model.predict(recent_data)
        predicted_price = scaler.inverse_transform(predicted_scaled_price)

        # Calculate the next timestamp
        predicted_timestamp = last_timestamp + timedelta(minutes=1)

        if last_record:
            current_price = last_record["close_price"]

            # Make a trade decision
            decision, amount = make_trade_decision(predicted_price[0][0], current_price, current_balance, risk_tolerance)

            # Log the decision
            print("-----------------------------------------------------------------------------------------------------------------------------")
            print(
                f"Decision - Time: {predicted_timestamp}, Decision: {decision}, Amount: {amount}, "
                f"Predicted Price: {predicted_price[0][0]}, Current Price: {current_price}"
            )
            print("-----------------------------------------------------------------------------------------------------------------------------")
            decision_data = {
                "Time": predicted_timestamp.strftime('%Y-%m-%d %H:%M'),
                "Decision": decision,
                "Amount": amount,
                "Predicted Price": float(predicted_price[0][0]),
                "Current Price": float(current_price),
            }

            # Save the decision to JSON
            with open(path, "w") as file:
                json.dump(decision_data, file, indent=4)

        else:
            print("No current data available for decision-making.")
        time.sleep(60)
        current_data = fetch_latest_data(symbol, last_timestamp)
        # Insert the new data into the database
        if current_data:
            btc.insert_one(current_data)
            print(f"Inserted new data: {current_data}")
        else:
            print("No new data fetched for updating.")

    except Exception as e:
        print(f"Error in update_database_and_predict: {e}")
        return None

# Trade decision-making
def make_trade_decision(predicted_price, current_price, current_balance, risk_tolerance=0.1):
    if predicted_price > current_price * 1.01:
        decision = "buy"
        amount_to_buy = current_balance * risk_tolerance
        return decision, amount_to_buy
    elif predicted_price < current_price * 0.99:
        decision = "sell"
        amount_to_sell = current_balance * risk_tolerance
        return decision, amount_to_sell
    else:
        decision = "hold"
        return decision, 0

def prediction_and_trade_loop(symbol=sym, current_balance=1, risk_tolerance=0.1):
    # Predict the first next-minute price immediately
    predicted_data = update_database_and_predict(symbol)
    if predicted_data:
        predicted_price = predicted_data["predicted_price"]
        timestamp = predicted_data["timestamp"]
        current_data = fetch_latest_data(symbol, timestamp - timedelta(minutes=1))
        if current_data:
            current_price = current_data["close_price"]
            decision, amount = make_trade_decision(predicted_price, current_price, current_balance, risk_tolerance)
            print(
                f"Time: {timestamp}, Decision: {decision}, Amount: {amount}, Predicted Price: {predicted_price}, Current Price: {current_price}")

    # Loop to update database and predict every 60 seconds
    while True:
        #time.sleep(60)  # Wait 60 seconds
        predicted_data = update_database_and_predict(symbol)
        if predicted_data:
            predicted_price = predicted_data["predicted_price"]
            timestamp = predicted_data["timestamp"]
            current_data = fetch_latest_data(symbol, timestamp - timedelta(minutes=1))
            if current_data:
                current_price = current_data["close_price"]
                decision, amount = make_trade_decision(predicted_price, current_price, current_balance, risk_tolerance)
                print(
                    f"Time: {timestamp}, Decision: {decision}, Amount: {amount}, Predicted Price: {predicted_price}, Current Price: {current_price}")
                data = {
                    "Time": timestamp.strftime('%Y-%m-%d %H:%M'),
                    "Decision": decision,
                    "amount": amount,
                    "Predicted Price": float(predicted_price),
                    "Current Price": float(current_price)
                }

                # Save to JSON file
                with open(path, "w") as file:
                    json.dump(data, file, indent=4)
        else:
            print("No prediction made or data unavailable.")


#Retrain the model every 5 minutes
def retrain_model_every_5min():
    global model
    while True:
        # Fetch the latest data
        data_1 = list(btc.find().sort('_id', -1).limit(1440))
        df = pd.DataFrame(data_1)
        df['timestamp'] = pd.to_datetime(df['timestamp']).dt.strftime('%Y-%m-%d %H:%M')
        df = df.sort_values('timestamp')
        scaler = MinMaxScaler(feature_range=(0, 1))
        df['close_price_scaled'] = scaler.fit_transform(df[['close_price']])

        # Prepare data
        X, y = [], []
        for i in range(len(df) - 3):
            X.append(df['close_price_scaled'].iloc[i:i + 3].values)
            y.append(df['close_price_scaled'].iloc[i + 3])
        X, y = np.array(X), np.array(y)
        X = X.reshape((X.shape[0], X.shape[1], 1))
        train_size = int(0.8 * len(X))
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]

        # Retrain the model
        model.fit(X_train, y_train, epochs=5, batch_size=128, validation_data=(X_test, y_test),
                  callbacks=[checkpoint_callback])
        model.load_weights(checkpoint_filepath)
        print("Model retrained.")
        time.sleep(3600)


#LSTM Model setup
last_record = btc.find_one(sort=[("timestamp", -1)])
print(last_record['timestamp'])
#print(is_last_timestamp_current(last_record['timestamp']))
if not is_last_timestamp_current(last_record['timestamp']):
    update_missing_data(sym, last_record['timestamp'])

data_1 = list(btc.find({"symbol": sym}).sort('_id', -1).limit(43200))
df = pd.DataFrame(data_1)
df['timestamp'] = pd.to_datetime(df['timestamp']).dt.strftime('%Y-%m-%d %H:%M')
df = df.sort_values('timestamp')
scaler = MinMaxScaler(feature_range=(0, 1))
df['close_price_scaled'] = scaler.fit_transform(df[['close_price']])
sequence_len = 3
X, y = [], []
for i in range(len(df) - sequence_len):
    X.append(df['close_price_scaled'].iloc[i:i + sequence_len].values)
    y.append(df['close_price_scaled'].iloc[i + sequence_len])
X, y = np.array(X), np.array(y)
X = X.reshape((X.shape[0], X.shape[1], 1))
train_size = int(0.8 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]
model = Sequential([
    LSTM(64, activation='relu', return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    LSTM(32, activation='relu'),
    Dense(1)
])
model.compile(optimizer='adam', loss='mean_squared_error')
checkpoint_filepath = "Weights/"+sym+'.weights.h5'
checkpoint_callback = CustomModelCheckpoint(
    filepath=checkpoint_filepath,
    monitor="val_loss",
    save_best_only=True,
    save_weights_only=True,
    verbose=1
)
model.fit(X_train, y_train, epochs=20, batch_size=128, validation_data=(X_test, y_test),
          callbacks=[checkpoint_callback])
model.load_weights(checkpoint_filepath)

prediction_and_trade_loop(sym)

threading.Thread(target=retrain_model_every_5min, daemon=True).start()
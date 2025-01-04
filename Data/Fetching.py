from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
import requests
from datetime import datetime

# MongoDB Connection
uri = "mongodb+srv://prompt:123@cluster0.admu7.mongodb.net"
client = MongoClient(uri, server_api=ServerApi('1'))
db = client['MemeTradeCO']

# Dictionary mapping symbols to their collections
collections = {
    "BTCUSDT": db['BTCUSDT'],
    "ETHUSDT": db['ETHUSDT'],
    "SOLUSDT": db['SOLUSDT'],
    "BNBUSDT": db['BNBUSDT'],
    "ADAUSDT": db['ADAUSDT']
}

def fetch_historical_1m_data(symbol, start_time, end_time):
    """
    Fetch historical data for a given symbol at 1-minute intervals.
    """
    url = "https://api.binance.com/api/v3/klines"
    interval = "1m"
    limit = 1000
    all_data = []

    while start_time < end_time:
        params = {
            "symbol": symbol,
            "interval": interval,
            "limit": limit,
            "startTime": start_time,
            "endTime": end_time
        }
        response = requests.get(url, params=params)
        response.raise_for_status()  # Raise error if the request fails
        data = response.json()
        if not data:
            break
        all_data.extend(data)
        start_time = data[-1][0] + 1  # Move to the next interval

    return all_data


def save_to_mongodb(data, symbol):
    """
    Save historical data to MongoDB in the appropriate collection.
    """
    formatted_data = [
        {
            "timestamp": datetime.fromtimestamp(entry[0] / 1000).strftime('%Y-%m-%d %H:%M'),  # Convert ms to datetime
            "open_price": float(entry[1]),
            "high_price": float(entry[2]),
            "low_price": float(entry[3]),
            "close_price": float(entry[4]),
            "volume": float(entry[5]),
            "symbol": symbol
        }
        for entry in data
    ]

    # Use the collection corresponding to the symbol
    collection = collections[symbol]
    collection.insert_many(formatted_data)
    print(f"Data for {symbol} successfully saved to its MongoDB collection.")


def fetch_btc_data(symbol="BTCUSDT", start_time="2025-01-01 12:51", end_time="2025-01-02 12:51"):
    """
    Fetch and save historical data for a specific symbol.
    """
    start_time = int(datetime.strptime(start_time, "%Y-%m-%d %H:%M").timestamp() * 1000)
    end_time = int(datetime.strptime(end_time, "%Y-%m-%d %H:%M").timestamp() * 1000)

    print(f"Fetching data for {symbol}...")
    data = fetch_historical_1m_data(symbol, start_time, end_time)
    save_to_mongodb(data, symbol)


if __name__ == "__main__":
    symbols = ["ETHUSDT", "BNBUSDT", 'ADAUSDT', 'SOLUSDT']  # Specify the 5 coins
    start_time = "2024-12-03 11:20"
    end_time = "2025-01-04 11:26"

    for symbol in symbols:
        fetch_btc_data(symbol=symbol, start_time=start_time, end_time=end_time)

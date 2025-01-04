from fastapi import FastAPI
import json
from pathlib import Path

app = FastAPI()

# File paths to the JSON files
json_files = [
    'Currencies\Output\BTCUSDT.json',
    'Currencies\Output\ETHUSDT.json',
    'Currencies\Output\BNBUSDT.json',
    'Currencies\Output\ADAUSDT.json',
    'Currencies\Output\SOLUSDT.json'
]


# Function to read JSON file
def read_json(file_path: str):
    with open(file_path, 'r') as file:
        return json.load(file)


@app.get("/prediction")
async def read_all_json():
    all_data = {}

    # Read all JSON files and store their contents in a dictionary
    for file_path in json_files:
        try:
            data = read_json(file_path)
            currency = Path(file_path).stem  # Extract the file name without extension (e.g., BTCUSDT)
            all_data[currency] = data
        except Exception as e:
            all_data[currency] = {"error": str(e)}

    return {"files_data": all_data}
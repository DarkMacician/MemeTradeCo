import base64
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, validator
from diffusers import DiffusionPipeline
import torch
from io import BytesIO
import uvicorn
import json
from pathlib import Path


app = FastAPI()

# Allowed origins
origins = [
    "http://localhost:3000",
    "https://memetrade-co.fun",
    "https://www.memetrade-co.fun"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# File paths to the JSON files
json_files = [
    'Currencies/Output/BTCUSDT.json',
    'Currencies/Output/ETHUSDT.json',
    'Currencies/Output/BNBUSDT.json',
    'Currencies/Output/ADAUSDT.json',
    'Currencies/Output/SOLUSDT.json'
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
            return {"error": str(e)}

    return {"files_data": all_data}
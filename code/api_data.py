import json
import requests
import csv
import time
from decimal import Decimal

API_KEY = "FQUSQEZUXGZPUIZCXUKIZMK8MMM1YJ568V"
CONTRACT_ADDR = "0xfC98e825A2264D890F9a1e68ed50E1526abCcacD"
BASE_URL = "https://api.etherscan.io/api"

def fetch_transactions(start_block, end_block):
    params = {
        "module": "account",
        "action": "tokentx",
        "contractaddress": CONTRACT_ADDR,
        "startblock": start_block,
        "endblock": end_block,
        "sort": "asc",
        "apikey": API_KEY
    }
    response = requests.get(BASE_URL, params=params)
    return response.json().get("result", [])

def format_value(value, decimals):
    return Decimal(value) / Decimal(10**decimals)

with open("datasets/mco2_transactions1.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile)

    if csvfile.tell() == 0:
        writer.writerow([
            "blockNumber", "timeStamp", "hash", "nonce", "blockHash",
            "from", "contractAddress", "to", "value", "tokenName",
            "tokenSymbol", "tokenDecimal", "transactionIndex", "gas",
            "gasPrice", "gasUsed", "cumulativeGasUsed", "input", "confirmations"
        ])

    all_data = []
    current_start_block = 0
    current_end_block = 99999999
    total = 0

    while True:
        try:
            data = fetch_transactions(current_start_block, current_end_block)
            all_data.extend(data)
            if not data:
                print("No more data. Exiting.")
                break

            # Process and write data
            for tx in data:
                decimals = int(tx.get("tokenDecimal", 0))
                tx["value"] = str(format_value(tx["value"], decimals))
                tx["gasPrice"] = str(format_value(tx["gasPrice"], decimals))
                tx["gasUsed"] = str(format_value(tx["gasUsed"], decimals))
                tx["cumulativeGasUsed"] = str(format_value(tx["cumulativeGasUsed"], decimals))
                tx["confirmations"] = str(format_value(tx["confirmations"], decimals))

                writer.writerow([tx.get(field, "") for field in [
                    "blockNumber", "timeStamp", "hash", "nonce", "blockHash",
                    "from", "contractAddress", "to", "value", "tokenName",
                    "tokenETSymbol", "tokenDecimal", "transactionIndex", "gas",
                    "gasPrice", "gasUsed", "cumulativeGasUsed", "input", "confirmations"
                ]])

            last_block = int(data[-1]["blockNumber"])
            current_start_block = last_block + 1
            total += len(data)
            print(f"Fetched blocks {current_start_block}-{current_end_block}: +{len(data)} (Total: {total})")

            if len(data) < 1000:
                current_end_block = current_start_block + 10000
            time.sleep(0.2)

        except Exception as e:
            print(f"Error: {e}, retrying...")
            time.sleep(5)

    with open("mco2_transactions.json1", "w") as jsonfile:
        json.dump(all_data, jsonfile, indent=4)

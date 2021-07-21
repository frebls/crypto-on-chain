# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'

# %%
import requests
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

apiKey = '71522f0d83a01e1974a110bcee4744b284e12b18e7cafdeb09d706a186f86bf5'
url = "https://min-api.cryptocompare.com/data/histominute"

params = {
    "api_key": apiKey,
    "fsym": "ETH",
    "tsym": "USD",
    "limit": 2000
}

i = 0

while True:
    response = requests.get(url, params=params).json()
    
    if response['Response'] != 'Success':
        print(response)
        break
    
    if i == 0:
        df = pd.DataFrame(response['Data'])
    else:
        df_ = pd.DataFrame(response['Data'])
        df = pd.concat([df, df_])
        del df_
        
    i += 1
    params["toTs"] = min(df['time']) - 1
    print(i, datetime.fromtimestamp(params["toTs"]).strftime('%d/%m/%Y, %H:%M:%S'))

# %%
df.set_index('time', inplace=True)
df.to_csv('cryptocompare_eth_price_histominute.csv')

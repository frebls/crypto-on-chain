# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'

# %%
from pandas.core.indexes.base import Index
import requests
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import io

def _url(path):
    return 'https://min-api.cryptocompare.com/data/histo/minute/daily?' + path

apiKey = "71522f0d83a01e1974a110bcee4744b284e12b18e7cafdeb09d706a186f86bf5"
dates = pd.date_range(start="2016-01-01",end="2021-07-25")

i = 0

for d in dates:
    response = requests.get(_url(f'fsym=ETH&tsym=USD&date={d.strftime("%Y-%m-%d")}&api_key={apiKey}'))
    if response.status_code != 200:
        print(d.strftime("%Y-%m-%d"), response)
        break
    
    if i == 0:
        df = pd.read_csv(io.StringIO(response.content.decode('utf-8')))
    else:
        df_ = pd.read_csv(io.StringIO(response.content.decode('utf-8')))
        df = pd.concat([df, df_])
        del df_
        
    i += 1
    print(d.strftime("%Y-%m-%d"), df.shape[0], "{:.2%}".format(df.shape[0]/1e6))
    
    if df.shape[0] >= 1e6 or d == max(dates):
        start_dt = datetime.fromtimestamp(min(df['time'])).strftime("%d%m%Y")
        end_dt = datetime.fromtimestamp(max(df['time'])).strftime("%d%m%Y")
        df.set_index('time', inplace=True)
        df.to_csv(f'data/eth_usd_min_{start_dt}_to_{end_dt}.csv')
        i = 0


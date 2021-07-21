# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'

# %%
import requests
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

api_key = '2ead02859ec550271e1e11dce64258a23f0429e0cb2cf045237d085476bc8709'

def _url(path):
    return 'https://min-api.cryptocompare.com' + path

# %%
response = requests.get(_url(f'/data/blockchain/histo/day?fsym=ETH&limit=2000&api_key={api_key}')).json()

if response.status_code != 200:
    raise f'APIError: status = {response.status_code}'

blckchn_hist = pd.DataFrame(response['Data']['Data'])
blckchn_hist.to_csv('cryptocompare_blockchain_hist.csv', index=False)

# %%
response = requests.get(_url(f'/data/v2/histoday?fsym=ETH&tsym=USD&limit=2000&api_key={api_key}'))

if response.status_code != 200:
    raise f'APIError: status = {response.status_code}'

eth_price_hist = pd.DataFrame(response.json()['Data']['Data'])
eth_price_hist.to_csv('cryptocompare_price_hist.csv', index=False)

# %%
# print(min(eth_price_hist['time']), max(eth_price_hist['time']))
test1 = pd.read_csv(r'C:\Users\Francesco\Desktop\git_repo\UCL_thesis\crypto-on-chain\data_retrieval\ETH_1min.csv')

# %%

plt.plot(test1['Close'], test1['Date'], 'b-', linewidth = .7)
plt.show()
# %%

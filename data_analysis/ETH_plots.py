# %%
import os
from google.cloud.bigquery.client import Client
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# %%
# set Google BigQuery client
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'C:\\Users\\Francesco\\fiery-rarity-322109-6ba6fa8a811c.json'
bq_client = Client()

# set seaborn plotting theme
sns.set_theme()

# set colour palette
COLORS_PALETTE = ['#00388F', '#FFB400', '#FF4B00', '#65B800', '#00B1EA']
sns.set_palette(sns.color_palette(COLORS_PALETTE))
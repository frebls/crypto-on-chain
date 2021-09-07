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
pal = ['#00388F', '#FFB400', '#FF4B00', '#65B800', '#00B1EA']
sns.set_palette(sns.color_palette(pal))

# %% [markdown]
## Nr of traces

# %%
# run query
traces_count_df = bq_client.query('''
    select TIMESTAMP_TRUNC(block_timestamp, MONTH) as date
    , sum(case when trace_type = 'call' and to_contract_type is not null then 1 else 0 end) as call
    , sum(case when trace_type = 'call' and to_contract_type is null then 1 else 0 end) as transfer
    , sum(case when trace_type = 'reward' then 1 else 0 end) as reward
    , sum(case when trace_type = 'create' then 1 else 0 end) as create_
    , sum(case when trace_type = 'suicide' then 1 else 0 end) as suicide
    , sum(case when trace_type = 'genesis' then 1 else 0 end) as genesis
    , sum(case when trace_type = 'daofork' then 1 else 0 end) as daofork
    FROM `fiery-rarity-322109.ethereum.traces_new`
    group by date
    order by date
''').to_dataframe()

# %%
fig, ax1 = plt.subplots(2, 1, figsize=(8, 6))

ax1[0].ticklabel_format(axis="y", style="sci", scilimits=(0,0))
ax1[0].stackplot(traces_count_df.date, traces_count_df.transfer, traces_count_df.call, labels = ['transfer', 'call'], colors = pal, linewidth = 0)
ax1[0].legend(loc = 'upper left', frameon = False)
ax1[0].set_ylabel('Number of records')

ax1[1].stackplot(traces_count_df.date, traces_count_df.reward, traces_count_df.create_, traces_count_df.suicide
                 , labels = ['reward', 'create', 'suicide'], colors = pal, linewidth = 0)
ax1[1].legend(loc = 'upper left', frameon = False)
ax1[1].ticklabel_format(axis="y", style="sci", scilimits=(0,0))
ax1[1].set_ylabel('Number of records')
# ax1[1].set_ylim(ymax = 1e3)
ax2 = ax1[1].twinx()
ax2.stackplot(traces_count_df.date, traces_count_df.genesis, traces_count_df.daofork
                 , labels = ['genesis', 'daofork'], colors = pal[3:], linewidth = 0)
ax2.legend(loc = 'upper right', frameon = False)
ax2.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
ax2.set_ylabel('Number of records')
ax2.grid(False)
# ax2.set_ylim(ymax = 1e3)

# plt.tight_layout()
# plt.show()

# plt.savefig('charts/number_transactions_monthly.pdf')

# %% [markdown]
## Traces value

# %%
# run query
traces_value_df = bq_client.query('''
    select TIMESTAMP_TRUNC(block_timestamp, MONTH) as date
    , sum(case when trace_type = 'call' and to_contract_type is not null then value else 0 end) / 1e18 as call
    , sum(case when trace_type = 'call' and to_contract_type is null then value else 0 end) / 1e18 as transfer
    , sum(case when trace_type = 'reward' then value else 0 end) / 1e18 as reward
    , sum(case when trace_type = 'create' then value else 0 end) / 1e18 as create_
    , sum(case when trace_type = 'suicide' then value else 0 end) / 1e18 as suicide
    , sum(case when trace_type = 'genesis' then value else 0 end) / 1e18 as genesis
    , sum(case when trace_type = 'daofork' then value else 0 end) / 1e18 as daofork
    FROM `fiery-rarity-322109.ethereum.traces_new`
    group by date
    order by date
''').to_dataframe()

# %%
fig, ax1 = plt.subplots(2, 1, figsize=(8,6))

ax1[0].ticklabel_format(axis="y", style="sci", scilimits=(0,0))
ax1[0].stackplot(traces_value_df.date, traces_value_df.transfer, traces_value_df.call, labels = ['transfer', 'call'], colors = pal, linewidth = 0)
ax1[0].legend(loc = 'upper left', frameon = False)
ax1[0].set_ylabel('Value, ETH')

ax1[1].stackplot(traces_value_df.date, traces_value_df.reward, traces_value_df.create_, traces_value_df.suicide
                 , labels = ['reward', 'create', 'suicide'], colors = pal, linewidth = 0)
ax1[1].legend(loc = 'upper left', frameon = False)
ax1[1].ticklabel_format(axis="y", style="sci", scilimits=(0,0))
ax1[1].set_ylabel('Value, ETH')


ax2 = ax1[1].twinx()
ax2.stackplot(traces_value_df.date, traces_value_df.genesis, traces_value_df.daofork
                 , labels = ['genesis', 'daofork'], colors = pal[3:], linewidth = 0)
ax2.legend(loc = 'upper right', frameon = False)
ax2.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
ax2.set_ylabel('Value, ETH')
ax2.grid(False)


# plt.tight_layout()
# plt.show()

# plt.savefig('charts/value_transactions_monthly.pdf')

# %%

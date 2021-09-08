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

df = pd.read_csv('data/ETH_features_hour.csv')

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
# '' number_transactions_monthly '''

fig, ax1 = plt.subplots(2, 1, figsize=(8, 6))

ax1[0].ticklabel_format(axis="y", style="sci", scilimits=(0,0))
ax1[0].stackplot(traces_count.date, traces_count.transfer, traces_count.call, labels = ['transfer', 'call'], colors = pal, linewidth = 0)
ax1[0].legend(loc = 'upper left', frameon = False)
ax1[0].set_ylabel('Number of records')

ax1[1].stackplot(traces_count.date, traces_count.reward, traces_count.create_, traces_count.suicide
                 , labels = ['reward', 'create', 'suicide'], colors = pal, linewidth = 0)
ax1[1].legend(loc = 'upper left', frameon = False)
ax1[1].ticklabel_format(axis="y", style="sci", scilimits=(0,0))
ax1[1].set_ylabel('Number of records')
# ax1[1].set_ylim(ymax = 1e3)
ax2 = ax1[1].twinx()
ax2.stackplot(traces_count.date, traces_count.genesis, traces_count.daofork
                 , labels = ['genesis', 'daofork'], colors = pal[3:], linewidth = 0)
ax2.legend(loc = 'upper right', frameon = False)
ax2.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
ax2.set_ylabel('Number of records')
ax2.grid(False)
# ax2.set_ylim(ymax = 1e3)

# plt.tight_layout()
# plt.show()

plt.savefig('charts/number_transactions_monthly.pdf')


# %%
# '' value_transactions_monthly '''

fig, ax1 = plt.subplots(2, 1, figsize=(8,6))

ax1[0].ticklabel_format(axis="y", style="sci", scilimits=(0,0))
ax1[0].stackplot(traces_value.date, traces_value.transfer, traces_value.call, labels = ['transfer', 'call'], colors = pal, linewidth = 0)
ax1[0].legend(loc = 'upper left', frameon = False)
ax1[0].set_ylabel('Value, ETH')

ax1[1].stackplot(traces_value.date, traces_value.reward, traces_value.create_, traces_value.suicide
                 , labels = ['reward', 'create', 'suicide'], colors = pal, linewidth = 0)
ax1[1].legend(loc = 'upper left', frameon = False)
ax1[1].ticklabel_format(axis="y", style="sci", scilimits=(0,0))
ax1[1].set_ylabel('Value, ETH')


ax2 = ax1[1].twinx()
ax2.stackplot(traces_value.date, traces_value.genesis, traces_value.daofork
                 , labels = ['genesis', 'daofork'], colors = pal[3:], linewidth = 0)
ax2.legend(loc = 'upper right', frameon = False)
ax2.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
ax2.set_ylabel('Value, ETH')
ax2.grid(False)


# plt.tight_layout()
# plt.show()

plt.savefig('charts/value_transactions_monthly.pdf')


# %%
# '' value_transactions_monthly1 '''

fig, ax1 = plt.subplots(1, 1, figsize=(8,4))

ax1.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
ax1.stackplot(traces_value1.date, traces_value1.transfer, traces_value1.call, labels = ['transfer', 'call'], colors = pal, linewidth = 0)
ax1.legend(loc = 'upper left', frameon = False)
ax1.set_ylabel('Value, ETH')

plt.savefig('charts/value_transactions_monthly1.pdf')


# %%
# '' supply_and_market_cap '''


market_cap = pd.concat([supply, market_data], axis=1, join="inner")
market_cap = (market_cap.supply_genesis + market_cap.supply_reward) * market_cap.close

fig, ax1 = plt.subplots()

ax2 = ax1.twinx()
ax1.ticklabel_format(axis = "y", style = "sci", scilimits = (0,0))
ax1.stackplot(supply.date, supply.supply_genesis, supply.supply_reward, labels = ['Genesis', 'Reward'], colors = pal[:2], linewidth = 0)
ax1.legend(loc = 'upper left', frameon = False)
ax1.set_ylabel('Supply, ETH')
ax2.plot(market_data.date, market_cap, pal[2], linewidth = 1) #, label = ""
ax2.set_ylabel('Market Cap, USD', color = pal[2])
ax2.ticklabel_format(axis = "y", style = "sci", scilimits = (0,0))
ax2.grid(False)
# plt.show()

plt.savefig("charts/supply_and_market_cap.pdf")


# %%
# '' number_contracts '''


fig, ax = plt.subplots(2, 2, figsize=(15, 8))

ax[0, 0].plot(new_addresses.date, new_addresses.new_addresses, color = pal[0], linewidth = .2)
ax[0, 0].set_ylabel('New addresses')
ax[0, 0].ticklabel_format(axis="y", style="sci", scilimits=(0,0))

ax[0, 1].plot(new_addresses.date, new_addresses.new_erc20, color = pal[0], linewidth = .2)
ax[0, 1].set_ylabel('New ERC-20 contracts')

ax[1, 0].plot(new_addresses.date, new_addresses.new_erc721, color = pal[0], linewidth = .2)
ax[1, 0].set_ylabel('New ERC-721 contracts')

ax[1, 1].plot(new_addresses.date, new_addresses.new_other_contract, color = pal[0], linewidth = .2)
ax[1, 1].set_ylabel('New other contracts')
ax[1, 1].ticklabel_format(axis="y", style="sci", scilimits=(0,0))

plt.tight_layout()
# plt.show()

plt.savefig("charts/number_contracts.pdf")


# %%
# '' price_and_return '''


import matplotlib.ticker as mtick

fig, ax1 = plt.subplots(2, 1, figsize=(8, 6))

ax1[0].fill_between(market_data.date, market_data.volume, color = pal[1], linewidth = .5)
ax1[0].set_ylabel('Volume, USD', color = pal[1])

ax2 = ax1[0].twinx()
ax2.plot(market_data.date, market_data.close, color = pal[0], linewidth = .5)
ax2.set_ylabel('Price, USD', color = pal[0])
ax2.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))
ax2.grid(False)

ax3 = ax1[0].twinx()
ax3.set_ylabel('Price, USD (log scale)', color = pal[2])
ax3.spines['right'].set_position(('outward', 60))
ax3.yaxis.set_ticks_position('right')
ax3.yaxis.set_label_position('right')
ax3.plot(market_data.date, market_data.close, color = pal[2], linewidth = .5)
ax3.set_yscale('log')
ax3.grid(False)

log_returns = np.log(market_data.close / market_data.close.shift(1))
high_low = (market_data.high - market_data.low) / market_data[['high', 'low']].mean(axis=1)

ax1[1].plot(market_data.date, log_returns, color = pal[0], label = "", linewidth = .5)
ax1[1].set_ylabel('Log return', color = pal[0])
ax1[1].yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=None, symbol='%')) #, is_latex=False

ax2 = ax1[1].twinx()
ax2.plot(market_data.date, high_low, color = pal[1], linewidth = .5)
ax2.set_ylabel('High - Low', color = pal[1])
ax2.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=None, symbol='%'))
ax2.grid(False)

plt.tight_layout()
# plt.show()

plt.savefig("charts/price_and_return.pdf")


# %%
# '' avg_gas_fees '''


fig, ax1 = plt.subplots(2, 1, figsize=(8, 8))

ax1[0].plot(traces_value.date, traces_value.std_call, color = pal[0], linewidth = .5)
ax1[0].set_ylabel('Avg. transferred amount, ETH (log scale)', color = pal[0])
ax1[0].set_yscale('log')

ax2 = ax1[0].twinx()
ax2.plot(traces_value.date, traces_value.std_call, color = pal[1], linewidth = .5)
ax2.set_ylabel('Avg. transferred amount, ETH', color = pal[1])
ax2.grid(False)

fee_ = traces_value.merge(market_data1, on = 'date', how = 'left')

ax1[1].scatter(fee_.close, fee_.std_call, color = pal[0], s = .5)
ax1[1].set_xlabel('Price, USD')
ax1[1].set_ylabel('Std. transferred amount, ETH')
ax1[1].set_xscale('log')
ax1[1].set_yscale('log')

# plt.tight_layout()
# plt.show()

# plt.savefig("charts/avg_gas_fees.pdf")


# %%
# '' Avg transferred amount '''

fig, ax1 = plt.subplots()

ax1.plot(traces_value1.date, traces_value1.avg_call, color = pal[0], linewidth = .2)
ax1.set_ylabel('Avg transferred amount, USD (log scale)', color = pal[0])
ax1.set_yscale('log')


# %%
# '' gini coefficient '''


fig, ax1 = plt.subplots(2, 1, figsize=(8, 8))

ax1[0].plot(gini.date, gini.gini, color = pal[0], linewidth = .8)
ax1[0].set_ylabel('Gini coefficient')

gini_ = gini.merge(market_data1, on = 'date', how = 'left')

ax1[1].scatter(gini_.close, gini_.gini, color = pal[0], s = .8)
ax1[1].set_xlabel('Price, USD')
ax1[1].set_ylabel('Gini coefficient')
ax1[1].set_xscale('log')
# ax1[1].set_yscale('log')

# plt.tight_layout()
# plt.show()

plt.savefig("charts/gini.pdf")


# %%
# '' Avg. Gas fees, ETH '''

fig, ax1 = plt.subplots(2, 1, figsize=(8, 8))

ax1[0].plot(fee1.date, fee1.avg_fee_usd, color = pal[0], linewidth = .5)
ax1[0].set_ylabel('Avg. Gas fees, ETH')

# ax2 = ax1[0].twinx()
# ax2.plot(fee.date, fee.avg_fee_usd, color = pal[0], linewidth = .5)
# ax2.set_ylabel('Avg fees, USD (log scale)', color = pal[0])
# ax2.set_yscale('log')
# ax2.grid(False)

fee_ = fee1.merge(market_data1, on = 'date', how = 'left')

ax1[1].scatter(fee_.close, fee_.avg_fee_usd, color = pal[0], s = .5)
ax1[1].set_xlabel('Price, USD')
ax1[1].set_ylabel('Avg. Gas fees, ETH')
# ax1[1].set_xscale('log')
ax1[1].set_yscale('log')

# plt.tight_layout()
# plt.show()

plt.savefig("charts/avg_gas_fees.pdf")


# %%
# '' degrees_power_law '''

fig, ax = plt.subplots(1, 2, figsize=(10, 4)) #

ax[0].scatter(indegree.indegree, indegree.frequency, color = pal[0], marker = '.')
ax[0].set_ylabel('Frequency')
ax[0].set_xlabel('inDegree')
ax[0].set_yscale('log')
ax[0].set_xscale('log')

ax[1].scatter(outdegree.outdegree, outdegree.frequency, color = pal[0], marker = '.')
ax[1].set_ylabel('Frequency')
ax[1].set_xlabel('outDegree')
ax[1].set_yscale('log')
ax[1].set_xscale('log')

plt.tight_layout()
# plt.show()

plt.savefig("charts/degrees_power_law.pdf")


# %%
# '' avg_degree '''

fig, ax1 = plt.subplots(2, 1, figsize=(8, 8))

ax1[0].plot(degree1.date, degree1.avg_indegree, color = pal[0], linewidth = .5)
ax1[0].set_ylabel('Avg. degree')
# ax1[0].set_yscale('log')

degree_ = degree1.merge(market_data1, on = 'date', how = 'left')

ax1[1].scatter(degree_.close, degree_.avg_indegree, color = pal[0], s = .5)
ax1[1].set_xlabel('Price, USD')
ax1[1].set_ylabel('Avg. degree')
# ax1[1].set_yscale('log')
ax1[1].set_xscale('log')

# plt.tight_layout()
# plt.show()

plt.savefig("charts/avg_degree.pdf")


# %%
# '' Std. indegree '''

fig, ax1 = plt.subplots(2, 1, figsize=(8, 8))

ax1[0].plot(degree1.date, degree1.stddev_indegree, color = pal[0], linewidth = .5)
ax1[0].set_ylabel('Std. indegree')
# ax1[0].set_yscale('log')

degree_ = degree1.merge(market_data1, on = 'date', how = 'left')

ax1[1].scatter(degree_.close, degree_.stddev_indegree, color = pal[0], s = .5)
ax1[1].set_xlabel('Price, USD')
ax1[1].set_ylabel('Std. indegree')
# ax1[1].set_yscale('log')
# ax1[1].set_xscale('log')

# plt.tight_layout()
# plt.show()

plt.savefig("charts/std_indegree.pdf")


# %%
# '' Std. outdegree '''


fig, ax1 = plt.subplots(2, 1, figsize=(8, 8))

ax1[0].plot(degree1.date, degree1.stddev_outdegree, color = pal[0], linewidth = .5)
ax1[0].set_ylabel('Std. outdegree')
# ax1[0].set_yscale('log')

degree_ = degree1.merge(market_data1, on = 'date', how = 'left')

ax1[1].scatter(degree_.close, degree_.stddev_outdegree, color = pal[0], s = .5)
ax1[1].set_xlabel('Price, USD')
ax1[1].set_ylabel('Std. outdegree')
# ax1[1].set_yscale('log')
# ax1[1].set_xscale('log')

# plt.tight_layout()
# plt.show()

plt.savefig("charts/std_outdegree.pdf")

ax2 = ax1.twinx()
ax2.plot(traces_value1.date, traces_value1.std_call, color = pal[1], linewidth = .2)
ax2.set_ylabel('Std transferred amount, USD', color = pal[1])
ax2.grid(False)



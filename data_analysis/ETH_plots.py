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
df.date = pd.to_datetime(df.date)


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

# plt.savefig("charts/price_and_return.pdf")


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

# plt.savefig('charts/number_transactions_monthly.pdf')


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

# plt.savefig('charts/value_transactions_monthly.pdf')


# %%
# '' value_transactions_monthly1 '''

fig, ax1 = plt.subplots(1, 1, figsize=(8,4))

ax1.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
ax1.stackplot(traces_value1.date, traces_value1.transfer, traces_value1.call, labels = ['transfer', 'call'], colors = pal, linewidth = 0)
ax1.legend(loc = 'upper left', frameon = False)
ax1.set_ylabel('Value, ETH')

# plt.savefig('charts/value_transactions_monthly1.pdf')


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

# plt.savefig("charts/supply_and_market_cap.pdf")


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

# plt.savefig("charts/number_contracts.pdf")


# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

# %%
# ''' number of addresses '''

fig, ax1 = plt.subplots(2, 1, figsize=(8, 8))

ax1[0].plot(df.date, df.nr_addresses, color = pal[0], linewidth = .1)
ax1[0].set_ylabel('New addresses') #, color = pal[0])
ax1[0].ticklabel_format(axis="y", style="sci", scilimits=(0,0))
ax1[0].set_ylim([0, 2.5e4])

# ax2 = ax1[0].twinx()
# ax2.plot(df.date, df.address_count, color = pal[1], linewidth = 1.5)
# ax2.set_ylabel('Number of addresses', color = pal[1])
# ax2.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
# ax2.grid(False)

ax1[1].scatter(df.close, df.nr_addresses, color = pal[0], s = .1)
# ax1[1].ticklabel_format(axis="y", style="sci", scilimits=(0,0))
ax1[1].set_xlabel('Price, USD (log scale)')
ax1[1].set_ylabel('New addresses (log scale)')
ax1[1].set_yscale('log')
ax1[1].set_xscale('log')

# plt.savefig("charts/nr_address.png")

# %%
# '' number of value transfers '''

fig, ax1 = plt.subplots(2, 1, figsize=(8, 8))

ax1[0].plot(df.date, df.nr_call, color = pal[0], linewidth = .1)
ax1[0].set_ylabel('Nr. of value transfers')
ax1[0].ticklabel_format(axis="y", style="sci", scilimits=(0,0))

ax1[1].scatter(df.close, df.nr_call, color = pal[0], s = .1)
# ax1[1].ticklabel_format(axis="y", style="sci", scilimits=(0,0))
ax1[1].set_xlabel('Price, USD (log scale)')
ax1[1].set_ylabel('Nr. of value transfers (log scale)')
ax1[1].set_yscale('log')
ax1[1].set_xscale('log')

# plt.savefig("charts/nr_calls.png")


# %%
# '' gas fees '''

fig, ax1 = plt.subplots(figsize=(8, 4)) #2, 1, 

# ax1[0].plot(df.date, df.stddev_fee_usd, color = pal[1], linewidth = .3)
# ax1[0].set_ylabel('StDev. Gas fees, ETH', color = pal[1])
ax1.plot(df.date, df.avg_fee_usd, color = pal[1], linewidth = .1)
ax1.set_ylabel('Avg. Gas fees, USD (log scale)', color = pal[1])
ax1.set_yscale('log')

ax2 = ax1.twinx()
ax2.plot(df.date, df.avg_fee_usd, color = pal[0], linewidth = .1)
ax2.set_ylabel('Avg. Gas fees, USD', color = pal[0])
ax2.set_ylim([-1, 100])
ax2.grid(False)

# ax1[1].scatter(df.close, df.avg_fee_usd, color = pal[0], s = .1)
# # ax1[1].ticklabel_format(axis="y", style="sci", scilimits=(0,0))
# ax1[1].set_xlabel('Price, USD (log scale)')
# ax1[1].set_ylabel('Avg. Gas fees, ETH (log scale)')
# ax1[1].set_yscale('log')
# ax1[1].set_xscale('log')

plt.savefig("charts/gas_fees.png")


# %%
# '' call_value '''

# call_df = bq_client.query('''
#     select TIMESTAMP_TRUNC(block_timestamp, HOUR) as date
#     , avg(value / 1e18) as avg_call_value_usd
#     , stddev(value / 1e18) as stddev_call_value_usd
#     , count(*) nr_call
#     FROM `fiery-rarity-322109.ethereum.traces_new`
#     where date(block_timestamp) >= '2017-01-01' and trace_type = 'call'
#     group by date
#     order by date
# ''').to_dataframe()

fig, ax1 = plt.subplots(2, 1, figsize=(8, 8))

ax1[0].plot(df.date, df.stddev_call_value_usd, color = pal[1], linewidth = .1)
ax1[0].set_ylabel('StDev. transfer amount, USD', color = pal[1])
ax1[0].ticklabel_format(axis="y", style="sci", scilimits=(0,0))

ax2 = ax1[0].twinx()
ax2.plot(df.date, df.avg_call_value_usd, color = pal[0], linewidth = .1)
ax2.set_ylabel('Avg. transfer amount, USD (log scale)', color = pal[0])
ax2.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
ax2.set_yscale('log')
ax2.grid(False)

ax1[1].scatter(df.close, df.avg_call_value_usd, color = pal[0], s = .1)
# ax1[1].ticklabel_format(axis="y", style="sci", scilimits=(0,0))
ax1[1].set_xlabel('Price, USD (log scale)')
ax1[1].set_ylabel('Avg. transfer amount, USD (log scale)')
ax1[1].set_yscale('log')
ax1[1].set_xscale('log')

# plt.savefig("charts/call_value.png")


# %%
# '' profit/loss, USD '''

fig, ax1 = plt.subplots(2, 1, figsize=(8, 8))

ax1[0].plot(df.date, df.stddev_balance_usd, color = pal[1])
ax1[0].set_ylabel('StDev. profit/loss, USD', color = pal[1])

ax2 = ax1[0].twinx()
ax2.plot(df.date, df.avg_balance_usd, color = pal[0])
ax2.set_ylabel('Avg. profit/loss, USD', color = pal[0])
ax2.grid(False)

ax1[1].scatter(df.close, df.stddev_balance_usd, color = pal[0], s = .1)
ax1[1].set_xlabel('Price, USD')
ax1[1].set_ylabel('StDev. profit/loss, USD')
# ax1[1].set_yscale('log')
# ax1[1].set_xscale('log')

# plt.savefig("charts/USD_profitloss.png")


# %%
# '' exchange flows '''

fig, ax1 = plt.subplots(2, 1, figsize=(8, 8))

ax1[0].plot(df.date, df.to_exchange, color = pal[0], linewidth = .3)
ax1[0].ticklabel_format(axis="y", style="sci", scilimits=(0,0))
ax1[0].set_ylabel('Exchanges inflows, ETH')

ax1[1].plot(df.date, df.net_exchange, color = pal[0], linewidth = .3)
ax1[1].ticklabel_format(axis="y", style="sci", scilimits=(0,0))
ax1[1].set_ylabel('Exchanges net-flows, ETH')

# plt.savefig("charts/exchanges_flows.png")

# %%
# '' gini coefficient '''


fig, ax1 = plt.subplots(2, 1, figsize=(8, 8))

ax1[0].plot(df.date, df.gini, color = pal[0], linewidth = .8)
ax1[0].set_ylabel('Gini coefficient')

ax1[1].scatter(df.close, df.gini, color = pal[0], s = .1)
ax1[1].set_xlabel('Price, USD')
ax1[1].set_ylabel('Gini coefficient')
ax1[1].set_xscale('log')
# ax1[1].set_yscale('log')

plt.savefig("charts/gini.png")


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

# plt.savefig("charts/degrees_power_law.pdf")


# %%
# '' avg_degree '''

fig, ax1 = plt.subplots(figsize=(8, 4))

ax1.plot(df.date, df.avg_outdegree, color = pal[0], linewidth = .1)#, marker = '.', markersize = .1, linewidth = 0)
ax1.set_ylabel('Avg. degree')
ax1.set_ylim([0.5, 2.5])

plt.savefig("charts/avg_degree.png")


# %%
# '' Std. indegree '''

fig, ax1 = plt.subplots(2, 1, figsize=(8, 8))

ax1[0].plot(df.date, df.stddev_indegree, color = pal[0], linewidth = .1)
ax1[0].set_ylabel('StDev. indegree')
ax1[0].set_ylim([0, 100])

ax1[1].plot(df.date, df.stddev_outdegree, color = pal[0], linewidth = .1)
ax1[1].set_ylabel('StDev. outdegree')
ax1[1].set_ylim([0, 100])

# plt.savefig("charts/std_degree.png")

# %%
import os
from google.cloud.bigquery.client import Client
import pandas as pd

# %%
# set Google BigQuery client
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'C:\\Users\\Francesco\\fiery-rarity-322109-6ba6fa8a811c.json'
bq_client = Client()

#%% [markdown]
## Market data

# %%
# run query
market_df = bq_client.query('''
    SELECT  TIMESTAMP_TRUNC(TIMESTAMP_SECONDS(time), HOUR) as date
    , close, high, low, volumeto as volume
    FROM `fiery-rarity-322109.ethereum.eth_usd_min`
    where TIMESTAMP_SECONDS(time) = TIMESTAMP_TRUNC(TIMESTAMP_SECONDS(time), HOUR)
    and DATE(TIMESTAMP_SECONDS(time)) between '2017-01-01' and '2021-07-30'
    order by date
''').to_dataframe()

#%% [markdown]
## Degree distribution

# %%
# run query
degree_df = bq_client.query('''
    with outdegree_ as (
        SELECT TIMESTAMP_TRUNC(block_timestamp, HOUR) as date
        , from_address as address
        , count(*) as outdegree
        FROM `fiery-rarity-322109.ethereum.traces_new`
        where date(block_timestamp) >= '2017-01-01' and from_address is not null
        GROUP BY date, address
    )
    , indegree_ as (
        SELECT TIMESTAMP_TRUNC(block_timestamp, HOUR) as date
        , to_address as address
        , count(*) as indegree
        FROM `fiery-rarity-322109.ethereum.traces_new`
        where date(block_timestamp) >= '2017-01-01' and to_address is not null
        GROUP BY date, address
    )
    select ifnull(t1.date, t2.date) as date
    , avg(ifnull(outdegree,0)) as avg_outdegree
    --, avg(ifnull(indegree,0)) as avg_indegree
    , stddev(ifnull(outdegree,0)) as stddev_outdegree
    , stddev(ifnull(indegree,0)) as stddev_indegree
    from outdegree_ t1
    full outer join indegree_ t2 using (address, date)
    group by date
    order by date
''').to_dataframe()

#%% [markdown]
## Exchanges flows

# %%
# run query
exchange_df = bq_client.query('''
    select TIMESTAMP_TRUNC(t1.block_timestamp, HOUR) as date
    , sum(case when t2.address is not null and t3.address is null then t1.value / 1e18 else 0 end) as from_exchange
    , sum(case when t3.address is not null and t2.address is null then t1.value / 1e18 else 0 end) as to_exchange
    , sum(case when t2.address is not null and t3.address is null then t1.value * t1.price_usd / 1e18 else 0 end) as from_exchange_usd
    , sum(case when t3.address is not null and t2.address is null then t1.value * t1.price_usd / 1e18 else 0 end) as to_exchange_usd
    from `fiery-rarity-322109.ethereum.traces_new` t1
    left join `fiery-rarity-322109.ethereum.exchanges_monthly` t2 on DATE_TRUNC(date(t1.block_timestamp), MONTH) = t2.date_month and t1.from_address = t2.address
    left join `fiery-rarity-322109.ethereum.exchanges_monthly` t3 on DATE_TRUNC(date(t1.block_timestamp), MONTH) = t3.date_month and t1.to_address = t3.address
    where date(t1.block_timestamp) >= '2017-01-01'
    group by date
    order by date
''').to_dataframe()

#%% [markdown]
## Transferred amount

# %%
# run query
call_df = bq_client.query('''
    select TIMESTAMP_TRUNC(block_timestamp, HOUR) as date
    , avg(value * price_usd / 1e18) as avg_call_value_usd
    , stddev(value * price_usd / 1e18) as stddev_call_value_usd
    , count(*) nr_call
    FROM `fiery-rarity-322109.ethereum.traces_new`
    where date(block_timestamp) >= '2017-01-01' and trace_type = 'call'
    group by date
    order by date
''').to_dataframe()

#%% [markdown]
## Gas fees

# %%
# run query
fee_df = bq_client.query('''
    SELECT TIMESTAMP_TRUNC(t1.block_timestamp, HOUR) as date
    , avg(cast(receipt_gas_used as numeric) * cast(gas_price as numeric) / 1e18 * close) as avg_fee_usd
    , stddev(cast(receipt_gas_used as numeric) * cast(gas_price as numeric) / 1e18 * close) as stddev_fee_usd
    FROM `bigquery-public-data.crypto_ethereum.transactions` t1
    left join `fiery-rarity-322109.ethereum.eth_usd_min` t2 on DATETIME_TRUNC(t1.block_timestamp, MINUTE) = TIMESTAMP_SECONDS(t2.time)
    where date(t1.block_timestamp) between '2017-01-01' and '2021-07-30'
    group by date
    order by date
''').to_dataframe()

#%% [markdown]
## Gini coefficient, addresses count and USD balance avg and stddvt

# %%
# run query
features_df = bq_client.query('''
    SELECT *
    FROM `fiery-rarity-322109.ethereum.features`
    where date(date) >= '2017-01-01'
    order by date
''').to_dataframe()

#%% [markdown]
## New adresses

# %%
# run query
address_df = bq_client.query('''
    with addresses as (
        select from_address as address, block_timestamp
        from `fiery-rarity-322109.ethereum.traces_new`
        where from_address is not null and value > 0
        union distinct
        select to_address as address, block_timestamp
        from `fiery-rarity-322109.ethereum.traces_new`
        where to_address is not null and value > 0
    )
    , new_addresses as (
        select address, min(TIMESTAMP_TRUNC(block_timestamp, HOUR)) date
        FROM addresses
        group by address
    )
    , new_addresses_with_gaps as (
        select date
        , count(*) nr_addresses
        FROM new_addresses
        group by date
    )
    , calendar as (
        select date
        from unnest(GENERATE_TIMESTAMP_ARRAY('2015-07-30', '2021-07-31', INTERVAL 1 HOUR)) as date
        where date(date) <= '2021-07-30'
    )
    select calendar.date as date
    , ifnull(nr_addresses, 0) as nr_addresses
    from new_addresses_with_gaps
    right join calendar on new_addresses_with_gaps.date = calendar.date
    where date(calendar.date) >= '2017-01-01'
    order by calendar.date
''').to_dataframe()

# %%
# market_df.to_csv('data/ETH_market_hour.csv', index=False)
# degree_df.to_csv('data/ETH_degree_hour.csv', index=False)
# exchange_df.to_csv('data/ETH_exchange_hour.csv', index=False)
# call_df.to_csv('data/ETH_call_hour.csv', index=False)
# fee_df.to_csv('data/ETH_fee_hour.csv', index=False)
# features_df.to_csv('data/ETH_gini_balance_usd_address_hour.csv', index=False)
# address_df.to_csv('data/ETH_address_hour.csv', index=False)

# %% [markdown]
# Merge all

# %%
# assert market_df.shape[0] == degree_df.shape[0] == exchange_df.shape[0] == call_df.shape[0] == fee_df.shape[0] == gini_df.shape[0] == balance_usd_df.shape[0] == address_df

dfs = [market_df, degree_df, exchange_df, call_df, fee_df, features_df, address_df]
dfs = [df.set_index('date') for df in dfs]
df = dfs[0].join(dfs[1:])

# %%
assert df.shape[0] == degree_df.shape[0]

# df.to_csv('data/ETH_features_hour.csv')

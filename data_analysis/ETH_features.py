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
    where TIMESTAMP_SECONDS(time) = TIMESTAMP_TRUNC(TIMESTAMP_SECONDS(time), DAY) and DATE(TIMESTAMP_SECONDS(time)) >= '2017-01-01'
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
        where date >= '2017-01-01' and from_address is not null
        GROUP BY date, address
    )
    , indegree_ as (
        SELECT TIMESTAMP_TRUNC(block_timestamp, HOUR) as date
        , to_address as address
        , count(*) as indegree
        FROM `fiery-rarity-322109.ethereum.traces_new`
        where date >= '2017-01-01' and to_address is not null
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
    , sum(case when t2.address is not null and t3.address is null then t1.value * t1.price_usd / 1e18 else 0 end) as from_exchange_usd
    , sum(case when t3.address is not null and t2.address is null then t1.value * t1.price_usd / 1e18 else 0 end) as to_exchange_usd
    from `fiery-rarity-322109.ethereum.traces_new` t1
    left join `fiery-rarity-322109.ethereum.exchanges_monthly1` t2 on DATE_TRUNC(date(t1.block_timestamp), MONTH) = t2.date_month and t1.from_address = t2.address
    left join `fiery-rarity-322109.ethereum.exchanges_monthly1` t3 on DATE_TRUNC(date(t1.block_timestamp), MONTH) = t3.date_month and t1.to_address = t3.address
    where date >= '2017-01-01'
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
## Gini coefficient

# %%
# run query
gini_df = bq_client.query('''
    SELECT date, gini
    FROM `fiery-rarity-322109.ethereum.features`
    where date(date) >= '2017-01-01'
    order by date
''').to_dataframe()

#%% [markdown]
## USD profit/loss

# %%
# run query
balance_usd_df = bq_client.query('''
    with 
    double_entry_book as (
        -- debits
        select to_address as address, value * price_usd as value_usd, block_timestamp
        from `fiery-rarity-322109.ethereum.traces_new`
        where to_address is not null
        union all
        -- credits
        select from_address as address, -value * price_usd as value_usd, block_timestamp
        from `fiery-rarity-322109.ethereum.traces_new`
        where from_address is not null
        union all
        -- transaction fees debits
        select miner as address
        , sum(cast(receipt_gas_used as numeric) * cast(gas_price as numeric) * close) as value_usd
        , block_timestamp
        from `bigquery-public-data.crypto_ethereum.transactions` t1
        left join `fiery-rarity-322109.ethereum.eth_usd_min` t2 on DATETIME_TRUNC(t1.block_timestamp, MINUTE) = TIMESTAMP_SECONDS(t2.time)
        join `bigquery-public-data.crypto_ethereum.blocks` as blocks on blocks.number = t1.block_number
        where date(t1.block_timestamp) <= '2021-07-30'
        group by blocks.miner, block_timestamp
        union all
        -- transaction fees credits
        select from_address as address
        , -(cast(receipt_gas_used as numeric) * cast(gas_price as numeric) * close) as value_usd
        , block_timestamp
        from `bigquery-public-data.crypto_ethereum.transactions` t1
        left join `fiery-rarity-322109.ethereum.eth_usd_min` t2 on DATETIME_TRUNC(t1.block_timestamp, MINUTE) = TIMESTAMP_SECONDS(t2.time)
        where date(block_timestamp) <= '2021-07-30' and from_address is not null
    )
    ,double_entry_book_by_date as (
        select 
            TIMESTAMP_TRUNC(block_timestamp, HOUR) as date, 
            address,
            sum(value_usd) as value_usd
        from double_entry_book
        where address is not null
        group by address, date
    )
    ,balances_with_gaps as (
        select 
            address, 
            date,
            sum(value_usd) over (partition by address order by date) as balance_usd,
            lead(date, 1, CAST('2021-07-30' AS TIMESTAMP)) over (partition by address order by date) as next_date
            from double_entry_book_by_date
    )
    ,calendar as (
        select date from unnest(GENERATE_TIMESTAMP_ARRAY('2015-07-30', '2021-07-30', INTERVAL 1 HOUR)) as date 
    )
    ,balances as (
        select address, calendar.date, balance_usd
        from balances_with_gaps
        join calendar on balances_with_gaps.date <= calendar.date and calendar.date < balances_with_gaps.next_date
    )
    select date
    , avg(balance_usd / 1e18) avg_balance_usd
    , stddev(balance_usd / 1e18) stddev_balance_usd
    from balances
    where date(date) >= '2017-01-01'
    group by date
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
        where from_address is not null
        union distinct
        select to_address as address, block_timestamp
        from `fiery-rarity-322109.ethereum.traces_new`
        where to_address is not null
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
        group by date, contract_type
    )
    , calendar as (
        select date from unnest(GENERATE_TIMESTAMP_ARRAY('2015-07-30', '2021-07-30', INTERVAL 1 HOUR)) as date 
    )
    select calendar.date as date
    , ifnull(nr_addresses, 0) as nr_addresses
    from new_addresses_with_gaps
    right join calendar on new_addresses_with_gaps.date = calendar.date
    where date(calendar.date) >= '2017-01-01'
    order by calendar.date
''').to_dataframe()

# %% [markdown]
# Merge all

# %%

market_df
degree_df
exchange_df
call_df
fee_df
gini_df
balance_usd_df
address_df

df = exchange.merge(market_data, on = 'date', how = 'left')

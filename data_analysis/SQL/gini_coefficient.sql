CREATE OR REPLACE TABLE
  `fiery-rarity-322109.ethereum.features` (
  date TIMESTAMP NOT NULL,
  address_count INT,
  gini FLOAT64,
  avg_balance_usd FLOAT64,
  stddev_balance_usd FLOAT64
  )
PARTITION BY DATE(date)
OPTIONS (
    description = "Ethereum hourly address count, Gini coefficient and USD balance mean and standard deviation features."
)
AS
with 
double_entry_book as (
    -- debits
    select to_address as address, value as value, value * price_usd as value_usd, block_timestamp
    from `fiery-rarity-322109.ethereum.traces_new`
    where to_address is not null
    union all
    -- credits
    select from_address as address, -value as value, -value * price_usd as value_usd, block_timestamp
    from `fiery-rarity-322109.ethereum.traces_new`
    where from_address is not null
    union all
    -- transaction fees debits
    select miner as address
    , sum(cast(receipt_gas_used as numeric) * cast(gas_price as numeric)) as value
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
    , -(cast(receipt_gas_used as numeric) * cast(gas_price as numeric)) as value
    , -(cast(receipt_gas_used as numeric) * cast(gas_price as numeric) * close) as value_usd
    , block_timestamp
    from `bigquery-public-data.crypto_ethereum.transactions` t1
    left join `fiery-rarity-322109.ethereum.eth_usd_min` t2 on DATETIME_TRUNC(t1.block_timestamp, MINUTE) = TIMESTAMP_SECONDS(t2.time)
    where date(block_timestamp) <= '2021-07-30'
)
,double_entry_book_by_date as (
    select 
        TIMESTAMP_TRUNC(block_timestamp, HOUR) as date, 
        address, 
        sum(value) as value,
        sum(value_usd) as value_usd
    from double_entry_book
    group by address, date
)
,balances_with_gaps as (
    select 
        address, 
        date,
        sum(value) over (partition by address order by date) as balance,
        sum(value_usd) over (partition by address order by date) as balance_usd,
        lead(date, 1, CAST('2021-07-30' AS TIMESTAMP)) over (partition by address order by date) as next_date
        from double_entry_book_by_date
)
,calendar as (
    select date from unnest(GENERATE_TIMESTAMP_ARRAY('2015-07-30', '2021-07-30', INTERVAL 1 HOUR)) as date 
)
,balances as (
    select address, calendar.date, balance, balance_usd
    from balances_with_gaps
    join calendar on balances_with_gaps.date <= calendar.date and calendar.date < balances_with_gaps.next_date
)
,supply as (
    select
        date,
        sum(balance) as supply_value
    from balances
    group by date
)
,ranked_balances as (
    select 
        balances.date,
        balance,
        balance_usd,
        row_number() over (partition by balances.date order by balance desc) as rank
    from balances
    join supply on balances.date = supply.date
    where safe_divide(balance, supply_value) >= 0.0001
    ORDER BY safe_divide(balance, supply_value) DESC
)
select date
, count(*) address_count
, 1 - 2 * sum((balance * (rank - 1) + balance / 2)) / count(*) / sum(balance) as gini
, avg(balance_usd / 1e18) avg_balance_usd
, stddev(balance_usd / 1e18) stddev_balance_usd
from ranked_balances
group by date
CREATE TABLE
  `fiery-rarity-322109.ethereum.traces_new` (
  block_timestamp TIMESTAMP NOT NULL,
  trace_type STRING NOT NULL,
  from_address STRING,
  to_address STRING,
  value NUMERIC,
  price_usd FLOAT64,
  gas INT64,
  gas_used INT64,
  block_number INT64 NOT NULL,
  block_hash STRING NOT NULL,
  transaction_hash STRING,
  transaction_index INT64,
  from_contract_type INT64,
  to_contract_type INT64,
  from_month_mined_blocks INT64,
  to_month_mined_blocks INT64,
  supply_eth INT64
  )
PARTITION BY
  DATE(block_timestamp)
OPTIONS (
    description = "Ethereum traces with receipt status success and positive value until 2021-07-30."
)
AS
WITH
ether_emitted_by_date  as (
  select block_timestamp
  , sum(value) as value_eth
  from `bigquery-public-data.crypto_ethereum.traces`
  where trace_type in ('genesis', 'reward') and status = 1 and date(block_timestamp) <= '2021-07-30'
  group by block_timestamp
)
, supply as (
  select case when date(block_timestamp) < '2015-07-30' then CAST('2015-07-30' AS TIMESTAMP) else block_timestamp end as block_timestamp
  , cast(sum(value_eth) OVER (ORDER BY block_timestamp) / power(10, 18) as INT64) AS supply_eth
  from ether_emitted_by_date
)
, miners as (
    select miner, FORMAT_DATE('%m%Y', DATE(timestamp)) date_month, count(distinct number) count_blocks
    from `bigquery-public-data.crypto_ethereum.blocks`
    group by miner, date_month
)
, contracts as (
    select address
    , case when is_erc20 is true then 1 when is_erc721 is true then 2 else 3 end contract_type
    , min(block_number) min_block_number
    from `bigquery-public-data.crypto_ethereum.contracts`
    group by address, contract_type
)
select case when date(t1.block_timestamp) < '2015-07-30' then CAST('2015-07-30' AS TIMESTAMP) else t1.block_timestamp end as block_timestamp
, trace_type, from_address, to_address, value
, case when t2.close is null then 3 else t2.close end as price_usd
, gas, gas_used, block_number, block_hash, transaction_hash, transaction_index
, t3.contract_type as from_contract_type
, t4.contract_type as to_contract_type
, t5.count_blocks as from_month_mined_blocks
, t6.count_blocks as to_month_mined_blocks
, t7.supply_eth
FROM `bigquery-public-data.crypto_ethereum.traces` t1
left join `fiery-rarity-322109.ethereum.eth_usd_min` t2 on DATETIME_TRUNC(t1.block_timestamp, MINUTE) = TIMESTAMP_SECONDS(t2.time)
left join contracts t3 on t1.from_address = t3.address and t1.block_number >= t3.min_block_number
left join contracts t4 on t1.to_address = t4.address and t1.block_number >= t4.min_block_number
left join miners t5 on t1.from_address = t5.miner and FORMAT_DATE('%m%Y', DATE(t1.block_timestamp)) = t5.date_month
left join miners t6 on t1.to_address = t6.miner and FORMAT_DATE('%m%Y', DATE(t1.block_timestamp)) = t6.date_month
left join supply t7 on t1.block_timestamp = t7.block_timestamp
WHERE t1.status = 1 and date(t1.block_timestamp) <= '2021-07-30' and (t1.call_type not in ('delegatecall', 'callcode', 'staticcall') or t1.call_type is null) and t1.value > 0
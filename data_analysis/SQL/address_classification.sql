CREATE OR REPLACE TABLE
  `fiery-rarity-322109.ethereum.address_classification` (
  address STRING NOT NULL,
  date_month DATE,
  avg_count_trace_address_to FLOAT64,
  avg_count_trace_contract_to FLOAT64,
  avg_count_address_to FLOAT64,
  avg_count_contract_to FLOAT64,
  avg_sum_value_address_to FLOAT64,
  avg_sum_value_contract_to FLOAT64,
  avg_count_token_address_to FLOAT64,
  nr_days_to INT,
  avg_count_trace_address_from FLOAT64,
  avg_count_trace_contract_from FLOAT64,
  avg_count_address_from FLOAT64,
  avg_count_contract_from FLOAT64,
  avg_sum_value_address_from FLOAT64,
  avg_sum_value_contract_from FLOAT64,
  avg_count_token_address_from FLOAT64,
  nr_days_from INT,
  avg_balance FLOAT64,
  rank INT
  )
--PARTITION BY date_month
OPTIONS (
    description = "Features by month date for Ethereum addresses that are neither contracts nor miners. This data is to be used as input for clustering algorithms."
)
AS
WITH
double_entry_book as (
    -- credits
    select from_address as address, -value as value, block_timestamp -- / 1e18 * price_usd
    from `fiery-rarity-322109.ethereum.traces_new`
    where from_address is not null and from_contract_type is null and (from_month_mined_blocks is null) --or from_month_mined_blocks < 20)
    union all
    -- debits
    select to_address as address, value as value, block_timestamp
    from `fiery-rarity-322109.ethereum.traces_new`
    where to_address is not null and to_contract_type is null and (to_month_mined_blocks is null) --or to_month_mined_blocks < 20)
    union all
    -- transaction fees credits
    select from_address as address, -(cast(receipt_gas_used as numeric) * cast(gas_price as numeric)) as value, block_timestamp -- / 1e18 * close
    from `bigquery-public-data.crypto_ethereum.transactions` t1
    --left join `fiery-rarity-322109.ethereum.eth_usd_min` t2 on DATETIME_TRUNC(t1.block_timestamp, MINUTE) = TIMESTAMP_SECONDS(t2.time)
    where date(block_timestamp) <= '2021-07-30'
)
, double_entry_book_by_date as (
    select 
        date(block_timestamp) as date, 
        address, 
        sum(value) as value
    from double_entry_book
    group by address, date
)
, daily_balances_with_gaps as (
    select 
        address, 
        date,
        sum(value) over (partition by address order by date) as balance,
        lead(date, 1, '2021-07-30') over (partition by address order by date) as next_date
        from double_entry_book_by_date
)
, calendar as (
    select date from unnest(generate_date_array('2015-08-01', '2021-07-30')) as date
)
, nr_days as (
    select DATE_TRUNC(date, MONTH) as date_month, count(*) nr_days
    from calendar
    group by date_month  
)
, daily_balances as (
    select address, calendar.date, balance
    from daily_balances_with_gaps
    join calendar on daily_balances_with_gaps.date <= calendar.date and calendar.date < daily_balances_with_gaps.next_date
)
, daily_from as (
    SELECT DATE(t1.block_timestamp) date
    , t1.from_address as address
    , sum(case when to_contract_type is not null then 0 else 1 end) count_trace_address
    , sum(case when to_contract_type is null then 0 else 1 end) count_trace_contract
    , count(distinct case when to_contract_type is not null then null else t1.to_address end) as count_address
    , count(distinct case when to_contract_type is null then null else t1.to_address end) as count_contract
    , sum(case when to_contract_type is not null then null else t1.value / 1e18 * price_usd end) as sum_value_address
    , sum(case when to_contract_type is null then null else t1.value / 1e18 * price_usd end) as sum_value_contract
    , count(distinct token_address) count_token_address
    FROM `fiery-rarity-322109.ethereum.traces_new` t1
    left join `bigquery-public-data.crypto_ethereum.token_transfers` t2 using(from_address, block_timestamp)
    WHERE t1.from_address is not null and from_contract_type is null and (from_month_mined_blocks is null) --or from_month_mined_blocks < 20)
    group by date, address
),
daily_to as (
    SELECT DATE(t1.block_timestamp) date
    , t1.to_address as address
    , sum(case when from_contract_type is not null then 0 else 1 end) count_trace_address
    , sum(case when from_contract_type is null then 0 else 1 end) count_trace_contract
    , count(distinct case when from_contract_type is not null then null else t1.from_address end) as count_address
    , count(distinct case when from_contract_type is null then null else t1.from_address end) as count_contract
    , sum(case when from_contract_type is not null then null else t1.value / 1e18 * price_usd end) as sum_value_address
    , sum(case when from_contract_type is null then null else t1.value / 1e18 * price_usd end) as sum_value_contract
    , count(distinct token_address) count_token_address
    FROM `fiery-rarity-322109.ethereum.traces_new` t1
    left join `bigquery-public-data.crypto_ethereum.token_transfers` t2 using(to_address, block_timestamp)
    WHERE to_address is not null and to_contract_type is null and (to_month_mined_blocks is null) --or to_month_mined_blocks < 20)
    group by date, address
)
, address_classification as (
    select
    ifnull(t1.address, t2.address) as address
    , ifnull(DATE_TRUNC(t1.date, MONTH), DATE_TRUNC(t2.date, MONTH)) as date_month --cast(ifnull(FORMAT_DATE('%Y%m', t1.date), FORMAT_DATE('%Y%m', t2.date)) as INT)

    , ifnull(avg(t1.count_trace_address), 0) as avg_count_trace_address_to
    , ifnull(avg(t1.count_trace_contract), 0) as avg_count_trace_contract_to
    , ifnull(avg(t1.count_address), 0) as avg_count_address_to
    , ifnull(avg(t1.count_contract), 0) as avg_count_contract_to
    , ifnull(avg(t1.sum_value_address), 0) as avg_sum_value_address_to
    , ifnull(avg(t1.sum_value_contract), 0) as avg_sum_value_contract_to
    , ifnull(avg(t1.count_token_address), 0) as avg_count_token_address_to
    , ifnull(count(distinct t1.date), 0) as nr_days_to

    , ifnull(avg(t2.count_trace_address), 0) as avg_count_trace_address_from
    , ifnull(avg(t2.count_trace_contract), 0) as avg_count_trace_contract_from
    , ifnull(avg(t2.count_address), 0) as avg_count_address_from
    , ifnull(avg(t2.count_contract), 0) as avg_count_contract_from
    , ifnull(avg(t2.sum_value_address), 0) as avg_sum_value_address_from
    , ifnull(avg(t2.sum_value_contract), 0) as avg_sum_value_contract_from
    , ifnull(avg(t2.count_token_address), 0) as avg_count_token_address_from
    , ifnull(count(distinct t2.date), 0) as nr_days_from

    , avg(balance) as avg_balance

    from daily_from t1
    left outer join daily_to t2 using (address, date)
    left join daily_balances t3 on ifnull(t1.address, t2.address) = t3.address and ifnull(t1.date, t2.date) = t3.date
    group by address, date_month
)
, address_rank as (
    SELECT t1.*
    , ROW_NUMBER() OVER(PARTITION BY date_month ORDER BY avg_balance DESC) AS rank
    from address_classification t1
)
SELECT t1.*
FROM address_rank t1
--inner join nr_days t2 on t1.date_month = t2.date_month and t1.nr_days_to = t2.nr_days
--where t1.nr_days_to = t1.nr_days_from
where rank >= 10000
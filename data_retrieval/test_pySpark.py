#!/usr/bin/python
"""BigQuery I/O PySpark example."""
from pyspark.sql import SparkSession

spark = SparkSession.builder \
  .master('yarn') \
  .appName('spark-bigquery-crypto') \
  .config('spark.jars', 'gs://spark-lib/bigquery/spark-bigquery-latest_2.12.jar') \
  .getOrCreate()

# Use the Cloud Storage bucket for temporary BigQuery export data used
# by the connector.
bucket = "dataproc-temp-us-central1-397709471406-ybhgchsn"
spark.conf.set('temporaryGcsBucket', bucket)

# Load data from BigQuery.
words = spark.read.format('bigquery') \
  .option('table', 'bigquery-public-data:crypto_ethereum.transactions') \
  .load()
words.createOrReplaceTempView('eth_transactions')

# Perform word count.
word_count = spark.sql(
    '''SELECT * FROM eth_transactions WHERE DATE(block_timestamp) between "2020-01-01" and "2020-01-31"''')
word_count.show()
word_count.printSchema()

# Saving the data to BigQuery
word_count.write.format('bigquery') \
  .option('table', 'crypto_ethereum.transactions') \
  .save()
#!/bin/bash

gcloud beta dataproc clusters create test-cluster \
    --zone us-central1-c \
    --master-machine-type n1-standard-2 \
    --master-boot-disk-size 500 \
    --num-workers 2 \
    --worker-machine-type n1-standard-2 \
    --worker-boot-disk-size 500 \
    --image-version 1.5-ubuntu18 \
    --enable-component-gateway \
    --optional-components ANACONDA,JUPYTER \
    --scopes 'https://www.googleapis.com/auth/cloud-platform' \
    --project charged-sector-315517







--jars=gs://spark-lib/bigquery/spark-bigquery-latest_2.12.jar


gcloud beta dataproc clusters create cluster-d369 \
    --enable-component-gateway \
    --region us-central1 \
    --zone us-central1-f \
    --master-machine-type n1-standard-2 \
    --master-boot-disk-size 500 \
    --num-workers 2 \
    --worker-machine-type n1-standard-2 \
    --worker-boot-disk-size 500 \
    --image-version 1.5-ubuntu18 \
    --optional-components ANACONDA,JUPYTER \
    --scopes 'https://www.googleapis.com/auth/cloud-platform' \
    --project charged-sector-315517




gcloud beta dataproc clusters create cluster-c311
    --enable-component-gateway
    --region us-central1
    --zone us-central1-c
    --master-machine-type n1-standard-2
    --master-boot-disk-size 500
    --num-workers 2
    --worker-machine-type n1-standard-2
    --worker-boot-disk-size 500
    --image-version 1.5-ubuntu18
    --optional-components ANACONDA,JUPYTER
    --max-idle 7200s
    --scopes 'https://www.googleapis.com/auth/cloud-platform'
    --dataproc-metastore projects/charged-sector-315517/locations/us-central1/services/service-9651
    --project charged-sector-315517


pyspark --packages graphframes:graphframes:0.8.1-spark2.4-s_2.11
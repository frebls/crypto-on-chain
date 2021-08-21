#!/bin/bash

gcloud beta dataproc clusters create cluster-c311sv
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
    --project fiery-rarity-322109
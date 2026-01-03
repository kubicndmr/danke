#!/bin/bash

N=5

for ((i=1; i<=N; i++)); do
    python spr.py --real_dataset_size 0 --syn_dataset_size 50 --real_data_path to_transfer/PoCaP-large-v3 --syn_data_path to_transfer/SynPoCaP
done
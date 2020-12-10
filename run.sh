#!/bin/bash

python train.py \
	--data_dir "data" \
	--sample_path "results" \
	--model_path "model" \
	--dataset "MNIST" \
	--sample_step 500 \
	--lr_D 1e-4 \
	--lr_G 2e-5 \

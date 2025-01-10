#!/bin/bash

# 3 different seeds to run the model with
train=false
test=true
seeds=(22 32 42)
datasets=('conll2003' 'mit_movies' 'ace2005')
k_shots=(50 20 10 5)
augmentation='oada'  # baseline, lsp
# run the model with each dataset and seed
for dataset in "${datasets[@]}"; do
  for k_shot in "${k_shots[@]}"; do
    for seed in "${seeds[@]}"; do
      echo "Running run.py with dataset: $dataset, k_shot: $k_shot, seed: $seed, augmentation: $augmentation"
      # using CLI from hydra to run the model with different datasets and seeds
      # https://hydra.cc/docs/tutorials/basic/your_first_app/config_file/
      python run.py train=$train test=$test dataset="$dataset" k_shot="$k_shot" training_args="${k_shot}_shot" seed="$seed" augmentation=$augmentation
    done
  done
done
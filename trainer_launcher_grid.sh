#!/bin/bash

# Define the batch sizes and learning rates as arrays
BATCH_SIZES=(8 12 16 24 32 64)
LEARNING_RATES=(1e-4 5e-4 1e-3 5e-3 1e-2 5e-2)

# Loop over each batch size
for BS in "${BATCH_SIZES[@]}"; do
  # Loop over each learning rate
  for LR in "${LEARNING_RATES[@]}"; do
    # Call train.py with the current batch size and learning rate
    echo "Training with Batch Size = $BS and Learning Rate = $LR"
    python trainMobileVit.py --batch-size $BS --learning-rate $LR
  done
done

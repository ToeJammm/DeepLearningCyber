#!/bin/bash

echo "Starting training..."
python train.py --model lenet --dataset mnist
echo "Training complete!"
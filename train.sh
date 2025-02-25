#!/bin/bash

echo "Starting training..."
python train.py --model resnet18 --dataset cifar10
echo "Training complete!"
#!/bin/bash

echo "Starting training..."
python train.py --model vgg16 --dataset cifar10
echo "Training complete!"
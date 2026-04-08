"""
airbench_v1_sgd.py
Version 1: ResNet + Standard SGD
- Standard PyTorch DataLoader (CPU-based)
- No JIT compilation
- Float32 precision
"""

import os
import sys
import uuid
from math import ceil
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader

# Import shared utilities
from airbench_utils import (
    CifarNet, ResNet18, CIFAR_MEAN, CIFAR_STD,
    print_columns, print_training_details, logging_columns_list
)

def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.cuda(), labels.cuda()
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return correct / total

def main(run, model):
    
    # Hyperparameters
    batch_size = 512 # Standard batch size for V1
    lr = 0.1
    momentum = 0.9
    weight_decay = 5e-4
    epochs = 20 # Reduced epochs for demonstration, or keep similar to original? 
    # Original runs for ~8 epochs of 2000 batch size.
    # Let's try to match the total training volume roughly.
    
    # Data Preparation (Standard DataLoader)
    transform_train = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])

    transform_test = T.Compose([
        T.ToTensor(),
        T.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])

    trainset = torchvision.datasets.CIFAR10(root='./cifar10', train=True, download=True, transform=transform_train)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)

    testset = torchvision.datasets.CIFAR10(root='./cifar10', train=False, download=True, transform=transform_test)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    # Optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Timing
    start_time = time.time()

    # Training Loop
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.cuda(), targets.cuda()
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = F.cross_entropy(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        scheduler.step()
        
        # Evaluation
        train_acc = correct / total
        val_acc = evaluate(model, testloader)
        
        time_seconds = time.time() - start_time
        
        # Logging
        print_training_details({
            "run": run if epoch == 0 else "",
            "epoch": epoch + 1,
            "train_acc": train_acc,
            "val_acc": val_acc,
            "tta_val_acc": "", # No TTA in V1
            "time_seconds": time_seconds
        }, is_final_entry=(epoch == epochs - 1))

    return val_acc

if __name__ == "__main__":
    # Initialize model in FP32
    model = ResNet18(dtype=torch.float32).cuda()
    
    print_columns(logging_columns_list, is_head=True)
    acc = main(0, model)
    print(f"Final Accuracy: {acc:.4f}")

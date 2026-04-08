"""
airbench_v3_mixed_precision.py
Version 3: Mixed Precision
- Custom CifarLoader (GPU-based)
- JIT Compilation (@torch.compile)
- Standard SGD
- Float16 precision (Mixed Precision)
"""

import os
import sys
import uuid
from math import ceil

import torch
import torch.nn.functional as F

# Import shared utilities
from airbench_utils import (
    CifarLoader, CifarNet, 
    print_columns, print_training_details, evaluate, logging_columns_list
)

torch.backends.cudnn.benchmark = True

def main(run, model):

    batch_size = 2000
    lr = 0.5
    momentum = 0.9
    wd = 5e-4
    epochs = 20
    
    # Use CifarLoader with FP16
    test_loader = CifarLoader("cifar10", train=False, batch_size=2000, dtype=torch.float16)
    train_loader = CifarLoader("cifar10", train=True, batch_size=batch_size, aug=dict(flip=True, translate=2), dtype=torch.float16)
    
    if run == "warmup":
        train_loader.labels = torch.randint(0, 10, size=(len(train_loader.labels),), device=train_loader.labels.device)

    total_train_steps = ceil(epochs * len(train_loader))

    # Optimizer (Standard SGD)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=wd, nesterov=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_train_steps)

    # For accurately timing GPU code
    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)
    time_seconds = 0.0
    def start_timer():
        starter.record()
    def stop_timer():
        ender.record()
        torch.cuda.synchronize()
        nonlocal time_seconds
        time_seconds += 1e-3 * starter.elapsed_time(ender)

    model.reset()
    step = 0

    # Initialize the whitening layer using training images
    start_timer()
    train_images = train_loader.normalize(train_loader.images[:5000])
    model.init_whiten(train_images)
    stop_timer()

    for epoch in range(epochs):

        ####################
        #     Training     #
        ####################

        start_timer()
        model.train()
        for inputs, labels in train_loader:
            outputs = model(inputs)
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            step += 1
        stop_timer()

        ####################
        #    Evaluation    #
        ####################

        # Save the accuracy and loss from the last training batch of the epoch
        train_acc = (outputs.detach().argmax(1) == labels).float().mean().item()
        val_acc = evaluate(model, test_loader, tta_level=0)
        print_training_details(locals(), is_final_entry=False)
        run = None # Only print the run number once

    return val_acc

if __name__ == "__main__":

    # Initialize model in FP16 (default) and compile
    model = CifarNet(dtype=torch.float16).cuda().to(memory_format=torch.channels_last)
    model = torch.compile(model, mode="max-autotune")

    print_columns(logging_columns_list, is_head=True)
    main("warmup", model)
    accs = torch.tensor([main(run, model) for run in range(5)])
    print("Mean: %.4f    Std: %.4f" % (accs.mean(), accs.std()))

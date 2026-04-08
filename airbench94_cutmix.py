#############################################
#                  Setup                    #
#############################################

import os
import sys
with open(sys.argv[0]) as f:
    code = f.read()
import uuid
from math import ceil

import torch
import torch.nn.functional as F

# Import shared utilities
from airbench_utils import (
    Muon, CifarLoader, CifarNet, 
    print_columns, print_training_details, evaluate, logging_columns_list
)

torch.backends.cudnn.benchmark = True

#############################################
#                CutMix                     #
#############################################

def apply_cutmix(batch, labels, alpha=1.0):
    data, targets = batch, labels
    indices = torch.randperm(data.size(0)).to(data.device)
    shuffled_data = data[indices]
    shuffled_targets = targets[indices]

    lam = torch.distributions.beta.Beta(alpha, alpha).sample().item()
    
    image_h, image_w = data.shape[2:]
    cx = torch.distributions.uniform.Uniform(0, image_w).sample()
    cy = torch.distributions.uniform.Uniform(0, image_h).sample()
    w = image_w * torch.sqrt(torch.tensor(1 - lam))
    h = image_h * torch.sqrt(torch.tensor(1 - lam))
    x0 = int(torch.round(torch.max(cx - w / 2, torch.tensor(0.0))))
    x1 = int(torch.round(torch.min(cx + w / 2, torch.tensor(image_w))))
    y0 = int(torch.round(torch.max(cy - h / 2, torch.tensor(0.0))))
    y1 = int(torch.round(torch.min(cy + h / 2, torch.tensor(image_h))))

    data[:, :, y0:y1, x0:x1] = shuffled_data[:, :, y0:y1, x0:x1]
    
    # Adjust lambda to match pixel ratio
    lam = 1 - ((x1 - x0) * (y1 - y0) / (image_w * image_h))
    
    # Create soft labels
    # Assuming labels are indices, convert to one-hot
    num_classes = 10
    targets_one_hot = F.one_hot(targets, num_classes=num_classes).float()
    shuffled_targets_one_hot = F.one_hot(shuffled_targets, num_classes=num_classes).float()
    
    mixed_labels = lam * targets_one_hot + (1 - lam) * shuffled_targets_one_hot
    
    return data, mixed_labels

############################################
#                Training                  #
############################################

def main(run, model):

    batch_size = 2000
    bias_lr = 0.053
    head_lr = 0.67
    wd = 2e-6 * batch_size

    test_loader = CifarLoader("cifar10", train=False, batch_size=2000)
    train_loader = CifarLoader("cifar10", train=True, batch_size=batch_size, aug={})
    if run == "warmup":
        # The only purpose of the first run is to warmup the compiled model, so we can use dummy data
        train_loader.labels = torch.randint(0, 10, size=(len(train_loader.labels),), device=train_loader.labels.device)
    total_train_steps = ceil(8 * len(train_loader))
    whiten_bias_train_steps = ceil(3 * len(train_loader))

    # Create optimizers and learning rate schedulers
    filter_params = [p for p in model.parameters() if len(p.shape) == 4 and p.requires_grad]
    norm_biases = [p for n, p in model.named_parameters() if "norm" in n and p.requires_grad]
    param_configs = [dict(params=[model.whiten.bias], lr=bias_lr, weight_decay=wd/bias_lr),
                     dict(params=norm_biases,         lr=bias_lr, weight_decay=wd/bias_lr),
                     dict(params=[model.head.weight], lr=head_lr, weight_decay=wd/head_lr)]
    optimizer1 = torch.optim.SGD(param_configs, momentum=0.85, nesterov=True, fused=True)
    optimizer2 = Muon(filter_params, lr=0.24, momentum=0.6, nesterov=True)
    optimizers = [optimizer1, optimizer2]
    for opt in optimizers:
        for group in opt.param_groups:
            group["initial_lr"] = group["lr"]

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

    for epoch in range(ceil(total_train_steps / len(train_loader))):

        ####################
        #     Training     #
        ####################

        start_timer()
        model.train()
        for inputs, labels in train_loader:
            
            # 1. Apply CutMix to the batch
            inputs, labels_mixed = apply_cutmix(inputs, labels, alpha=1.0) 
            
            # 2. Forward pass with mixed inputs
            outputs = model(inputs, whiten_bias_grad=(step < whiten_bias_train_steps))
            
            # 3. Custom Loss for Soft Labels (replacing cross_entropy)
            # Loss = -sum(labels_mixed * log_softmax(outputs))
            log_softmax_outputs = F.log_softmax(outputs, dim=1)
            loss = -(labels_mixed * log_softmax_outputs).sum()
            
            loss.backward()
            for group in optimizer1.param_groups[:1]:
                group["lr"] = group["initial_lr"] * (1 - step / whiten_bias_train_steps)
            for group in optimizer1.param_groups[1:]+optimizer2.param_groups:
                group["lr"] = group["initial_lr"] * (1 - step / total_train_steps)
            for opt in optimizers:
                opt.step()
            model.zero_grad(set_to_none=True)
            step += 1
            if step >= total_train_steps:
                break
        stop_timer()

        ####################
        #    Evaluation    #
        ####################

        # Save the accuracy and loss from the last training batch of the epoch
        train_acc = (outputs.detach().argmax(1) == labels).float().mean().item()
        val_acc = evaluate(model, test_loader, tta_level=0)
        print_training_details(locals(), is_final_entry=False)
        run = None # Only print the run number once

    ####################
    #  TTA Evaluation  #
    ####################

    start_timer()
    tta_val_acc = evaluate(model, test_loader, tta_level=2)
    stop_timer()
    epoch = "eval"
    print_training_details(locals(), is_final_entry=True)

    return tta_val_acc

if __name__ == "__main__":

    # We re-use the compiled model between runs to save the non-data-dependent compilation time
    model = CifarNet().cuda().to(memory_format=torch.channels_last)
    model.compile(mode="max-autotune")

    print_columns(logging_columns_list, is_head=True)
    main("warmup", model)
    accs = torch.tensor([main(run, model) for run in range(200)])
    print("Mean: %.4f    Std: %.4f" % (accs.mean(), accs.std()))

    log_dir = os.path.join("logs", str(uuid.uuid4()))
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "log.pt")
    torch.save(dict(code=code, accs=accs), log_path)
    print(os.path.abspath(log_path))

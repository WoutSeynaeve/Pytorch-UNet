import torch
import torch.nn.functional as F
from tqdm import tqdm
import torch.nn as nn

from utils.dice_score import multiclass_dice_coeff, dice_coeff


@torch.inference_mode()
def evaluate(net, dataloader, device, amp):
    net.eval()
    num_val_batches = len(dataloader)
    dice_score = 0

    # iterate over the validation set
    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
            image, mask_true = batch['image'], batch['mask']

            # move images and labels to correct device and type
            image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            mask_true = mask_true.to(device=device, dtype=torch.long)

            # predict the mask
            mask_pred = net(image)

            if net.n_classes == 1:
                assert mask_true.min() >= 0 and mask_true.max() <= 1, 'True mask indices should be in [0, 1]'
                mask_pred = (F.sigmoid(mask_pred) > 0.5).float()
                # compute the Dice score
                dice_score += dice_coeff(mask_pred, mask_true, reduce_batch_first=False)
            else:
                assert mask_true.min() >= 0 and mask_true.max() < net.n_classes, 'True mask indices should be in [0, n_classes['
                # convert to one-hot format
                mask_true = F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float()
                mask_pred = F.one_hot(mask_pred.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
                # compute the Dice score, ignoring background
                dice_score += multiclass_dice_coeff(mask_pred[:, 1:], mask_true[:, 1:], reduce_batch_first=False)

    net.train()
    return dice_score / max(num_val_batches, 1)

@torch.inference_mode()
def evaluateWeaklySupervised2(net, dataloader, device, amp):
    net.eval()
    num_val_batches = len(dataloader)
    overlap_score = 0

    # iterate over the validation set
    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        for batch in dataloader:
            image, mask_true = batch['image'], batch['mask']
        
            """FOR THIS IMPLEMENTATION, IM ASSUMING BATCH SIZE == 1"""

            # move images and labels to correct device and type
            image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            mask_true = mask_true.to(device=device, dtype=torch.long)

            # predict the mask
            mask_pred = net(image)
            # Compute softmax probabilities
            mask_pred = F.softmax(mask_pred, dim=1)
            # Get the class predictions
            argmax_mask = mask_pred.argmax(dim=1)  # Class with the highest probability for each pixel
            max_probs, _ = mask_pred.max(dim=1)    # Maximum probability for each pixel

            # Create a mask for ambiguous pixels
            ambiguous_pixels = max_probs <= 0.3
            argmax_mask[ambiguous_pixels] = 0  # Assign ambiguous pixels to class 0

            # Convert to one-hot encoding
            mask_pred = F.one_hot(argmax_mask, net.n_classes).permute(0, 3, 1, 2).float()

            # Ensure one-hot consistency (this step might now be redundant due to the above logic)
            mask_pred = (mask_pred == mask_pred.max(dim=1, keepdim=True)[0]).float()

            # Get the class indices by taking the argmax over the class dimension (dim=1)
            long_tensor = mask_pred.argmax(dim=1).squeeze(0)  # Shape: [250, 198]
            mask_true = mask_true[0,:,:]
            if long_tensor.shape != mask_true.shape:
                raise ValueError("Tensors must have the same shape")

            # Compare elements and calculate the number of equal ones
            equal_elements = (long_tensor == mask_true).float()

            # Calculate the proportion of equal elements
            equal_proportion = equal_elements.sum() / equal_elements.numel()
            overlap_score += equal_proportion.item()
           

    net.train()
    return overlap_score / max(num_val_batches, 1)

@torch.inference_mode()
def evaluateWeaklySupervised3(net, dataloader, device, amp):
    net.eval()  # Set the model to evaluation mode
    num_val_batches = len(dataloader)
    overlap_score = 0
    num_classes = 21  # Pascal VOC
    # Initialize IoU accumulators
    iou_per_class = torch.zeros(num_classes, device=device)
    valid_classes = torch.zeros(num_classes, device=device)  # To track classes present in the dataset

    # Iterate over the validation set
    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        for batch in dataloader:
            image, mask_true = batch['image'], batch['mask']

            # Move images and labels to correct device and type
            image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            mask_true = mask_true.to(device=device, dtype=torch.long)

            # Predict the mask
            mask_pred = net(image)

            # Compute softmax probabilities
            mask_pred = F.softmax(mask_pred, dim=1)

            # Get predicted class per pixel (argmax over class dimension)
            mask_pred_class = torch.argmax(mask_pred, dim=1)

            # Flatten the masks for easier processing
            mask_pred_class = mask_pred_class.view(-1)
            mask_true = mask_true.view(-1)
         
            # Compute IoU for each class
            for cls in range(num_classes):
                # Binary masks for the current class
                pred_mask = (mask_pred_class == cls)
                true_mask = (mask_true == cls)

                # Intersection and union
                intersection = (pred_mask & true_mask).sum().float()
                union = (pred_mask | true_mask).sum().float()

                if union > 0:  # Avoid division by zero
                    iou_per_class[cls] += intersection / union
                    valid_classes[cls] += 1

    missing_classes = [cls for cls in range(num_classes) if valid_classes[cls] == 0]

    if missing_classes:
        print(f"Warning: The following classes are not present in the evaluation set: {missing_classes}")
    # Compute mean IoU, ignoring classes not present in the ground truth
    mean_iou = (iou_per_class / valid_classes.clamp(min=1)).nanmean().item()

    # Convert IoU scores to a dictionary for per-class analysis
    per_class_iou = {cls: (iou_per_class[cls] / valid_classes[cls]).item() if valid_classes[cls] > 0 else None
                     for cls in range(num_classes)}
    print(per_class_iou)

    
    net.train()

    return mean_iou
@torch.inference_mode()
def evaluateWeaklySupervised(net, dataloader, device, amp):
    net.eval()  # Set the model to evaluation mode
    num_classes = 21  # Pascal VOC

    # Initialize IoU accumulators
    iou_per_class = torch.zeros(num_classes, device=device)  # Stores cumulative IoU for each class
    valid_classes = torch.zeros(num_classes, device=device)  # Tracks the count of valid images per class

    # Iterate over the validation set
    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        valsize = 0
        for batch in dataloader:
            valsize += 1
            image, mask_true = batch['image'], batch['mask']

            # Move data to device and ensure correct data types
            image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            mask_true = mask_true.to(device=device, dtype=torch.long)

            # Predict the mask
            mask_pred = net(image)

            # Compute softmax probabilities and get class predictions
            mask_pred = F.softmax(mask_pred, dim=1)
            mask_pred_class = torch.argmax(mask_pred, dim=1)

            # Flatten masks for easier processing
            mask_pred_class = mask_pred_class.view(-1)
            mask_true = mask_true.view(-1)

            # Compute IoU for each class
            for cls in range(num_classes):
                # Binary masks for the current class
                pred_mask = (mask_pred_class == cls)
                true_mask = (mask_true == cls)

                # Only compute IoU if the ground truth mask is not empty
                if true_mask.sum() > 0:
                    intersection = (pred_mask & true_mask).sum().float()
                    union = (pred_mask | true_mask).sum().float()


                    iou_per_class[cls] += intersection / union
                    valid_classes[cls] += 1

    # Warn about classes not present in the evaluation set
    missing_classes = [cls for cls in range(num_classes) if valid_classes[cls] == 0]
    if missing_classes:
        print(f"Warning: The following classes are not present in the evaluation set: {missing_classes}")

    weights = valid_classes/valid_classes.sum()
    print(weights,weights.sum())

    meanIoUclasses = [iou_per_class[cl]/valid_classes[cl] for cl in range(num_classes) if valid_classes[cls] != 0 else "none"]

    totalWeightedMeanIoU = 0
    for cl in range(0,21):
        if meanIoUclasses[cl] !== "none":
            totalWeightedMeanIoU += meanIoUclasses[cl]*weights[cl]

    # Print per-class IoU for debugging or analysis

    # Restore model to training mode
    net.train()

    return totalWeightedMeanIoU

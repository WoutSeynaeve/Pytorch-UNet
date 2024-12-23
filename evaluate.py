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
def evaluateWeaklySupervised(net, dataloader, device, amp):
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

import argparse
import logging
import os
import random
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from pathlib import Path
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import numpy as np
from evaluate import evaluate, evaluateWeaklySupervised
from unet import UNet
from utils.data_loading import BasicDataset, CarvanaDataset
from utils.dice_score import dice_loss


debug = False
if debug:
    dir_img = Path('../../DebugDataset/JPEGImages/')
    dir_mask = Path('../../DebugDataset/segmentationGroundTruth/')
    dir_weaklabel = Path('../../DebugDataset/Dataset1/')
    dir_checkpoint = Path('./DebugCheckpoints/')
else:   
    dir_img = Path('../../datasetPascalVOC/JPEGImages/')
    dir_mask = Path('../../datasetPascalVOC/segmentationGroundTruth/')
    dir_weaklabel = Path('../../datasetPascalVOC/Dataset1/')
    dir_checkpoint = Path('./checkpoints/')



def train_model(
        model,
        device,
        epochs: int = 5,
        batch_size: int = 1,
        learning_rate: float = 1e-5,
        val_percent: float = 0.1,
        save_checkpoint: bool = True,
        img_scale: float = 0.5,
        amp: bool = False,
        weight_decay: float = 1e-8,
        momentum: float = 0.999,
        gradient_clipping: float = 1.0,
):
    # 1. Create dataset
    # try:
    #     dataset = CarvanaDataset(dir_img, dir_mask, img_scale)
    # except (AssertionError, RuntimeError, IndexError):
    dataset = BasicDataset(dir_img, dir_mask, img_scale)

    # 2. Split into train / validation partitions
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))
    # 3. Create data loaders
    loader_args = dict(batch_size=batch_size, num_workers=os.cpu_count(), pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)

    # # (Initialize logging)
    # experiment = wandb.init(project='U-Net', resume='allow', anonymous='must')
    # experiment.config.update(
    #     dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
    #          val_percent=val_percent, save_checkpoint=save_checkpoint, img_scale=img_scale, amp=amp)
    # )

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Images scaling:  {img_scale}
        Mixed Precision: {amp}
    ''')

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    optimizer = optim.RMSprop(model.parameters(),
                              lr=learning_rate, weight_decay=weight_decay, momentum=momentum, foreach=True)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)  # goal: maximize Dice score
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)

    #these are the class frequencies of the dataset !! for background some arbritray weight is chosen
    class_counts = torch.tensor([
    500000,178, 144, 208, 150, 183, 152, 255, 250, 271, 135, 
    157, 249, 147, 157, 887, 167, 120, 183, 167, 158
    ])
    # Compute the weight for each class (inverse frequency)
    weightsVOC = 1.0 / class_counts.float()
    print(weightsVOC)
    weightsVOC /= torch.sum(weightsVOC)
    weightsVOC = weightsVOC**2
    weightsVOC /= torch.sum(weightsVOC)
    print(weightsVOC)
    weightsVOC = weightsVOC.cuda(0)
    #added ignore_index to ignore uncertain-labelled pixels in the ground truth masks
    criterion = nn.CrossEntropyLoss(weight = weightsVOC, ignore_index=21)

    global_step = 0
    if debug:
        weightsVOC[0] = 0.01
        criterion = nn.CrossEntropyLoss(weight = weightsVOC, ignore_index=21)
     
        epochs = 1  
        signal = 0
            # 5. Begin training
        for epoch in range(1, epochs + 1):
            model.train()
            epoch_loss = 0
            with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
                for batch in train_loader:
                    for i in range(300):
                        images, true_masks = batch['image'], batch['mask']
                        
                        assert images.shape[1] == model.n_channels, \
                            f'Network has been defined with {model.n_channels} input channels, ' \
                            f'but loaded images have {images.shape[1]} channels. Please check that ' \
                            'the images are loaded correctly.'

                        images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                        true_masks = true_masks.to(device=device, dtype=torch.long)

                        with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                            masks_pred = model(images)
                            #after a while, mask_pred becomes all NAN !! problem!!

                            loss = multiclass_tversky_loss(masks_pred, true_masks,class_counts,alpha, beta)
                            if loss.item() > 0 and loss.item() < np.inf:
                                pass
                            else:
                                print(loss,"\n",masks_pred)
                                report = 0
                                assert(report == 1)
                        if loss.item() < 0.0005:
                            break
                        optimizer.zero_grad(set_to_none=True)
                        grad_scaler.scale(loss).backward()
                        grad_scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                        grad_scaler.step(optimizer)
                        grad_scaler.update()
                        
                        pbar.update(images.shape[0])
                        global_step += 1
                        epoch_loss += loss
                        print("average loss so far:",epoch_loss/(global_step))
                        # experiment.log({
                        #     'train loss': loss.item(),
                        #     'step': global_step,
                        #     'epoch': epoch
                        # })
                        pbar.set_postfix(**{'loss (batch)': loss.item()})

                        # Evaluation round
                        
                        if i%20 == 1:
                            print("TRAIN EVAL:",evaluateWeaklySupervised(model,train_loader,device,amp))
                                

                    if save_checkpoint:
                        Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
                        state_dict = model.state_dict()
                        state_dict['mask_values'] = dataset.mask_values
                        torch.save(state_dict, str(dir_checkpoint / 'checkpoint_epoch{}.pth'.format(epoch)))
                        logging.info(f'Checkpoint {epoch} saved!')

                   
    else:
        # 5. Begin training
        for epoch in range(1, epochs + 1):
            model.train()
            epoch_loss = 0

            """added"""
            if epoch == 100:
                print("changing weights")
                class_counts[0] = 50000
            if epoch == 150:
                print("changing weights")
                class_counts[0] = 10000
            if epoch == 200:
                print("changing weights")
                class_counts[0] = 5000
            if epoch == 240:
                print("changing lr")
                print("and class counts back to ignore more background")
                class_counts[0] = 50000
                optimizer = optim.RMSprop(model.parameters(),
                lr=1e-10, weight_decay=weight_decay, momentum=momentum, foreach=True)    
            if epoch == 300:
                print("changing lr")
                class_counts[0] = 5000
                optimizer = optim.RMSprop(model.parameters(),
                lr=1e-8, weight_decay=weight_decay, momentum=momentum, foreach=True)            
            """^^added^^^"""

            with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
                for batch in train_loader:
                    images, true_masks = batch['image'], batch['mask']

                    assert images.shape[1] == model.n_channels, \
                        f'Network has been defined with {model.n_channels} input channels, ' \
                        f'but loaded images have {images.shape[1]} channels. Please check that ' \
                        'the images are loaded correctly.'

                    images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                    true_masks = true_masks.to(device=device, dtype=torch.long)

                    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                        masks_pred = model(images)
                        # if model.n_classes == 1:
                        #     loss = criterion(masks_pred.squeeze(1), true_masks.float())
                        #     loss += dice_loss(F.sigmoid(masks_pred.squeeze(1)), true_masks.float(), multiclass=False)
                        # else:
                        #     loss = criterion(masks_pred, true_masks)
                        #     loss += dice_loss(
                        #         F.softmax(masks_pred, dim=1).float(),
                        #         F.one_hot(true_masks, model.n_classes).permute(0, 3, 1, 2).float(),
                        #         multiclass=True
                        #     )
                        
                        #loss = multiclass_tversky_loss(masks_pred, true_masks,class_counts,0.5,0.5)  # Computes loss
                        loss = multiclass_tversky_loss2(masks_pred, true_masks,class_counts,0.5,0.5)


                    optimizer.zero_grad(set_to_none=True)
                    grad_scaler.scale(loss).backward()
                    grad_scaler.unscale_(optimizer)
                    clipped_grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                    # print(f"Gradient norm before clipping: {clipped_grad_norm:.4f}")
                    # print(f"Clipping threshold: {gradient_clipping}")
                    grad_scaler.step(optimizer)
                    grad_scaler.update()
                    

                    
                    global_step += 1
                    epoch_loss += loss.item()
                    # experiment.log({
                    #     'train loss': loss.item(),
                    #     'step': global_step,
                    #     'epoch': epoch
                    # })
                    if False:
                        pbar.update(images.shape[0])
                        pbar.set_postfix(**{'loss (batch)': loss.item()})
                    
                    # Evaluation round
                    division_step = (n_train // (2 * batch_size))
                    if division_step > 0:
                        if global_step % division_step == 0:
                            # histograms = {}
                            # for tag, value in model.named_parameters():
                            #     tag = tag.replace('/', '.')
                            #     if not (torch.isinf(value) | torch.isnan(value)).any():
                            #         histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
                            #     if not (torch.isinf(value.grad) | torch.isnan(value.grad)).any():
                            #         histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())

                            #val_score = evaluate(model, val_loader, device, amp)
                            val_score = evaluateWeaklySupervised(model, val_loader, device, amp)
                            scheduler.step(val_score)
                                    
                            logging.info('Validation IoU score: {}'.format(val_score))
                            # try:
                            #     experiment.log({
                            #         'learning rate': optimizer.param_groups[0]['lr'],
                            #         'validation Dice': val_score,
                            #         'images': wandb.Image(images[0].cpu()),
                            #         'masks': {
                            #             'true': wandb.Image(true_masks[0].float().cpu()),
                            #             'pred': wandb.Image(masks_pred.argmax(dim=1)[0].float().cpu()),
                            #         },
                            #         'step': global_step,
                            #         'epoch': epoch,
                            #         **histograms
                            #     })
                            # except:
                            #     pass
                    
            if epoch%10 == 5:
                print("TRAIN EVAL:",evaluateWeaklySupervised(model,train_loader,device,amp))
            print("average loss this epoch:",epoch_loss/2622)
            if save_checkpoint:
                Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
                state_dict = model.state_dict()
                state_dict['mask_values'] = dataset.mask_values
                torch.save(state_dict, str(dir_checkpoint / 'checkpoint_epoch{}.pth'.format(epoch)))
                logging.info(f'Checkpoint {epoch} saved!')
def multiclass_tversky_loss(logits, true_mask,class_counts,alpha, beta, num_classes=21, ignore_index=21, smooth=1e-6):
    """
    Computes multi-class Tversky loss for segmentation.

    Args:
        logits (Tensor): Model output of shape (B, C, H, W) (before softmax).
        true_mask (Tensor): Ground truth of shape (B, H, W) with class indices (0 to 21).
        num_classes (int): Number of relevant classes (excluding ignored class).
        ignore_index (int): Class index to ignore in loss computation.
        alpha (float): Controls penalty for false positives.
        beta (float): Controls penalty for false negatives.
        smooth (float): Small constant to avoid division by zero.

    Returns:
        Tensor: Scalar Tversky loss value.
    """
    # Convert logits to probabilities
    probs = F.softmax(logits, dim=1)  # Shape: (B, C, H, W)

    # Create one-hot encoding of true_mask, ignoring ignore_index
    true_mask = true_mask.clone()
    true_mask[true_mask == ignore_index] = num_classes  # Temporarily set ignored pixels to an out-of-range value
    true_one_hot = F.one_hot(true_mask, num_classes=num_classes + 1).permute(0, 3, 1, 2).float()  # Shape: (B, C+1, H, W)

    # Remove ignored class channel
    true_one_hot = true_one_hot[:, :num_classes, :, :]  # Shape: (B, num_classes, H, W)

    # Compute per-class Tversky index
    TP = (probs * true_one_hot).sum(dim=(2, 3))  # True positives (sum over H, W)
    FP = ((1 - true_one_hot) * probs).sum(dim=(2, 3))  # False positives
    FN = (true_one_hot * (1 - probs)).sum(dim=(2, 3))  # False negatives
    # Compute per-class Tversky score
    tversky_score = (TP + smooth) / (TP + alpha * FP + beta * FN + smooth)
    # Compute class weights as inverse class frequencies
    class_weights = 1.0 / (class_counts.float() + 1e-6)
    class_weights = class_weights / class_weights.sum()  # Normalize to sum to 1

    # Compute weighted mean Tversky loss
    weighted_loss = (1 - tversky_score) * class_weights.to(logits.device)  # Move to GPU if needed
    return weighted_loss.sum()  # Sum ensures proper weighting
def multiclass_tversky_loss2(logits, true_mask, class_counts, alpha, beta, num_classes=21, ignore_index=21, smooth=1e-6):
    """
    Computes multi-class Tversky loss for segmentation, only using classes present in the true mask.
    """
    # Convert logits to probabilities
    probs = F.softmax(logits, dim=1)  # Shape: (B, C, H, W)
    
    # Identify unique classes in the mask (excluding ignore_index)
    unique_classes = torch.unique(true_mask)
    unique_classes = unique_classes[unique_classes != ignore_index]  # Remove ignored class
    
    if unique_classes.numel() == 0:  # If no valid classes exist, return zero loss
        return torch.tensor(0.0, device=logits.device, dtype=logits.dtype)
    # Create one-hot encoding of true_mask, ignoring ignore_index
    true_mask = true_mask.clone()
    true_mask[true_mask == ignore_index] = num_classes  # Temporarily set ignored pixels to an out-of-range value
    true_one_hot = F.one_hot(true_mask, num_classes=num_classes + 1).permute(0, 3, 1, 2).float()  # Shape: (B, C+1, H, W)
    
    # Remove ignored class channel
    true_one_hot = true_one_hot[:, :num_classes, :, :]
    
    # Compute per-class Tversky index
    TP = (probs * true_one_hot).sum(dim=(2, 3))  # True positives
    FP = ((1 - true_one_hot) * probs).sum(dim=(2, 3))  # False positives
    FN = (true_one_hot * (1 - probs)).sum(dim=(2, 3))  # False negatives
    
    # Compute per-class Tversky score
    tversky_score = (TP + smooth) / (TP + alpha * FP + beta * FN + smooth)
    
    # Filter to only use present classes
    present_class_mask = torch.zeros(num_classes, device=logits.device, dtype=torch.bool)
    present_class_mask[unique_classes] = True  # Mark present classes
    
    # Compute class weights as inverse class frequencies
    class_weights = 1.0 / (class_counts.float() + 1e-6)
    class_weights = class_weights / class_weights.sum()  # Normalize
    class_weights = class_weights.to(logits.device)  # Ensure it is on the same device
    # Apply mask to ignore absent classes
    filtered_tversky_score = tversky_score[present_class_mask.expand_as(tversky_score)]
    filtered_class_weights = class_weights[present_class_mask.squeeze(0)]
    
    # Compute weighted mean Tversky loss
    weighted_loss = (1 - filtered_tversky_score) * filtered_class_weights.to(logits.device)
    return weighted_loss.sum()
   

def get_args():
    #note: Batch size can be upped, but the images must be resized (scaled or padded) to have the same format!!
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=360, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=1, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-9,
                        help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--scale', '-s', type=float, default=0.5, help='Downscaling factor of the images')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--amp', action='store_true', default=True, help='Use mixed precision')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=21, help='Number of classes')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    model = UNet(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)
    model = model.to(memory_format=torch.channels_last)

    logging.info(f'Network:\n'
                 f'\t{model.n_channels} input channels\n'
                 f'\t{model.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if model.bilinear else "Transposed conv"} upscaling')

    if args.load:
        state_dict = torch.load(args.load, map_location=device)
        del state_dict['mask_values']
        model.load_state_dict(state_dict)
        logging.info(f'Model loaded from {args.load}')

    model.to(device=device)
    try:
        train_model(
            model=model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            img_scale=args.scale,
            val_percent=args.val / 100,
            amp=args.amp
        )
    except torch.cuda.OutOfMemoryError:
        logging.error('Detected OutOfMemoryError! '
                      'Enabling checkpointing to reduce memory usage, but this slows down training. '
                      'Consider enabling AMP (--amp) for fast and memory efficient training')
        torch.cuda.empty_cache()
        model.use_checkpointing()
        train_model(
            model=model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            img_scale=args.scale,
            val_percent=args.val / 100,
            amp=args.amp
        )

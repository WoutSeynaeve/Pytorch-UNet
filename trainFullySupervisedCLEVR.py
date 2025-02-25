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
from LogicLossVOC.WeakLabelLogicLossCLEVR import calculateLogicLoss
from evaluate import evaluate, evaluateWeaklySupervised, evaluateWeaklySupervisedCLEVR,evaluateWeaklySupervisedCLEVROLD
from unet import UNet
from utils.data_loading import WeakLabelDataset,BasicDataset,WeakLabelDatasetCLEVR
import numpy as np 

debug = False
if debug:
    dir_img = Path('../../DebugDatasetCLEVR/imagesWeakDataset/')
    dir_weaklabel = Path('../../DebugDatasetCLEVR/annotationsTrain/')
    dir_checkpoint = Path('./DebugCheckpoints/')
else:   
    # dir_img = Path('../../datasetCLEVR/imagesWeakDataset/')
    # dir_weaklabel = Path('../../datasetCLEVR/annotationsTrain/')
    # dir_checkpoint = Path('./checkpoints/')
    dir_img = Path('../../datasetCLEVRaug/augmented/')
    dir_weaklabel = Path('../../datasetCLEVRaug/scaledAnnotationsTrain4/')
    dir_checkpoint = Path('./checkpoints/')

class_values = {
    "background": 0,
    "cylinder": 1,
    "sphere": 2,
    "cube": 3
}
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
        configuration: int = 0,
):
    # 1. Create dataset
    # try:
    #     dataset = CarvanaDataset(dir_img, dir_mask, img_scale)
    # except (AssertionError, RuntimeError, IndexError):
    dataset = WeakLabelDatasetCLEVR(dir_img, dir_weaklabel, img_scale)

    # 2. Split into train / validation partitions
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    #train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))
    train_set = torch.utils.data.Subset(dataset, range(n_train))
    val_set = torch.utils.data.Subset(dataset, range(n_train, len(dataset)))
    print(train_set.indices, val_set.indices)
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
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)  # goal: maximize overlap
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)

    global_step = 0

 
    if debug:
        epochs = 1
        signal = 0
            # 5. Begin training
        for epoch in range(1, epochs + 1):
            model.train()
            epoch_loss = 0
            with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
                for batch in train_loader:
                    for i in range(300):
                        images, weaklabel = batch['image'], batch["weaklabel"]
                        assert images.shape[1] == model.n_channels, \
                            f'Network has been defined with {model.n_channels} input channels, ' \
                            f'but loaded images have {images.shape[1]} channels. Please check that ' \
                            'the images are loaded correctly.'

                        images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                        with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                            masks_pred = model(images)
                            #after a while, mask_pred becomes all NAN !! problem!!

                            loss = calculateLogicLoss(masks_pred,weaklabel,signal, True)
                            if loss.item() > 0 and loss.item() < np.inf:
                                pass
                            else:
                                print(loss,"\n",masks_pred)
                                report = 0
                                assert(report == 1)
                        if loss.item() < 0.5:
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
                        
                        if i%10 == 5:
                            print("TRAIN EVAL:",evaluateWeaklySupervisedCLEVR(model,train_loader,device,amp))
                                

                    if save_checkpoint:
                        Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
                        state_dict = model.state_dict()
                        #state_dict['mask_values'] = dataset.mask_values
                        torch.save(state_dict, str(dir_checkpoint / 'checkpoint_epoch{}.pth'.format(epoch)))
                        logging.info(f'Checkpoint {epoch} saved!')

                   
    else:
        showPbar = False
        signal = 0
        # 5. Begin training
        for epoch in range(1, epochs + 1):
            model.train()
            epoch_loss = 0
            with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
                
                for batch in train_loader:
                    images, weaklabel = batch['image'], batch["weaklabel"]
                    assert images.shape[1] == model.n_channels, \
                        f'Network has been defined with {model.n_channels} input channels, ' \
                        f'but loaded images have {images.shape[1]} channels. Please check that ' \
                        'the images are loaded correctly.'

                    images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)

                    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                        masks_pred = model(images)
                        #after a while, mask_pred becomes all NAN !! problem!!
                        # if epoch > 30:
                        #     signal = 1
                        # if epoch > 50:  #testing purposes
                        #     signal = 2
                        #loss = calculateLogicLoss(masks_pred,weaklabel,signal)
                        _, _, H, W = images.shape  # Get image dimensions
                        loss = diceLoss(masks_pred,weaklabel,H,W,device)
                        if loss.item() > 0 and loss.item() < np.inf:
                            optimizer.zero_grad(set_to_none=True)
                            grad_scaler.scale(loss).backward()
                            grad_scaler.unscale_(optimizer)
                            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                            grad_scaler.step(optimizer)
                            grad_scaler.update()
                        else:
                            print(loss,"\n",masks_pred)
                            report = 0
                            #assert(report == 1)
                    
                    
                    if showPbar:
                        pbar.update(images.shape[0])
                        pbar.set_postfix(**{'loss (batch)': loss.item()})
                    global_step += 1
                    
                    epoch_loss += loss.detach() 
                    del images, weaklabel, masks_pred, loss  # Free memory
                    torch.cuda.empty_cache()  # Clear GPU memory
                    # experiment.log({
                    #     'train loss': loss.item(),
                    #     'step': global_step,
                    #     'epoch': epoch
                    # })
                    

                    # Evaluation round
                    division_step = (n_train // (5 * batch_size))
                    if division_step > 0:
                        if global_step % division_step == 0:
                            # histograms = {}
                            # for tag, value in model.named_parameters():
                            #     tag = tag.replace('/', '.')
                            #     if not (torch.isinf(value) | torch.isnan(value)).any():
                            #         histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
                            #     if not (torch.isinf(value.grad) | torch.isnan(value.grad)).any():
                            #         histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())

                            #val_score = evaluateWeaklySupervised2(model, val_loader, device, amp)
                          
                            val_score = evaluateWeaklySupervisedCLEVR(model, val_loader, device, amp)
                            val_score = evaluateWeaklySupervisedCLEVROLD(model, val_loader, device, amp)
                            if epoch%10 == 5:
                            
                                print("TRAIN EVAL:",evaluateWeaklySupervisedCLEVR(model,train_loader,device,amp))
                                
                            logging.info('Validation overlap score: {}'.format(val_score))
                            print( " new lr: ", optimizer.param_groups[0]['lr'])
                            scheduler.step(val_score)

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
            print("average loss during this epoch = ",epoch_loss/431) #pas dit nog aan eventueel
            if save_checkpoint:
                Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
                state_dict = model.state_dict()
                #state_dict['mask_values'] = dataset.mask_values
                torch.save(state_dict, str(dir_checkpoint / 'checkpoint_epoch{}.pth'.format(epoch)))
                logging.info(f'Checkpoint {epoch} saved!')


def diceLoss(mask_pred, weaklabel, H, W, device, smooth=1.0):
    """
    Compute Dice Loss with bounding box supervision.
    Uses softmax probabilities directly for class predictions.
    """

    # Convert logits to probabilities
    mask_pred = F.softmax(mask_pred, dim=1)  # (1, C, H, W)

    total_dice_loss = 0.0  # Initialize variable to accumulate loss

    # Process weak labels (bounding boxes)
    bboxlist = weaklabel[0][4]
    indd = 0
    for i in bboxlist:
        indd += 1
        if indd % 2 == 1:
            i = i[0].split(',')
            objecIndex = class_values[i[0]]  # Convert class label to index
            x1, x2, y1, y2 = map(int, i[1:5])

            # Initialize the ground truth for this bounding box (class-specific)
            mask_gt = torch.zeros((H, W), dtype=torch.float32, device=device)
            mask_gt[y1:y2+1, x1:x2+1] = 1

            # Get the softmax probability for the specific class
            pred_mask = mask_pred[:, objecIndex, :, :]  # Softmax probabilities for class objecIndex

            # Compute intersection and union for Dice calculation
            intersection = torch.sum(mask_gt * pred_mask)  # Intersection (AND)
            union = torch.sum(mask_gt) + torch.sum(pred_mask)  # Union (OR)

            # Dice coefficient calculation
            dice_score = (2.0 * intersection + smooth) / (union + smooth)
            
            # Accumulate the loss without in-place operation
            total_dice_loss += (1 - dice_score)  # Minimize (1 - Dice)

    # Return the average Dice loss over all bounding boxes
    return total_dice_loss / (len(bboxlist)/2)  # Average Dice loss over all bounding boxes

def diceLoss3(mask_pred, weaklabel, H, W, device, num_classes=4, smooth=1.0):
    """
    Compute Dice Loss with bounding box supervision.
    Uses softmax probabilities directly for class predictions.
    """

    # Convert logits to probabilities
    mask_pred = F.softmax(mask_pred, dim=1)  # (1, C, H, W)

    # Create ground truth mask initialized with background (0)
    mask_gt = torch.zeros((H, W), dtype=torch.long, device=device)

    # Process weak labels (bounding boxes)
    bboxlist = weaklabel[0][4]
    for i in bboxlist:
        i = i[0].split(',')
        objecIndex = class_values[i[0]]  # Convert class label to index
        x1, x2, y1, y2 = map(int, i[1:5])
        
        # Assign the label only if the region is background (0)
        mask_region = mask_gt[y1:y2+1, x1:x2+1]
        mask_gt[y1:y2+1, x1:x2+1] = torch.where(mask_region == 0, objecIndex, mask_region)

    dice_loss = torch.tensor(0.0, device=device, requires_grad=True)  # Ensure differentiability
    num_fg_classes = num_classes - 1  # Excluding background (index 0)

    # Compute Dice Loss per class (excluding background)
    for c in range(1, num_classes):  # Start from 1 to exclude background
        gt_mask = (mask_gt == c).float()  # Ground truth mask for class c (binary)
        pred_mask = mask_pred[:, c, :, :]  # Softmax probabilities for class c

        intersection = torch.sum(gt_mask * pred_mask)  # Intersection (AND)
        union = torch.sum(gt_mask) + torch.sum(pred_mask)  # Union (OR)

        # Dice coefficient calculation
        dice_score = (2.0 * intersection + smooth) / (union + smooth)
        dice_loss = dice_loss + (1 - dice_score)  # Minimize (1 - Dice)

    return dice_loss / num_fg_classes  # Average Dice loss over foreground classes

    # Compute Dice Loss per class (excluding background)
    for c in range(1, num_classes):  
        gt_mask = (mask_gt == c).float()  # Ground truth binary mask for class c
        pred_mask = (mask_pred_class == c).float()  # Predicted binary mask for class c

        intersection = torch.sum(gt_mask * pred_mask)  # Element-wise multiplication (AND)
        union = torch.sum(gt_mask) + torch.sum(pred_mask)  # Total pixels of both masks

        dice_score = (2.0 * intersection + smooth) / (union + smooth)  # Dice coefficient
        dice_loss += (1 - dice_score)  # Minimize (1 - Dice)

    return dice_loss / num_fg_classes  # Average Dice loss over foreground classes
    
def get_args():
    #note: Batch size can be upped, but the images must be resized (scaled or padded) to have the same format!!
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=180, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=1, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-7,
                        help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--scale', '-s', type=float, default=1, help='Downscaling factor of the images')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--amp', action='store_true', default=True, help='Use mixed precision')
    parser.add_argument('--bilinear', action='store_true', default=True, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=4, help='Number of classes')
    parser.add_argument('--configuration', '-conf', dest='config', type=int, default=0, help='configuration id')

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
            amp=args.amp,
            configuration=args.config
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
            amp=args.amp,
            configuration=arg.config
        )

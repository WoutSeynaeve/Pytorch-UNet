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
from LogicLossVOC.WeakLabelLogicLoss import calculateLogicLoss
from evaluate import evaluate, evaluateWeaklySupervised,evaluateWeaklySupervised2
from unet import UNet
from utils.data_loading import WeakLabelDataset,BasicDataset
import numpy as np 

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
        configuration: int = 0,
):
    # 1. Create dataset
    # try:
    #     dataset = CarvanaDataset(dir_img, dir_mask, img_scale)
    # except (AssertionError, RuntimeError, IndexError):
    dataset = WeakLabelDataset(dir_img, dir_mask, dir_weaklabel, img_scale)

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
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)  # goal: maximize overlap
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)

    global_step = 0

    #             ImageLevelLoss, Adjacencies, BBoxObject, OutsideBBoxNotObject, BBoxBackground, Smoothness, Scribbles, Relations
    configuration1 = [[True,5],   [False,1] ,    [True,0.1],        [True,1] ,        [True,10],     [False,100], [False,1],  [False,1]]
    
    configurations = [configuration1]
    configuration_instance = configurations[configuration]
    if debug:
        epochs = 1
            # 5. Begin training
        for epoch in range(1, epochs + 1):
            model.train()
            epoch_loss = 0
            with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
                for batch in train_loader:
                    for i in range(300):
                        images, true_masks, weaklabel = batch['image'], batch['mask'], batch["weaklabel"]
                        
                        assert images.shape[1] == model.n_channels, \
                            f'Network has been defined with {model.n_channels} input channels, ' \
                            f'but loaded images have {images.shape[1]} channels. Please check that ' \
                            'the images are loaded correctly.'

                        images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                        with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                            masks_pred = model(images)
                            #after a while, mask_pred becomes all NAN !! problem!!

                            loss = calculateLogicLoss(masks_pred,weaklabel,True)
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
                    if save_checkpoint:
                        Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
                        state_dict = model.state_dict()
                        state_dict['mask_values'] = dataset.mask_values
                        torch.save(state_dict, str(dir_checkpoint / 'checkpoint_epoch{}.pth'.format(epoch)))
                        logging.info(f'Checkpoint {epoch} saved!')

                   
    else:
        showPbar = False
        # 5. Begin training
        for epoch in range(1, epochs + 1):
            model.train()
            epoch_loss = 0
            with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
                
                for batch in train_loader:
                
                    images, true_masks, weaklabel = batch['image'], batch['mask'], batch["weaklabel"]
                
                    assert images.shape[1] == model.n_channels, \
                        f'Network has been defined with {model.n_channels} input channels, ' \
                        f'but loaded images have {images.shape[1]} channels. Please check that ' \
                        'the images are loaded correctly.'

                    images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)

                    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                        masks_pred = model(images)
                        #after a while, mask_pred becomes all NAN !! problem!!
                     
                        loss = calculateLogicLoss(masks_pred,weaklabel)
                        if loss.item() > 0 and loss.item() < np.inf:
                            pass
                        else:
                            print(loss,"\n",masks_pred)
                            report = 0
                            assert(report == 1)
                    optimizer.zero_grad(set_to_none=True)
                    grad_scaler.scale(loss).backward()
                    grad_scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                    grad_scaler.step(optimizer)
                    grad_scaler.update()
                    
                    if showPbar:
                        pbar.update(images.shape[0])
                        pbar.set_postfix(**{'loss (batch)': loss.item()})
                    global_step += 1
                    
                    epoch_loss += loss.detach() 
                    del images, true_masks, weaklabel, masks_pred, loss  # Free memory
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
                          
                            val_score = evaluateWeaklySupervised(model, val_loader, device, amp)
                            if epoch%10 == 5:
                                print("TRAIN EVAL:",evaluateWeaklySupervised(model,train_loader,device,amp))
                                
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
            print("average loss during this epoch = ",epoch_loss/2622) #pas dit nog aan eventueel
            if save_checkpoint:
                Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
                state_dict = model.state_dict()
                state_dict['mask_values'] = dataset.mask_values
                torch.save(state_dict, str(dir_checkpoint / 'checkpoint_epoch{}.pth'.format(epoch)))
                logging.info(f'Checkpoint {epoch} saved!')


def get_args():
    #note: Batch size can be upped, but the images must be resized (scaled or padded) to have the same format!!
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=180, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=1, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-6,
                        help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--scale', '-s', type=float, default=1, help='Downscaling factor of the images')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=21, help='Number of classes')
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

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
from evaluate import evaluate, evaluateWeaklySupervised, evaluateWeaklySupervisedCLEVR, evaluateFullySupervisedCLEVR,evaluateFullySupervisedCLEVRwPrecisionRecall
from unet import UNet
from utils.data_loading import WeakLabelDataset,BasicDataset,WeakLabelDatasetCLEVR,BasicDatasetCLEVR
import numpy as np 

seed = 42
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

debug = False
printLosses = True
debugIts = 400
if debug:
    dir_img = Path('../../DebugDatasetCLEVR/imagesWeakDataset/')
    dir_weaklabel = Path('../../DebugDatasetCLEVR/annotationsTrain/')
    dir_mask = Path('../../DebugDatasetCLEVR/maskstrain')
    dir_checkpoint = Path('./DebugCheckpoints/')
else:
    # dir_img = Path('../../datasetCLEVR/imagesWeakDataset/')
    # dir_weaklabel = Path('../../datasetCLEVR/annotationsTrain/')
    # dir_checkpoint = Path('./checkpoints/')
    dir_img = Path('../../datasetCLEVRaug/ImagesTraining/')
    dir_weaklabel = Path('../../datasetCLEVRaug/WeakLabelsTraining/')
    dir_mask = Path('../../datasetCLEVRaug/MasksTraining')
    dir_img_test = Path('../../datasetCLEVRaug/ImagesValidation/')
    dir_mask_test = Path('../../datasetCLEVRaug/MasksValidation')
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
    configuration_dict = {}
    if configuration == 0:
        configuration_dict = {
            "ImageLevel": [False, 1],
            #                   outsideBbox  BboxAtmost
            "BBox": [[True, 1],   [True, 1],   [True, 1]],
            "BBoxFull": [True, 1],
            "Scribbles": [True, 1],
            "Area": [True, 1],
            "Point": [True, 10],
            "Adjacency": [True, 1],
            "Relations": [True, 1],
            "SoftRelations": [True, 1],
            #global constraints:
            "OneHot": [True, 10],
            "MinSizeBackground": [False, 1],
            "MaxSizeBackground": [False, 1],
            "MinSizeShapes": [False, 1],
            "MaxSizeShapes": [False, 1],
            "Smoothness": [False, 100],
        }
    # 1. Create dataset
    # try:
    #     dataset = CarvanaDataset(dir_img, dir_mask, img_scale)
    # except (AssertionError, RuntimeError, IndexError):
    dataset = WeakLabelDatasetCLEVR(dir_img, dir_weaklabel, img_scale)
    if not debug:
        dataset_test = BasicDatasetCLEVR(dir_img_test, dir_mask_test, img_scale)
    dataset_test_trainset = BasicDatasetCLEVR(dir_img, dir_mask, img_scale)

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
    if not debug:
        test_loader = DataLoader(dataset_test,shuffle=True,**loader_args)
    test_trainset_loader = DataLoader(dataset_test_trainset,shuffle=True,**loader_args)
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
                    for i in range(debugIts):
                        images, weaklabel = batch['image'], batch["weaklabel"]
                        assert images.shape[1] == model.n_channels, \
                            f'Network has been defined with {model.n_channels} input channels, ' \
                            f'but loaded images have {images.shape[1]} channels. Please check that ' \
                            'the images are loaded correctly.'

                        images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                        with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                            masks_pred = model(images)
                            #after a while, mask_pred becomes all NAN !! problem!!
                            if i == debugIts-1:
                                loss = calculateLogicLoss(masks_pred,weaklabel,configuration_dict,True)
                            else:
                                loss = calculateLogicLoss(masks_pred,weaklabel,configuration_dict,printLosses)
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
                        
                        # experiment.log({
                        #     'train loss': loss.item(),
                        #     'step': global_step,
                        #     'epoch': epoch
                        # })
                        pbar.set_postfix(**{'loss (batch)': loss.item()})

                        # Evaluation round
                        if i%10 == 0:
                            print("TRAIN EVAL:",evaluateFullySupervisedCLEVR(model,test_trainset_loader,device,amp))
                            print(evaluateFullySupervisedCLEVRwPrecisionRecall(model,test_trainset_loader,device,amp))
                            
                                

                    if save_checkpoint:
                        print("TRAIN EVAL:",evaluateFullySupervisedCLEVR(model,test_trainset_loader,device,amp))
                        print(evaluateFullySupervisedCLEVRwPrecisionRecall(model,test_trainset_loader,device,amp))
                        Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
                        state_dict = model.state_dict()
                        state_dict['mask_values'] = dataset_test_trainset.mask_values
                        torch.save(state_dict, str(dir_checkpoint / 'checkpoint_epoch{}.pth'.format(epoch)))
                        logging.info(f'Checkpoint {epoch} saved!')

                   
    else:
        showPbar = False
        signal = 0
        old_learning_rate = optimizer.param_groups[0]['lr']
        # 5. Begin training
        for epoch in range(1, epochs + 1):
            model.train()
            epoch_loss = 0
            print(f'Epoch {epoch}/{epochs}:\n')
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
                    if epoch == epochs:
                        loss = calculateLogicLoss(masks_pred,weaklabel,configuration_dict,True)
                    else:
                        loss = calculateLogicLoss(masks_pred,weaklabel,configuration_dict)
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
                        assert(report == 1)
                
                
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
                division_step = (n_train // (3 * batch_size))
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
                        print("Test Set Eval:")
                        test_score = evaluateFullySupervisedCLEVRwPrecisionRecall(model, test_loader, device, amp)
                        new_learning_rate = optimizer.param_groups[0]['lr']
                        if new_learning_rate != old_learning_rate:
                            print( "new learning rate !!: ", optimizer.param_groups[0]['lr'])
                            old_learning_rate = new_learning_rate
                        print("")
                        scheduler.step(test_score)

                        
                        

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
            if epoch%3 == 0:
                print("Training Set Eval:")
                evaluateFullySupervisedCLEVRwPrecisionRecall(model,test_trainset_loader,device,amp)
                print("")

            print("Average loss this epoch = ",epoch_loss.item()/n_train) #pas dit nog aan eventueel
            if save_checkpoint:
                Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
                state_dict = model.state_dict()
                state_dict['mask_values'] = dataset_test_trainset.mask_values
                torch.save(state_dict, str(dir_checkpoint / 'checkpoint_epoch{}.pth'.format(epoch)))
                logging.info(f'Checkpoint {epoch} saved!')
                print("/////////////////////////")


def get_args():
    #note: Batch size can be upped, but the images must be resized (scaled or padded) to have the same format!!
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=80, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=1, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-7,
                        help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--scale', '-s', type=float, default=1, help='Downscaling factor of the images')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=0,
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

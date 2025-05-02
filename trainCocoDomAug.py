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
import itertools
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from LogicLossVOC.WeakLabelLogicLossCLEVR import calculateLogicLoss, domainlossCOCO
from evaluate import evaluate, evaluateWeaklySupervised, evaluateWeaklySupervisedCLEVR, evaluateFullySupervisedCLEVR,evaluateFullySupervisedCOCOwPrecisionRecall
from unet.unet_model import UNet
from utils.data_loading import WeakLabelDataset,BasicDataset,BasicDatasetCOCOdomExtended,WeakLabelDatasetCLEVR,BasicDatasetCLEVR,CombinedDatasetCLEVR
import numpy as np 



debug = False
printLosses = False
calc_test_loss = False

debugIts = 400
if debug:
    dir_img = Path('../../DebugDatasetCLEVR2/imagesWeakDataset/')
    dir_weaklabel = Path('../../DebugDatasetCLEVR2/annotationsTrain/')
    dir_mask = Path('../../DebugDatasetCLEVR2/maskstrain')
    dir_checkpoint = Path('./DebugCheckpoints/')
else:
    # dir_img = Path('../../datasetCLEVR/imagesWeakDataset/')
    # dir_weaklabel = Path('../../datasetCLEVR/annotationsTrain/')
    # dir_checkpoint = Path('./checkpoints/')
    dir_img = Path('../../datasetCOCO/DomainAugImages/')
    dir_mask = Path('../../datasetCOCO/DomainAugMasks')
    dir_img_test = Path('../../datasetCOCO/AugmentedImagesTest/')
    dir_mask_test = Path('../../datasetCOCO/AugmentedMasksTest')
    
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
        earlyStoppingAmount: int = 100,
):
    
    configuration_dict = {}
    
    if configuration == 0: 
        configuration_dict = {
            "seed": 42,
            "percentFullySupervised": 0.2,

            "domainLossMultiplier": 0.001,   
            #                 implied-NOT  norm-multiplier  implied-multiplier  backgroundAdjacentToEachShape   Symmetric
            "Adjacency": [True,  True,         30,              0.0001,                 False,                  False],
            #                 norm-mult   impl   impl-mult   symmetric 
            "Relations": [True,   2,      True,    0.1,       False, True, False],
            #global constraints:
            "OneHot": [True, 20],

            "Smoothness": [False, 100],
            "MinSizeShapes": [True, 1,0.5,0.0035,0.015],
            "MaxSizeShapes": [True, 1,0.98,0.20,0.42],
        }
    if configuration == 1: 
        configuration_dict = {
            "seed": 123,
            "percentFullySupervised": 0.2,

            "domainLossMultiplier": 0.001,   
            #                 implied-NOT  norm-multiplier  implied-multiplier  backgroundAdjacentToEachShape   Symmetric
            "Adjacency": [True,  True,         30,              0.0001,                 False,                  False],
            #                 norm-mult   impl   impl-mult   symmetric 
            "Relations": [True,   2,      True,    0.1,       False, True, False],
            #global constraints:
            "OneHot": [True, 20],

            "Smoothness": [False, 100],
            "MinSizeShapes": [True, 1,0.5,0.0035,0.015],
            "MaxSizeShapes": [True, 1,0.98,0.20,0.42],
        }
    if configuration == 2: 
        configuration_dict = {
            "seed": 999,
            "percentFullySupervised": 0.2,

            "domainLossMultiplier": 0.001,   
            #                 implied-NOT  norm-multiplier  implied-multiplier  backgroundAdjacentToEachShape   Symmetric
            "Adjacency": [True,  True,         30,              0.0001,                 False,                  False],
            #                 norm-mult   impl   impl-mult   symmetric 
            "Relations": [True,   2,      True,    0.1,       False, True, False],
            #global constraints:
            "OneHot": [True, 20],

            "Smoothness": [False, 100],
            "MinSizeShapes": [True, 1,0.5,0.0035,0.015],
            "MaxSizeShapes": [True, 1,0.98,0.20,0.42],
        }
    if configuration == 3: 
        configuration_dict = {
            "seed": 0,
            "percentFullySupervised": 0.2,

            "domainLossMultiplier": 0.001,   
            #                 implied-NOT  norm-multiplier  implied-multiplier  backgroundAdjacentToEachShape   Symmetric
            "Adjacency": [True,  True,         30,              0.0001,                 False,                  False],
            #                 norm-mult   impl   impl-mult   symmetric 
            "Relations": [True,   2,      True,    0.1,       False, True, False],
            #global constraints:
            "OneHot": [True, 20],

            "Smoothness": [False, 100],
            "MinSizeShapes": [True, 1,0.5,0.0035,0.015],
            "MaxSizeShapes": [True, 1,0.98,0.20,0.42],
        }
    if configuration == 4: 
        configuration_dict = {
            "seed": 100,
            "percentFullySupervised": 0.2,

            "domainLossMultiplier": 0.001,   
            #                 implied-NOT  norm-multiplier  implied-multiplier  backgroundAdjacentToEachShape   Symmetric
            "Adjacency": [True,  True,         30,              0.0001,                 False,                  False],
            #                 norm-mult   impl   impl-mult   symmetric 
            "Relations": [True,   2,      True,    0.1,       False, True, False],
            #global constraints:
            "OneHot": [True, 20],

            "Smoothness": [False, 100],
            "MinSizeShapes": [True, 1,0.5,0.0035,0.015],
            "MaxSizeShapes": [True, 1,0.98,0.20,0.42],
        }
    if configuration == 5: 
        configuration_dict = {
            "seed": 234,
            "percentFullySupervised": 0.2,

            "domainLossMultiplier": 0.001,   
            #                 implied-NOT  norm-multiplier  implied-multiplier  backgroundAdjacentToEachShape   Symmetric
            "Adjacency": [True,  True,         30,              0.0001,                 False,                  False],
            #                 norm-mult   impl   impl-mult   symmetric 
            "Relations": [True,   2,      True,    0.1,       False, True, False],
            #global constraints:
            "OneHot": [True, 20],

            "Smoothness": [False, 100],
            "MinSizeShapes": [True, 1,0.5,0.0035,0.015],
            "MaxSizeShapes": [True, 1,0.98,0.20,0.42],
        }
    
    
    
    experimentFileName = f"./experimentResultsCOCO/experiment_{configuration}.txt"
    writeInfo = ""
    for k in configuration_dict.keys():
        if k == 'BBox':
            if configuration_dict[k][0][0] == True:
                print("Active:  BBox loss ",end='')
                writeInfo += "Active:  BBox loss "
                for info in configuration_dict[k]:
                    print(info,' ',end='')
                    writeInfo += str(info) + " "
                print("")
                writeInfo += "\n"
        elif k == "domainLossMultiplier":
            writeInfo += k +" "+ str(configuration_dict[k]) + "\n"
        elif k == "percentFullySupervised":
            writeInfo += k + " " + str(configuration_dict[k]) + "\n"
        elif k == "seed":
            writeInfo += k + " " + str(configuration_dict[k]) + "\n"
        else:
            if configuration_dict[k][0] == True:
                print("Active: ",k,"loss ",end='')
                writeInfo += "Active: "+k+"loss "
                for info in configuration_dict[k]:
                    print(info,' ',end='')
                    writeInfo += str(info) + " "
                print("")
                writeInfo += "\n"
        
    seed = configuration_dict["seed"]
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # 1. Create dataset
    # try:
    #     dataset = CarvanaDataset(dir_img, dir_mask, img_scale)
    # except (AssertionError, RuntimeError, IndexError):
    dataset = BasicDatasetCOCOdomExtended(dir_img, dir_mask, img_scale)
    if not debug:
        dataset_test = BasicDatasetCLEVR(dir_img_test, dir_mask_test, img_scale)
        n_test = len(dataset_test)

    # 2. Split into train / validation partitions
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    print(n_train,n_test)
    #train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))
    train_set = torch.utils.data.Subset(dataset, range(n_train))
    val_set = torch.utils.data.Subset(dataset, range(n_train, len(dataset)))
    print(train_set.indices, val_set.indices)
    # 3. Create data loaders
    
    loader_args = dict(batch_size=batch_size, num_workers=os.cpu_count(), pin_memory=True)
    if calc_test_loss:
        test_loss_set = WeakLabelDatasetCLEVR(dir_img_test, dir_weaklabel_test, img_scale)
        test_weaklabel_loader = DataLoader(test_loss_set, shuffle=True, **loader_args)

    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)
    if not debug:
        test_loader = DataLoader(dataset_test,shuffle=True,**loader_args)
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
    optimizer = optim.RMSprop(model.parameters(),lr=learning_rate, weight_decay=weight_decay, momentum=momentum, foreach=True)
    #RMSprop werkt beter voor batch size 1 en CLEVR

    #optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)  # goal: maximize overlap
    #scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=50)
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)

    global_step = 0
    gradientVisualisations = True #TO DO !! voor thesisverdediging
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
                        images, weaklabel,idx = batch['image'], batch["weaklabel"]
                        assert images.shape[1] == model.n_channels, \
                            f'Network has been defined with {model.n_channels} input channels, ' \
                            f'but loaded images have {images.shape[1]} channels. Please check that ' \
                            'the images are loaded correctly.'

                        images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                        with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                            masks_pred = model(images)
                            #after a while, mask_pred becomes all NAN !! problem!!
                            if i == debugIts-1:
                                loss = calculateLogicLoss(masks_pred,weaklabel,configuration_dict,-1,True)
                            else:
                                loss = calculateLogicLoss(masks_pred,weaklabel,configuration_dict,-1,printLosses)
                                
                            if loss >=0 and loss < np.inf:
                                pass
                            else:
                                print(loss,"\n",masks_pred)
                                report = 0
                                assert(report == 1)
                            loss += cross_entropy(masks_pred,mask,H,W,device)
                            loss += diceLoss(masks_pred,mask,H,W,device)
                        if loss < 0.0005:
                            break
                       
                        optimizer.zero_grad(set_to_none=True)
                        grad_scaler.scale(loss).backward()
                        grad_scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                        grad_scaler.step(optimizer)
                        grad_scaler.update()
                        
                        pbar.update(images.shape[0])
                        global_step += 1
                            
                        pbar.set_postfix(**{'loss (batch)': loss.item()})

                        # Evaluation round
                        if i%30 == 0:
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
        max_test_score = 0
        max_test_score_epoch = 0
        max_test_score_shape = 0
        max_test_score_epoch_shape = 0
        test_scores = []
        test_scores_shapes = []
        train_scores = []
        train_scores_shapes = []
        train_losses = []
        test_losses = []
        epochssinceimprovement = 0
        domainlossMult = configuration_dict['domainLossMultiplier']
        # 5. Begin training
        for epoch in range(1, epochs + 1):
            model.train()
            if epoch == 170:
                optimizer = optim.RMSprop(model.parameters(),lr=1e-9, weight_decay=weight_decay, momentum=momentum, foreach=True)
            epochssinceimprovement += 1
            if epochssinceimprovement > earlyStoppingAmount:
                break
            epoch_loss = 0
            print(f'Epoch {epoch}/{epochs}:\n')
            batch_n = 0
            testbatch_index = 0
            tot_test_loss = 0
            if calc_test_loss:
                test_iter_loader = iter(test_weaklabel_loader)
            for batch in train_loader:
                batch_n += 1
                images, mask, batch_id = batch['image'], batch["mask"], batch["id"]
                batch_id = batch_id[0]
                assert images.shape[1] == model.n_channels, \
                    f'Network has been defined with {model.n_channels} input channels, ' \
                    f'but loaded images have {images.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                    masks_pred = model(images)
                  
                    _, _, H, W = images.shape 
                    #add domain loss: Adjacency + person above horse + horse under person + always horse + always persone + always background
                    loss = torch.zeros(1, device="cuda") 
                    #bactchID goes from 1 to 130 or -130 to 130 (in case of augmentation)
                    
                        #print("domain loss:",loss)
                    
                    if abs(batch_id) > 120 and batch_id < 999:
                        loss += cross_entropy(masks_pred,mask,H,W,device)
                        loss += diceLoss(masks_pred,mask,H,W,device)
                    # elif batch_id >= 0:
                    #     loss += domainlossCOCO(masks_pred,configuration_dict)
                        

                    if not loss.isnan():
                        if loss > 0:
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
                del images, masks_pred, loss  # Free memory
                torch.cuda.empty_cache()  # Clear GPU memory

                # Evaluation round
                n_of_rounds = 1
                division_step = (n_train // (n_of_rounds * batch_size))
                if division_step > 0:
                    if global_step % division_step == 0:
                        
                        print("Test Set Eval:")
                        test_score,test_score_shape = evaluateFullySupervisedCOCOwPrecisionRecall(model, test_loader, device, amp,-1000,1000)
                        test_scores.append(round(test_score.item(),3))
                        test_scores_shapes.append(round(test_score_shape.item(),3))
                        if test_score > max_test_score:
                            epochssinceimprovement = 0
                            max_test_score_epoch = epoch
                            max_test_score = test_score
                        if test_score_shape > max_test_score_shape:
                            epochssinceimprovement = 0
                            max_test_score_shape_epoch = epoch
                            max_test_score_shape = test_score_shape

                        new_learning_rate = optimizer.param_groups[0]['lr']
                        if new_learning_rate != old_learning_rate:
                            print( "new learning rate !!: ", optimizer.param_groups[0]['lr'])
                            old_learning_rate = new_learning_rate
                        print("")

                        
                #test loss: 90 test in-mages for 478 train, so every 5 iterations, we do a test one:
                if calc_test_loss:
                    if batch_n%5 == 0:
                        if testbatch_index <= 89:           
                            batchTest = next(test_iter_loader)
                            imagestest, weaklabeltest = batchTest['image'], batchTest["weaklabel"]
                            imagestest = imagestest.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                            with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                                masks_pred_test = model(imagestest)
                                newloss_test = calculateLogicLoss(masks_pred_test,weaklabeltest,configuration_dict,testbatch_index)
                            if not newloss_test.isnan():
                                tot_test_loss += newloss_test.item()
                            del imagestest, weaklabeltest, masks_pred_test, newloss_test  # Free memory
                            torch.cuda.empty_cache()
                        testbatch_index += 1
            if calc_test_loss:
                test_loss = tot_test_loss/n_test
                test_losses.append(round(test_loss,3))
                scheduler.step(test_loss)

            if True: #epoch%3 == 0:
                print("Training Set Eval:")
                train_score,train_score_shape = evaluateFullySupervisedCOCOwPrecisionRecall(model,train_loader,device,amp,120,130)
                train_scores.append(round(train_score.item(),3))
                train_scores_shapes.append(round(train_score_shape.item(),3))
                print("")

            print("Average loss this epoch = ",epoch_loss.item()/n_train) #pas dit nog aan eventueel
            train_losses.append(round(epoch_loss.item()/n_train,3))
            
            if save_checkpoint:
                Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
                state_dict = model.state_dict()
                state_dict['mask_values'] = dataset.mask_values
                torch.save(state_dict, str(dir_checkpoint / 'checkpoint_epoch{}.pth'.format(epoch)))
                logging.info(f'Checkpoint {epoch} saved!')
                print("/////////////////////////")

        print("Max test score:",round(max_test_score.item(),3),"found at epoch:",max_test_score_epoch)
        print("Max test score only shapes:",round(max_test_score_shape.item(),3),"found at epoch:",max_test_score_shape_epoch)
        # Calculate and print the median values for each epoch
        generalizationRatioList = []
        generalizationDifferenceList = []
        for epoch in range(len(test_scores_shapes)):
            if test_scores_shapes[epoch] != 0:
                generalizationRatioList.append(test_scores_shapes[epoch] / train_scores_shapes[epoch])
            generalizationDifferenceList.append(-test_scores_shapes[epoch] + train_scores_shapes[epoch])
            # Calculate and print the average and median of the generalizationRatioList
        avg_generalization_ratio = sum(generalizationRatioList) / len(generalizationRatioList)
        median_generalization_ratio = sorted(generalizationRatioList)[len(generalizationRatioList) // 2]
        avg_generatization_difference = sum(generalizationDifferenceList) / len(generalizationDifferenceList)
        median_generatization_difference = sorted(generalizationDifferenceList)[len(generalizationDifferenceList) // 2]

        with open(experimentFileName, "w") as f:
            f.write(writeInfo + "\n")
            f.write("Test Scores: " + str(test_scores) + "\n")
            f.write("Test Scores w/o background: " + str(test_scores_shapes) + "\n")
            f.write("Train Scores: " + str(train_scores) + "\n")
            f.write("Train Scores w/o background: " + str(train_scores_shapes) + "\n")
            f.write("Train Loss: " + str(train_losses) + "\n")
            if calc_test_loss:
                f.write("Test Loss: " + str(test_losses) + "\n")
            f.write(f"Max test score: {round(max_test_score.item(),3)} found at epoch: {max_test_score_epoch}\n")
            f.write(f"Max test score shapes: {round(max_test_score_shape.item(),3)} found at epoch: {max_test_score_shape_epoch}\n")
            f.write(f"Avg generalisation ratio: {round(avg_generalization_ratio,3)}\n")
            f.write(f"Avg generalisation diff: {round(avg_generatization_difference,3)}")


def diceLoss(mask_pred, true_mask, H, W, device, smooth=1.0):
    """
    Computes the Dice Loss for multi-class segmentation.
    
    Parameters:
    mask_pred (torch.Tensor): Predicted logits (before softmax) of shape [1, C, H, W]
    true_mask (torch.Tensor): Ground truth mask of shape [1, H, W] with class indices
    H (int): Height of the image
    W (int): Width of the image
    device (torch.device): Device to perform computations on
    smooth (float): Smoothing factor to prevent division by zero
    
    Returns:
    torch.Tensor: Dice loss value
    """
    # Apply softmax to obtain class probabilities
    mask_pred = F.softmax(mask_pred, dim=1)  # Shape: [1, C, H, W]
    
    # Convert true_mask to one-hot encoding
    C = mask_pred.shape[1]  # Number of classes
    true_mask_one_hot = F.one_hot(true_mask.long(), num_classes=C).permute(0, 3, 1, 2)  # Shape: [1, C, H, W]
    true_mask_one_hot = true_mask_one_hot.to(device, dtype=torch.float32)
    
    # Compute Dice coefficient per class
    intersection = torch.sum(mask_pred * true_mask_one_hot, dim=(2, 3))  # Sum over spatial dimensions
    union = torch.sum(mask_pred, dim=(2, 3)) + torch.sum(true_mask_one_hot, dim=(2, 3))
    dice_score = (2. * intersection + smooth) / (union + smooth)
    
    # Compute mean Dice loss across all classes
    dice_loss = 1 - dice_score.mean()
    
    return dice_loss


def cross_entropy(mask_pred, mask, H, W, device, smooth=1.0):
    # Convert logits to probabilities
    mask_pred = F.softmax(mask_pred, dim=1)  # (1, C, H, W)

    # Reshape weak label to match (H, W)
    mask = mask.view(H, W).long().to(device)  # Ensure it's the right shape and on the correct device

    # Flatten the predictions and labels
    mask_pred = mask_pred.permute(0, 2, 3, 1).contiguous().view(-1, mask_pred.shape[1])  # (H*W, C)
    mask = mask.view(-1)  # (H*W)

    # Calculate Cross Entropy Loss
    loss = F.cross_entropy(mask_pred, mask, reduction='mean')

    return loss

def get_args():
    #note: Batch size can be upped, but the images must be resized (scaled or padded) to have the same format!!
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=120, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=1, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-7,
                        help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--scale', '-s', type=float, default=1, help='Downscaling factor of the images')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--amp', action='store_true', default=True, help='Use mixed precision')
    parser.add_argument('--bilinear', action='store_true', default=True, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=3, help='Number of classes')
    parser.add_argument('--configuration', '-conf', dest='config', type=int, default=0, help='configuration id')
    parser.add_argument('--earlyStoppingAmount', '-earlystoppingAm', dest='earlystoppingAm', type=int, default=300, help='how many epochs without improvement to stop early')

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
            configuration=args.config,
            earlyStoppingAmount=args.earlystoppingAm
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
            configuration=args.config,
            earlyStoppingAmount=args.earlystoppingAm
        )

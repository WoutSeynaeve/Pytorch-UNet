import torch
import torch.nn.functional as F
from tqdm import tqdm
import torch.nn as nn

from utils.dice_score import multiclass_dice_coeff, dice_coeff

class_values = {
    "background": 0,
    "cylinder": 1,
    "sphere": 2,
    "cube": 3
}

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
            #print(mask_pred_class.unique(),mask_true.unique())
            assert(mask_true.shape == mask_pred_class.shape)
            # Compute IoU for each class
            for clss in range(num_classes):
                # Binary masks for the current class
                pred_mask = (mask_pred_class == clss)
                true_mask = (mask_true == clss)

                # Only compute IoU if the ground truth mask is not empty
                if true_mask.sum() > 0:
                    intersection = (pred_mask & true_mask).sum().float()
                    union = (pred_mask | true_mask).sum().float()


                    iou_per_class[clss] += intersection / union
                    valid_classes[clss] += 1

    # Warn about classes not present in the evaluation set
    missing_classes = [clss for clss in range(num_classes) if valid_classes[clss] == 0]
    if missing_classes:
        print(f"Warning: The following classes are not present in the evaluation set: {missing_classes}")

    weights = valid_classes/valid_classes.sum()

    meanIoUclasses = [iou_per_class[cl]/valid_classes[cl] for cl in range(num_classes)]
    print([meanIoUclass.item() for meanIoUclass in meanIoUclasses])
    totalWeightedMeanIoU = 0
    for cl in range(0,21):
        if meanIoUclasses[cl] >= 0 and meanIoUclasses[cl] <= 1 :
            totalWeightedMeanIoU += meanIoUclasses[cl]*weights[cl]

    weightsWithoutbackground = valid_classes/(valid_classes.sum()-valid_classes[0])
    totalWeightedMeanIoUWithoutBackground = 0
    for cl in range(1,21):
        if meanIoUclasses[cl] >= 0 and meanIoUclasses[cl] <= 1 :
            totalWeightedMeanIoUWithoutBackground += meanIoUclasses[cl]*weightsWithoutbackground[cl]
    checksumWeights = sum(weightsWithoutbackground[1:])
    if checksumWeights.item() != 1:
        print(checksumWeights.item()," should be = 1")
    print("eval withouth background",totalWeightedMeanIoUWithoutBackground)

    # Print per-class IoU for debugging or analysis

    # Restore model to training mode
    net.train()

    return totalWeightedMeanIoU
@torch.inference_mode()
def evaluateWeaklySupervisedCLEVR(net, dataloader, device, amp):
    net.eval()  # Set the model to evaluation mode
    num_classes = 4  # CLEVR

    # Initialize IoU accumulators
    valid_classes = torch.zeros(num_classes, device=device)  # Tracks the count of valid images per class
    iou_per_class = torch.zeros(num_classes, device=device)
    # Iterate over the validation set
    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        valsize = 0
        for batch in dataloader:
            valsize += 1
            image, weaklabel = batch['image'], batch["weaklabel"]
            image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            _, _, H, W = image.shape  # Extract height (H) and width (W)

            # Initialize the mask with background (0)
            # Predict the mask

            mask_pred = net(image)

            # Compute softmax probabilities and get class predictions
            mask_pred = F.softmax(mask_pred, dim=1)
            mask_pred_class = torch.argmax(mask_pred, dim=1)
            mask_pred_class.squeeze(0)
            # Flatten masks for easier processing
            bboxlist = weaklabel[0][4]
            indd = 0
            for i in bboxlist:
                indd += 1
                if indd % 2 != 0:
                    mask = torch.zeros((H, W), dtype=torch.long, device=device)
                    i = i[0]
                    i = i.split(',')
                    objecIndex = class_values[i[0]]
                    x1, x2, y1, y2 = map(int, i[1:5])
                    mask[y1:y2+1, x1:x2+1] = objecIndex

                    gt_mask = (mask == objecIndex)  # Ground truth mask for class
                    pred_mask = (mask_pred_class == objecIndex)  # Predicted mask for class

                    intersection = torch.sum(gt_mask & pred_mask)  # Logical AND
                    union = torch.sum(gt_mask | pred_mask)  # Logical OR

                    iou = intersection / union if union > 0 else torch.tensor(0.0, device=mask.device)
                    assert(iou <= 1)
                    iou_per_class[objecIndex] += iou
            mask = torch.zeros((H, W), dtype=torch.long, device=device)
            for i in bboxlist:
                i = i[0]
                i = i.split(',')
                objecIndex = class_values[i[0]]
                x1, x2, y1, y2 = map(int, i[1:5])
                mask[y1:y2+1, x1:x2+1] = objecIndex

            gt_mask = (mask == 0)  # Ground truth mask for class
            pred_mask = (mask_pred_class == 0)  # Predicted mask for class

            intersection = torch.sum(gt_mask & pred_mask)  # Logical AND
            union = torch.sum(gt_mask | pred_mask)  # Logical OR

            iou = intersection / union if union > 0 else torch.tensor(0.0, device=mask.device)
            assert(iou <= 1)
            iou_per_class[0] += iou
        for i in range(0,4):
            iou_per_class[i] /= valsize
            assert(iou_per_class[i] <= 1)
        print("val result",iou_per_class)
        net.train()
        return iou_per_class.mean()
@torch.inference_mode()
def evaluateWeaklySupervisedCLEVROLD(net, dataloader, device, amp):
    net.eval()  # Set the model to evaluation mode
    num_classes = 4  # CLEVR

    # Initialize IoU accumulators
    valid_classes = torch.zeros(num_classes, device=device)  # Tracks the count of valid images per class
    iou_per_class = torch.zeros(num_classes, device=device)
    # Iterate over the validation set
    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        valsize = 0
        for batch in dataloader:
            valsize += 1
            image, weaklabel = batch['image'], batch["weaklabel"]
            image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            _, _, H, W = image.shape  # Extract height (H) and width (W)

            # Initialize the mask with background (0)
            mask = torch.zeros((H, W), dtype=torch.long, device=device)
            # Predict the mask

            mask_pred = net(image)

            # Compute softmax probabilities and get class predictions
            mask_pred = F.softmax(mask_pred, dim=1)
            mask_pred_class = torch.argmax(mask_pred, dim=1)
            mask_pred_class.squeeze(0)
            # Flatten masks for easier processing
            bboxlist = weaklabel[0][4]
            for i in bboxlist:
                i = i[0]
                i = i.split(',')
                objecIndex = class_values[i[0]]
                x1, x2, y1, y2 = map(int, i[1:5])
                mask[y1:y2+1, x1:x2+1] = objecIndex

            

            for class_idx in range(0, 4):  # Exclude background (index 0)
                gt_mask = (mask == class_idx)  # Ground truth mask for class
                pred_mask = (mask_pred_class == class_idx)  # Predicted mask for class

                intersection = torch.sum(gt_mask & pred_mask)  # Logical AND
                union = torch.sum(gt_mask | pred_mask)  # Logical OR

                iou = intersection / union if union > 0 else torch.tensor(0.0, device=mask.device)
                assert(iou <= 1)
                iou_per_class[class_idx] += iou
        for i in range(0,4):
            iou_per_class[i] /= valsize
            assert(iou_per_class[i] <= 1)
        print("val result",iou_per_class)
        net.train()
        return iou_per_class.mean()
@torch.inference_mode()
def evaluateFullySupervisedCLEVRwPrecisionRecall(net, dataloader, device, amp):
    net.eval()  # Set the model to evaluation mode
    num_classes = 4  # CLEVR has 4 classes

    # Initialize accumulators
    valid_classes = torch.zeros(num_classes, device=device)  # Track class presence
    iou_per_class = torch.zeros(num_classes, device=device)
    precision_per_class = torch.zeros(num_classes, device=device)
    recall_per_class = torch.zeros(num_classes, device=device)

    # Iterate over the validation set
    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        valsize = 0  # Count the number of validation samples
        for batch in dataloader:
            valsize += 1
            image, true_mask = batch['image'], batch["mask"]
            image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            true_mask = true_mask.to(device=device)

            # Predict the mask
            mask_pred = net(image)

            # Compute softmax probabilities and get class predictions
            mask_pred = F.softmax(mask_pred, dim=1)
            mask_pred_class = torch.argmax(mask_pred, dim=1)  # Get predicted class per pixel

            # Compute metrics for each class
            for i in range(num_classes):
                pred_i = (mask_pred_class == i)
                true_i = (true_mask == i)

                intersection = torch.sum(pred_i & true_i).float()
                union = torch.sum(pred_i | true_i).float()
                
                tp = intersection  # True Positives
                fp = torch.sum(pred_i & ~true_i).float()  # False Positives
                fn = torch.sum(~pred_i & true_i).float()  # False Negatives

                if union > 0:
                    iou_per_class[i] += (intersection / union)
                    valid_classes[i] += 1  # Class i is present in this batch
                
                if (tp + fp) > 0:  # Avoid division by zero
                    precision_per_class[i] += (tp / (tp + fp))
                if (tp + fn) > 0:  # Avoid division by zero
                    recall_per_class[i] += (tp / (tp + fn))

        # Compute final metrics (average over valid classes)
        for i in range(num_classes):
            if valid_classes[i] > 0:
                iou_per_class[i] /= valid_classes[i]
                precision_per_class[i] /= valid_classes[i]
                recall_per_class[i] /= valid_classes[i]

        iouPerClass = [round(iou_per_class[i].item(), 3) for i in range(len(iou_per_class))]

        print(
            "IoU per class:", iouPerClass, 
            "mIoU:", round(iou_per_class.mean().item(), 3), 
            "mIoU_shapes:", round(iou_per_class[1:].mean().item(), 3)
        )
        # print("Validation Precision per class:", precision_per_class)
        # print("Validation Recall per class:", recall_per_class)

        net.train()
        return iou_per_class.mean(),iou_per_class[1:].mean()
@torch.inference_mode()
def evaluateFullySupervisedCOCOwPrecisionRecall(net, dataloader, device, amp,idmin,idmax):
    net.eval()  # Set the model to evaluation mode
    num_classes = 3 # CLEVR has 4 classes

    # Initialize accumulators
    valid_classes = torch.zeros(num_classes, device=device)  # Track class presence
    iou_per_class = torch.zeros(num_classes, device=device)
    precision_per_class = torch.zeros(num_classes, device=device)
    recall_per_class = torch.zeros(num_classes, device=device)

    # Iterate over the validation set
    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        valsize = 0  # Count the number of validation samples
        for batch in dataloader:
            valsize += 1
            image, true_mask, batch_id = batch['image'], batch["mask"], batch["id"]
            if batch_id[0] <= idmax and batch_id[0] >= idmin:
                image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                true_mask = true_mask.to(device=device)

                # Predict the mask
                mask_pred = net(image)

                # Compute softmax probabilities and get class predictions
                mask_pred = F.softmax(mask_pred, dim=1)
                mask_pred_class = torch.argmax(mask_pred, dim=1)  # Get predicted class per pixel

                # Compute metrics for each class
                for i in range(num_classes):
                    pred_i = (mask_pred_class == i)
                    true_i = (true_mask == i)

                    intersection = torch.sum(pred_i & true_i).float()
                    union = torch.sum(pred_i | true_i).float()
                    
                    tp = intersection  # True Positives
                    fp = torch.sum(pred_i & ~true_i).float()  # False Positives
                    fn = torch.sum(~pred_i & true_i).float()  # False Negatives

                    if union > 0:
                        iou_per_class[i] += (intersection / union)
                        valid_classes[i] += 1  # Class i is present in this batch
                    
                    if (tp + fp) > 0:  # Avoid division by zero
                        precision_per_class[i] += (tp / (tp + fp))
                    if (tp + fn) > 0:  # Avoid division by zero
                        recall_per_class[i] += (tp / (tp + fn))

        # Compute final metrics (average over valid classes)
        for i in range(num_classes):
            if valid_classes[i] > 0:
                iou_per_class[i] /= valid_classes[i]
                precision_per_class[i] /= valid_classes[i]
                recall_per_class[i] /= valid_classes[i]

        iouPerClass = [round(iou_per_class[i].item(), 3) for i in range(len(iou_per_class))]

        print(
            "IoU per class:", iouPerClass, 
            "mIoU:", round(iou_per_class.mean().item(), 3), 
            "mIoU_shapes:", round(iou_per_class[1:].mean().item(), 3)
        )
        # print("Validation Precision per class:", precision_per_class)
        # print("Validation Recall per class:", recall_per_class)

        net.train()
        return iou_per_class.mean(),iou_per_class[1:].mean()
            
@torch.inference_mode()
def evaluateFullySupervisedCLEVR(net, dataloader, device, amp):
    net.eval()  # Set the model to evaluation mode
    num_classes = 4  # CLEVR has 4 classes

    # Initialize IoU accumulators
    valid_classes = torch.zeros(num_classes, device=device)  # Track class presence
    iou_per_class = torch.zeros(num_classes, device=device)

    # Iterate over the validation set
    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        valsize = 0  # Count the number of validation samples
        for batch in dataloader:
            valsize += 1
            image, true_mask = batch['image'], batch["mask"]
            image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            true_mask = true_mask.to(device=device)

            # Predict the mask
            mask_pred = net(image)

            # Compute softmax probabilities and get class predictions
            mask_pred = F.softmax(mask_pred, dim=1)
            mask_pred_class = torch.argmax(mask_pred, dim=1)  # Get predicted class per pixel

            # Compute IoU for each class
            for i in range(num_classes):
                pred_i = (mask_pred_class == i)
                true_i = (true_mask == i)

                intersection = torch.sum(pred_i & true_i).float()
                union = torch.sum(pred_i | true_i).float()

                if union > 0:
                    iou_per_class[i] += (intersection / union)
                    valid_classes[i] += 1  # Class i is present in this batch

        # Average IoU over valid classes
        for i in range(num_classes):
            if valid_classes[i] > 0:
                iou_per_class[i] /= valid_classes[i]

        print("Validation IoU per class:", iou_per_class)
        net.train()
        return iou_per_class.mean()  # Return mean IoU over all classes
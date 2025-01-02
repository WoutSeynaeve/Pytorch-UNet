import os
from LogicLossVOC.LogicConstraints import adjacency, ifXthenXadjecent,atmost_p_percent_is_class_in_bounding_box, ifXthenYatRelation, scribble, image_level_label, about_p_percent_is_class, about_p_percent_is_class_in_bounding_box
import torch.nn.functional as F
import torch
import random
import numpy as np
class_values = {
    "background": 0,
    "aeroplane": 1,
    "bicycle": 2,
    "bird": 3,
    "boat": 4,
    "bottle": 5,
    "bus": 6,
    "car": 7,
    "cat": 8,
    "chair": 9,
    "cow": 10,
    "diningtable": 11,
    "dog": 12,
    "horse": 13,
    "motorbike": 14,
    "person": 15,
    "pottedplant": 16,
    "sheep": 17,
    "sofa": 18,
    "train": 19,
    "tvmonitor": 20
}

def outsideBoundingBoxes(output_tensor,bounding_boxes,configuration,printLosses):
    C,H,W = output_tensor.shape
    # for b in bounding_boxes:
    #     print(b)
    # Parse bounding box data
    partloss = 0
    class_masks = {}
    for bbox in bounding_boxes:
        # Parse class and bounding box coordinates
        class_name, x1, x2, y1, y2, _ = bbox[0].split(',')
        x1, x2, y1, y2 = map(int, [x1, x2, y1, y2])

        # Initialize mask for the class if not already present
        if class_name not in class_masks:
            class_masks[class_name] = torch.zeros((H, W), dtype=torch.bool)

        # Mark bounding box region as True for the class
        class_masks[class_name][y1:y2, x1:x2] = True  # Combine regions for the same class

    combined_bbox_mask = torch.zeros((H, W), dtype=torch.bool)

    # Combine all class-specific masks into a single mask
    for mask in class_masks.values():
        combined_bbox_mask |= mask  # Union of all bounding box regions

    # Get regions outside all bounding boxes (background region)
    background_mask = ~combined_bbox_mask  # Regions not covered by bounding boxes
    background_region = output_tensor[:, background_mask]

    # Check the intersection of background and non-background masks
    intersection_mask = combined_bbox_mask & background_mask  # Logical AND to find overlaps
    intersection_non_empty = intersection_mask.any()  # True if there are overlapping pixels

    if intersection_non_empty:
        print("Warning: Non-background and background regions intersect!")
    

    # Proceed with loss calculations
    # background_region1 = output_tensor[:, combined_bbox_mask]

    # partloss += about_p_percent_is_class(background_region1, [class_values['background']], 0, "single")

    # Non-bounding-box loss for each class
    non_bbox_regions = {}
    for class_name, mask in class_masks.items():
        # Invert the mask to get regions outside bounding boxes for the specific class
        inverted_mask = ~mask
        non_bbox_regions[class_name] = output_tensor[:, inverted_mask]
    if configuration[4][0]:
        addloss = about_p_percent_is_class(background_region, [class_values['background']], 1, "single")/configuration[4][1]
        partloss += addloss
        if printLosses:
            print("Loss for parts outside of bboxes to be background: ",addloss)
    
    if configuration[3][0]:
        for vv in non_bbox_regions.keys():
            addloss = about_p_percent_is_class(non_bbox_regions[vv],[class_values[vv]],0,"single")/configuration[3][1]
            partloss += addloss
            if printLosses:
                print("Loss for parts outside of bbox to be NOT for",vv," : ",addloss)
    return partloss

def read_dataset(file_path):
    with open(file_path, 'r') as file:
        data = file.readlines()
    return data

def parse_data(data):
    adjacency = []
    relations = []
    scribbles = []
    image_level = []
    bboxes = []

    for line in data:
        if line.startswith("Adjacency:"):
            adjacency.append(line.strip().split(": ")[1])
        elif line.startswith("Relation:"):
            relations.append(line.strip().split(": ")[1])
        elif line.startswith("Scribble:"):
            scribbles.append(line.strip().split(": ")[1])
        elif line.startswith("ImageLevel:"):
            image_level.append(line.strip().split(": ")[1])
        elif line.startswith("Bbox:"):
            bboxes.append(line.strip().split(": ")[1])
    
    return adjacency, relations, scribbles, image_level, bboxes


def calculateLogicLoss(output_tensor,weaklabels,printLosses = False):
   
    #             ImageLevelLoss, Adjacencies, BBoxObject, OutsideBBoxNotObject, BBoxBackground, Smoothness, Scribbles, Relations
    configuration = [[True,5],   [False,1] ,    [True,1,0.5],        [True,2] ,        [True,20],     [False,100], [True,1],  [False,1]]
    
    #             Scr. Objects, Scr. Background, Scr.NOT objects, Scr.NOT Background   
    ScribbleTypes = [[True,5],    [True,10],         [True,10],       [True,1]]   

    output_tensor = output_tensor[0,:,:,:]
    output_tensor = F.softmax(output_tensor, dim=0)
    adjacencies, relations, scribbles, image_level, bboxes = weaklabels[0]
    loss = 0
      
    #print(adjacencies, relations, scribbles, image_level,bboxes)
    if configuration[1][0]:
        for i in adjacencies:
            i = i[0]
            objects = i.split(',')
            addloss = adjacency(output_tensor, class_values[objects[0]], class_values[objects[1]])
            if addloss > 0.1:
                loss += addloss
                if printLosses:
                    print("loss for adjacency between",objects,": ",addloss)

    if configuration[7][0]:
        for i in relations:
            i = i[0]
            objects = i.split(',')
            X = objects[2]
            Y = objects[0]
            relation = objects[1]
            loss += ifXthenYatRelation(output_tensor, class_values[X], class_values[Y],relation)
    
    if configuration[6][0]:
        for i in scribbles:
            i = i[0]
            label = i.split(',')
            objectString = label[0]
            scribbleCoords = [(int(pair[0][1:]), int(pair[1][:-1])) for pair in zip(label[1:][::2], label[1:][1::2])]

            if objectString == "background":
                i = image_level[0][0]
                info = i.split(',')
                for objects in info[0::2]:
                    if objects != "background":
                        if ScribbleTypes[2][0]:
                            if len(scribbleCoords) != 0:
                                addloss = scribble(output_tensor,np.array(scribbleCoords),class_values[objects],"not")/ScribbleTypes[2][1]
                                if printLosses:
                                    print("loss for background scribble not being class",objects,": ",addloss.item())
                                loss += addloss
                    else:
                        if ScribbleTypes[1][0]:
                            if len(scribbleCoords) != 0:
                                addloss = scribble(output_tensor,np.array(scribbleCoords),class_values[objects])/ScribbleTypes[1][1]
                                if printLosses:
                                    print("loss for 'background scribble shoud be class (lowered loss)",objects,": ",addloss.item())
                                loss += addloss
            else:
                if len(scribbleCoords) != 0:
                    if ScribbleTypes[0][0]:
                        addloss = scribble(output_tensor,np.array(scribbleCoords),class_values[objectString])/ScribbleTypes[0][1]
                        if printLosses:
                            print("loss for scribbles for class",objectString,": ",addloss.item())
                        loss += addloss
                    if ScribbleTypes[3][0]:
                        addloss = scribble(output_tensor,np.array(scribbleCoords),class_values["background"],"not")
                        if printLosses:
                            print("loss for scribble of class",objectString,"should not be background",addloss.item())
                        loss += addloss
    if configuration[0][0]:
        for i in image_level:
            i = i[0]
            info = i.split(',')
            objects = info[0::2]
            percentages = info[1::2]
            for notObject in class_values.keys(): #IMAGELEVELLABEL NOT !
                    if not notObject in objects:
                        #loss += image_level_label(output_tensor,[class_values[notObject]],"not")
                        #alternative:
                        addloss= about_p_percent_is_class(output_tensor,[class_values[notObject]],0)/configuration[0][1]
                        if printLosses:
                            print("loss to not predict other classes in the image",addloss.item())
                        loss += addloss

            # for i in range(len(objects)):
            #     if objects[i] != "background":
            #         print([class_values[objects[i]]],int(percentages[i].replace('%',''))/100)
            #         loss += about_p_percent_is_class(output_tensor,[class_values[objects[i]]],int(percentages[i].replace('%',''))/100)
    
            # #I DONT THINK YOU WOULD EVEN WANT THIS:
            # for i in range(len(objects)):
            #     if objects[i] == "background":
            #         if random.random() <= 1:
            #             addloss = about_p_percent_is_class(output_tensor,[class_values[objects[i]]],int(percentages[i].replace('%',''))/100)
            #             if printLosses:
            #                 print("loss to predict x percentage background in the image",addloss.item())
            #             loss += addloss
        
    addloss = outsideBoundingBoxes(output_tensor,bboxes,configuration,printLosses)
    
    loss += addloss
    
    if configuration[2][0]:
        for i in bboxes:
            i = i[0]
            info = i.split(',')
            objectt = info[0]
            x1,x2,y1,y2 = info[1:-1]
            percentage = int(info[-1].replace('%',''))/100
            addloss = about_p_percent_is_class_in_bounding_box(output_tensor,[class_values[objectt]],percentage,int(x1),int(x2),int(y1),int(y2))/configuration[2][1]
            if printLosses:
                print("loss for boundingboxes for",objectt," : ",addloss.item())
            loss += addloss
            addloss = atmost_p_percent_is_class_in_bounding_box(output_tensor,[class_values['background']],1-percentage,int(x1),int(x2),int(y1),int(y2))/configuration[2][2]
            if addloss != 0:
                if printLosses:
                    print("loss for background to not take in boundingbox",objectt," : ",addloss.item())
                loss += addloss
            
    
    #GLOBAL SMOOTHNESS CONSTRAINT
    if configuration[5][0]:
        for i in image_level:
            i = i[0]
            info = i.split(',')
            objects = info[0::2]
            for i in objects:
                if loss != np.inf:
                    addloss = ifXthenXadjecent(output_tensor, class_values[i])/configuration[5][1]
                    if addloss > 0.1:
                        loss += addloss  
    # if loss.item() < 22:
    #     print(weaklabels[0][3],loss)
    return loss
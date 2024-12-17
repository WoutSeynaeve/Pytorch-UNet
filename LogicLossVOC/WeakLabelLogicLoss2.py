import os
from LogicLossVOC.LogicConstraints import adjacency, ifXthenXadjecent, ifXthenYatRelation, scribble, image_level_label, about_p_percent_is_class, about_p_percent_is_class_in_bounding_box
import torch.nn.functional as F
import torch
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

def outsideBoundingBoxesNotClass(output_tensor,bounding_boxes):
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

    # Extract regions outside bounding boxes for each class
    non_bbox_regions = {}
    for class_name, mask in class_masks.items():
        # Invert the mask to get regions outside bounding boxes
        inverted_mask = ~mask

        # Apply the inverted mask to the image tensor
        non_bbox_regions[class_name] = output_tensor[:, inverted_mask]
    for vv in non_bbox_regions.keys():
        partloss += about_p_percent_is_class(non_bbox_regions[vv],[class_values[vv]],0,"single")
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
    output_tensor = output_tensor[0,:,:,:]
    output_tensor = F.softmax(output_tensor, dim=0)
    adjacencies, relations, scribbles, image_level, bboxes = weaklabels[0]
    loss = 0
    #print(adjacencies, relations, scribbles, image_level,bboxes)
    for i in adjacencies:
        i = i[0]
        objects = i.split(',')
        #print(class_values[objects[0]],class_values[objects[1]])
        addloss = adjacency(output_tensor, class_values[objects[0]], class_values[objects[1]])
        if addloss > 1e-7:
            loss += addloss
        if printLosses:
            print("loss for adjacency",addloss)
        #loss += addloss
    for i in relations:
        i = i[0]
        objects = i.split(',')
        X = objects[2]
        Y = objects[0]
        relation = objects[1]
        #print(class_values[X],class_values[Y],relation)
        #loss += ifXthenYatRelation(output_tensor, class_values[X], class_values[Y],relation)
    for i in scribbles:
        i = i[0]
        label = i.split(',')
        objectString = label[0]
        scribbleCoords = [(int(pair[0][1:]), int(pair[1][:-1])) for pair in zip(label[1:][::2], label[1:][1::2])]
        #print(scribbleCoords,class_values[objectString])
        if len(scribbleCoords) != 0:
            addloss = scribble(output_tensor,np.array(scribbleCoords),class_values[objectString])
            if printLosses:
                print("loss for scribbles",addloss.item())
            loss += addloss

    for i in image_level:
        i = i[0]
        info = i.split(',')
        objects = info[0::2]
        percentages = info[1::2]

        for notObject in class_values.keys(): #IMAGELEVELLABEL NOT !
                if not notObject in objects:
                    #loss += image_level_label(output_tensor,[class_values[notObject]],"not")
                    """alternative: """
                    addloss= about_p_percent_is_class(output_tensor,[class_values[notObject]],0)/10
                    if printLosses:
                        print("loss to not predict other classes in the image",addloss.item())
                    loss += addloss
        """
        for i in range(len(objects)):
            #print([class_values[objects[i]]],int(percentages[i].replace('%','')))
            if objects[i] != "background":
                print([class_values[objects[i]]],int(percentages[i].replace('%',''))/100)
                loss += about_p_percent_is_class(output_tensor,[class_values[objects[i]]],int(percentages[i].replace('%',''))/100)
        """
        for i in range(len(objects)):
            #print([class_values[objects[i]]],int(percentages[i].replace('%','')))
            if objects[i] == "background":
                addloss = about_p_percent_is_class(output_tensor,[class_values[objects[i]]],int(percentages[i].replace('%',''))/100)/10
                if printLosses:
                    print("loss to predict x percentage background in the image",addloss.item())
                loss += addloss
    #"Outside of the bboxes for class I should not be class I"
    addloss = outsideBoundingBoxesNotClass(output_tensor,bboxes)
    if printLosses:
        print("loss for objects to not exist outside their bbox",addloss.item())
    loss += addloss
    for i in bboxes:
        i = i[0]
        info = i.split(',')
        objectt = info[0]
        x1,x2,y1,y2 = info[1:-1]
        percentage = int(info[-1].replace('%',''))/100
        #print(class_values[objectt],percentage,x1,x2,y1,y2)
        addloss = about_p_percent_is_class_in_bounding_box(output_tensor,[class_values[objectt]],percentage,int(x1),int(x2),int(y1),int(y2))
        if printLosses:
            print("loss for boundingboxes",addloss.item())
        loss += addloss
    #GLOBAL SMOOTHNESS CONSTRAINT
    globalsmoothness = False   #set to True to enable !!!
    if globalsmoothness:
        for i in range(0,21):
            loss += ifXthenXadjecent(output_tensor, i)

    #IMAGE LEVEL LABEL NOT for all classes which should not be in the image
    # --> to do !! 
    #but: Maybe this forces too much prediction of background?   

    return loss

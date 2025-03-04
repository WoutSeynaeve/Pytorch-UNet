import os
from LogicLossVOC.LogicConstraints import onehot,bounding_box_loss,onehot2,adjacency,atmost_p_percent_is_class,alteast_p_percent_is_class, ifXthenXadjecent,atmost_p_percent_is_class_in_bounding_box, ifXthenYatRelation, scribble_loss, image_level_label, about_p_percent_is_class, about_p_percent_is_class_in_bounding_box
import torch.nn.functional as F
import torch
import random
import numpy as np
class_values = {
    "background": 0,
    "cylinder": 1,
    "sphere": 2,
    "cube": 3
}
names_from_classes = {0: 'background', 1: 'cylinder', 2: 'sphere', 3: 'cube'}

def read_dataset(file_path):
    with open(file_path, 'r') as file:
        data = file.readlines()
    return data

def parse_dataCLEVR(data):
    image_level_label = []
    bounding_box = []
    point = []
    relation = []
    soft_relation = []
    scribble = []
    area = []
    adjacency = []
    full_bounding_box = []
    for line in data:
        if line.startswith("Image-level"):
            image_level_label.append(line.strip().split(",")[1:])
        elif line.startswith("BoundingBox"):
            bounding_box.append(line.strip().split(",")[1:])
        elif line.startswith("Point"):
            point.append(line.strip().split(",")[1:])
        elif line.startswith("Relation"):
            relation.append(line.strip().split(",")[1:])
        elif line.startswith("SoftRelation"):
            soft_relation.append(line.strip().split(",")[1:])
        elif line.startswith("Scribble"):
            scribble.append(line.strip().split(";"))
        elif line.startswith("Area"):
            area.append(line.strip().split(",")[1:])
        elif line.startswith("FullBoundingBox"):
            full_bounding_box.append(line.strip().split(",")[1:])
        elif line.startswith("Adjacency"):
            adjacency.append(line.strip().split(",")[1:])
        else:
            print("should not be here")
    return image_level_label, bounding_box, point, relation, soft_relation, scribble, area, full_bounding_box, adjacency


def calculateLogicLoss(output_tensor,weaklabels,configuration,printLosses = False):

    output_tensor = output_tensor[0, :, :, :]  # Remove batch dimension
    output_tensor = F.softmax(output_tensor, dim=0)  # Apply softmax over class dimension

    C,H,W = output_tensor.shape
    image_level_label, bounding_box, point, relation, soft_relation, scribble, area, full_bounding_box, adjacency = weaklabels[0]
    
    loss = 0

    #Image-level
    imglvl = configuration.get("ImageLevel")
    if imglvl[0]:
        for label in image_level_label:
            shape,percentage = label[0][0],label[1][0]
            addloss = about_p_percent_is_class(output_tensor,[class_values[shape]],float(percentage[:-1])/100)
            if printLosses:
                print("Loss for imageLevel label for shape",shape," to be percentage", percentage, " = ",addloss*imglvl[1])
            loss += addloss*imglvl[1]

    #Bounding Boxes
    bboxes = configuration.get("BBox")
    if bboxes[0][0]:
        for bbox in bounding_box:
            shape,x1,x2,y1,y2,percentage = bbox
            shape,x1,x2,y1,y2,percentage = shape[0],x1[0],x2[0],y1[0],y2[0],percentage[0]
            addloss = about_p_percent_is_class_in_bounding_box(output_tensor,[class_values[shape]],float(percentage[:-1])/100,int(x1),int(x2),int(y1),int(y2))
            
            if printLosses:
                print("loss for",shape," bounding box to be filled",percentage,"= ",addloss*bboxes[0][1])
            loss += addloss*bboxes[0][1]
            #implied constraint: "outside bounding box dont predict object"
            if bboxes[1][0]:
                addloss = 0
                if bboxes[3] == "linear":
                    if int(x1) > 1:
                        addloss += about_p_percent_is_class_in_bounding_box(output_tensor,[class_values[shape]],0,0,int(x1)-1,0,H)
                    if int(y1) > 1:
                        addloss += about_p_percent_is_class_in_bounding_box(output_tensor,[class_values[shape]],0,int(x1),int(x2),0,int(y1)-1)
                    if int(y2) < H:
                        addloss += about_p_percent_is_class_in_bounding_box(output_tensor,[class_values[shape]],0,int(x1),int(x2),int(y2)+1,H)
                    if int(x2) < W:
                        addloss += about_p_percent_is_class_in_bounding_box(output_tensor,[class_values[shape]],0,int(x2)+1,W,0,H)
                elif bboxes[3] == "prob":
                    if int(x1) > 1:
                        addloss += bounding_box_loss(output_tensor,0,int(x1)-1,0,H,class_values[shape],"not")
                    if int(y1) > 1:
                        addloss += bounding_box_loss(output_tensor,int(x1),int(x2),0,int(y1)-1,class_values[shape],"not")
                    if int(y2) < H:
                        addloss += bounding_box_loss(output_tensor,int(x1),int(x2),int(y2)+1,H,class_values[shape],"not")
                    if int(x2) < W:
                        addloss += bounding_box_loss(output_tensor,int(x2)+1,W,0,H,class_values[shape],"not")
                else:
                    print("should not be here")
                if printLosses:
                    print("loss for not predicting",shape, " outside its bounding box",addloss*bboxes[1][1])
                loss += addloss*bboxes[1][1]

            #Implied constraint: for each other class, it cannot take up more than 1-p percent of bbox
            if bboxes[2][0]:
                addloss = 0
                for i in range(4): 
                    if i != class_values[shape]:
                        addloss += atmost_p_percent_is_class_in_bounding_box(output_tensor,[i],1-float(percentage[:-1])/100,int(x1),int(x2),int(y1),int(y2))
                if printLosses:
                    print("loss for other classes to not take in too much of Bbox: ",addloss*bboxes[2][1])
                loss += addloss*bboxes[2][1]

    #one-hot global constraint
    onehot = configuration.get("OneHot")
    if onehot[0]:
        addloss = onehot2(output_tensor)
        if printLosses:
            print("oneHot loss = ",addloss*onehot[1])
        loss += addloss*onehot[1]
    
    #minimum size constraint for background class: atleast 70% is background
    minSizeGlobalBackground = configuration.get("MinSizeBackground")
    if minSizeGlobalBackground[0]:
        addloss = alteast_p_percent_is_class(output_tensor,[0],0.7)
        if printLosses:
            print("loss for atleast 70% to be background",addloss*minSizeGlobalBackground[1])
        loss += addloss*minSizeGlobalBackground[1]

    #smoothness global constraint
    smthns = configuration.get("Smoothness")
    if smthns[0]:
        for classes in range(4):
            addloss = ifXthenXadjecent(output_tensor,classes)
            if printLosses:
                print("loss for smoothness for class",names_from_classes[classes],addloss*smthns[1])
            loss += addloss*smthns[1]

    #minimum size global constraint: atleast 0.35% of the image is filled by each shape
    minSizeShapes = configuration.get("MinSizeShapes")
    if minSizeShapes[0]:
        for classes in range(1,4):
            addloss = alteast_p_percent_is_class(output_tensor,[classes],0.35/100)
            if printLosses:
                print("loss for minimum size constraint for:",names_from_classes[classes],addloss*minSizeShapes[1])
            loss += addloss*minSizeShapes[1]

    #FUll 100% bounding boxes
    bboxFull = configuration.get("BBoxFull")
    if bboxFull[0]:
        for bbox in full_bounding_box:
            shape,x1,x2,y1,y2 = bbox
            shape,x1,x2,y1,y2 = shape[0],x1[0],x2[0],y1[0],y2[0]
            if bboxFull[2] == "linear":
                addloss = about_p_percent_is_class_in_bounding_box(output_tensor,[class_values[shape]],1,int(x1),int(x2),int(y1),int(y2))
            elif bboxFull[2] == "prob":
                addloss = bounding_box_loss(output_tensor,int(x1),int(x2),int(y1),int(y2),class_values[shape],"all")
            else:
                print("should not be here")
            if printLosses:
                print("loss for",shape," bounding box to be FULLY filled","= ",addloss*bboxFull[1])
            loss += addloss*bboxFull[1]

    #maximum size global constraint for background: not more than 100-3*0.35 = 99.95 percent should be filled by background
    #true max value for train set is actually 97.98, so we can also use 98 maybe
    maxSizeBackground = configuration.get("MaxSizeBackground")
    if maxSizeBackground[0]:
        addloss = atmost_p_percent_is_class(output_tensor,[0],0.98)
        if printLosses:
            print("loss for atmost 98.95 to be background",addloss*maxSizeBackground[1])
        loss += addloss*maxSizeBackground[1] 

    #maximum size global constraint for shapes: each shape should not take in more than: 9.18 percent
    maxSizeShapes = configuration.get("MaxSizeShapes")
    if maxSizeShapes[0]:
        for classes in range(1,4):
            addloss = atmost_p_percent_is_class(output_tensor,[classes],9.18/100)
            if printLosses:
                print("loss for maximize size constraint for:",names_from_classes[classes],addloss*maxSizeShapes[1])
            loss += addloss*maxSizeShapes[1]

    #scribble
    scribbles = configuration.get('Scribbles')
    if scribbles[0]:
        for scribb in scribble:          
            first_entry = scribb[0][0].split(",")  # Split the first string
            shape = first_entry[1]  # Extract the shape name
            coords = [[int(first_entry[2]), int(first_entry[3])]]  # First coordinate

            # Convert the remaining coordinates to integers
            coords += [list(map(int, item[0].split(','))) for item in scribb[1:]]
            addloss = scribble_loss(output_tensor,coords,class_values[shape],"all")
            if printLosses:
                print("Loss for scribble for shape",shape," = ",addloss*scribbles[1])
            loss += addloss*scribbles[1]



    #area
    area = configuration.get('Area')

    #relation
    relations = configuration.get('Relations')

    #softRelation
    softrelations = configuration.get('SoftRelations')

    #point
    points = configuration.get('Point')

    #adjacency
    adjacencies = configuration.get("Adjacency")

    return loss





import os
from LogicLossVOC.LogicConstraints import atleast_p_percent_is_class_in_bounding_box,onehot,bounding_box_loss,onehot2,adjacency_loss,atmost_p_percent_is_class,atleast_p_percent_is_class, ifXthenXadjecent,atmost_p_percent_is_class_in_bounding_box, ifXthenYatRelation, scribble_loss, image_level_label, about_p_percent_is_class, about_p_percent_is_class_in_bounding_box
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


def calculateLogicLoss(output_tensor,weaklabels,configuration,batch_n,printLosses = False):
    
    output_tensor = output_tensor[0, :, :, :]  # Remove batch dimension
    output_tensor = F.softmax(output_tensor, dim=0)  # Apply softmax over class dimension

    C,H,W = output_tensor.shape
    image_level_label, bounding_box, point, relation, soft_relation, scribble, area, full_bounding_box, adjacency = weaklabels[0]
    
    loss = 0

    #Image-level
    imglvl = configuration.get("ImageLevel")
    imlvlpercdict = {'cylinder': image_level_label[1][1][0][:-1],'sphere': image_level_label[2][1][0][:-1],'cube': image_level_label[3][1][0][:-1]}
    if imglvl[0]:
        for label in image_level_label:
            shape,percentage = label[0][0],label[1][0]
            if shape != 'background':
                if imglvl[1]:
                    if imglvl[2]:
                        addloss = atleast_p_percent_is_class(output_tensor,[class_values[shape]],float(percentage[:-1])/100)
                    else:
                        addloss = about_p_percent_is_class(output_tensor,[class_values[shape]],float(percentage[:-1])/100)
                else:
                    #if you are not give exact percentages, assume minim 0.35 percent of image is filled
                    addloss = atleast_p_percent_is_class(output_tensor,[class_values[shape]],0.35/100)
                if printLosses:
                    print("loss for imageLevel label for shape",shape," to be percentage", percentage, " = ",addloss*imglvl[3])
                loss += addloss*imglvl[3]

    #Bounding Boxes
    bboxes = configuration.get("BBox")
    if bboxes[0][0]:
        for bbox in bounding_box:
            shape,x1,x2,y1,y2,percentage = bbox
            shape,x1,x2,y1,y2,percentage = shape[0],x1[0],x2[0],y1[0],y2[0],percentage[0]
            if bboxes[0][1]:
                addloss = about_p_percent_is_class_in_bounding_box(output_tensor,[class_values[shape]],float(percentage[:-1])/100,int(x1),int(x2),int(y1),int(y2))
            else:
                #if you are not using exact percentages, assume that atleast 60 percent is filled
                addloss = atleast_p_percent_is_class_in_bounding_box(output_tensor,[class_values[shape]],0.6,int(x1),int(x2),int(y1),int(y2))
            if printLosses:
                print("loss for",shape," bounding box to be filled",percentage,"= ",addloss*bboxes[0][2])
            loss += addloss*bboxes[0][2]
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
            print("loss onehot = ",addloss*onehot[1])
        loss += addloss*onehot[1]
    
    #minimum size constraint for background class: atleast 70% is background
    minSizeGlobalBackground = configuration.get("MinSizeBackground")
    if minSizeGlobalBackground[0]:
        addloss = atleast_p_percent_is_class(output_tensor,[0],0.7)
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
            addloss = atleast_p_percent_is_class(output_tensor,[classes],0.35/100)
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
    #you want to put the upper bound higher, because you want "around 9.18 to be okay", because now this loss is more easily satisfied when predicting a lot less than 9.18 percent
    #so lets take 15%
    maxSizeShapes = configuration.get("MaxSizeShapes")
    if maxSizeShapes[0]:
        for classes in range(1,4):
            addloss = atmost_p_percent_is_class(output_tensor,[classes],15/100)
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
    areainfo = configuration.get('Area')
    if areainfo[0]:
        for arealbl in area:
            ar1,ar2 = arealbl[1][0].split(";")
            shape = arealbl[0][0]
            addloss = 0
            if areainfo[1]: #use true percentages (assume they are given)
                minpercentage = float(imlvlpercdict.get(shape))
            else:
                minpercentage = 0.35

            if ar1 == 'left_half':
                if ar2 == 'top_half':
                    #top-left
                    minpercentage = 4*minpercentage
                    addloss += atleast_p_percent_is_class_in_bounding_box(output_tensor, [class_values[shape]], minpercentage / 100, 0, W // 2 - 1, 0, H // 2 - 1) * areainfo[3]
                    addloss += atmost_p_percent_is_class_in_bounding_box(output_tensor, [0], (100 - minpercentage) / 100, 0, W // 2 - 1, 0, H // 2 - 1)
                    addloss += about_p_percent_is_class_in_bounding_box(output_tensor, [class_values[shape]], 0, W // 2, W, 0, H)
                    addloss += about_p_percent_is_class_in_bounding_box(output_tensor, [class_values[shape]], 0, 0, W, H // 2, H)
                elif ar2 == 'bottom_half':
                    #bottom-left
                    minpercentage = 4*minpercentage
                    addloss += atleast_p_percent_is_class_in_bounding_box(output_tensor, [class_values[shape]], minpercentage / 100, 0, W // 2 - 1, H // 2, H) * areainfo[3]
                    addloss += atmost_p_percent_is_class_in_bounding_box(output_tensor, [0], (100 - minpercentage) / 100, 0, W // 2 - 1, H // 2, H)
                    addloss += about_p_percent_is_class_in_bounding_box(output_tensor, [class_values[shape]], 0, W // 2, W, 0, H)
                    addloss += about_p_percent_is_class_in_bounding_box(output_tensor, [class_values[shape]], 0, 0 , W, 0, H//2  - 1)
                else:
                    #left-half
                    minpercentage = 2*minpercentage
                    addloss += atleast_p_percent_is_class_in_bounding_box(output_tensor, [class_values[shape]], minpercentage/100, 0, W // 2 - 1, 0, H)*areainfo[3]
                    addloss += atmost_p_percent_is_class_in_bounding_box(output_tensor, [0], (100-minpercentage)/100, 0, W // 2 - 1, 0, H)
                    addloss += about_p_percent_is_class_in_bounding_box(output_tensor, [class_values[shape]], 0, W // 2, W, 0, H)
            elif ar1 == 'right_half':
                if ar2 == 'top_half':
                    #top-right
                    minpercentage = 4*minpercentage
                    addloss += atleast_p_percent_is_class_in_bounding_box(output_tensor, [class_values[shape]], minpercentage / 100, W // 2, W, 0, H // 2 - 1) * areainfo[3]
                    addloss += atmost_p_percent_is_class_in_bounding_box(output_tensor, [0], (100 - minpercentage) / 100, W // 2, W, 0, H // 2 - 1)
                    addloss += about_p_percent_is_class_in_bounding_box(output_tensor, [class_values[shape]], 0, 0, W // 2 - 1, 0, H)
                    addloss += about_p_percent_is_class_in_bounding_box(output_tensor, [class_values[shape]], 0, 0, W, H // 2, H)
                elif ar2 == 'bottom_half':
                    #bottom-right
                    minpercentage = 4*minpercentage
                    addloss += atleast_p_percent_is_class_in_bounding_box(output_tensor, [class_values[shape]], minpercentage / 100, W // 2, W, H // 2, H) * areainfo[3]
                    addloss += atmost_p_percent_is_class_in_bounding_box(output_tensor, [0], (100 - minpercentage) / 100, W // 2, W, H // 2, H)
                    addloss += about_p_percent_is_class_in_bounding_box(output_tensor, [class_values[shape]], 0, 0, W // 2 - 1, 0, H)
                    addloss += about_p_percent_is_class_in_bounding_box(output_tensor, [class_values[shape]], 0, 0 , W, 0, H//2 - 1)

                else:
                    #right-half
                    minpercentage = 2*minpercentage
                    addloss += atleast_p_percent_is_class_in_bounding_box(output_tensor, [class_values[shape]], minpercentage/100, W // 2, W, 0, H)*areainfo[3]
                    addloss += atmost_p_percent_is_class_in_bounding_box(output_tensor, [0], (100-minpercentage)/100, W // 2, W, 0, H)
                    addloss += about_p_percent_is_class_in_bounding_box(output_tensor, [class_values[shape]], 0, 0, W // 2 - 1, 0, H)
            else:
                if ar2 == 'top_half':
                    #top-half
                    minpercentage = 2*minpercentage
                    addloss += atleast_p_percent_is_class_in_bounding_box(output_tensor, [class_values[shape]], minpercentage/100, 0, W, 0, H // 2 - 1)*areainfo[3]
                    addloss += atmost_p_percent_is_class_in_bounding_box(output_tensor, [0], (100-minpercentage)/100, 0, W, 0, H // 2 - 1)
                    addloss += about_p_percent_is_class_in_bounding_box(output_tensor, [class_values[shape]], 0, 0, W, H // 2, H)
                elif ar2 == 'bottom_half':
                    #bottom-half
                    minpercentage = 2*minpercentage
                    addloss += atleast_p_percent_is_class_in_bounding_box(output_tensor, [class_values[shape]], minpercentage/100, 0, W, H // 2, H)*areainfo[3]
                    addloss += atmost_p_percent_is_class_in_bounding_box(output_tensor, [0], (100-minpercentage)/100, 0, W, H // 2, H)
                    addloss += about_p_percent_is_class_in_bounding_box(output_tensor, [class_values[shape]], 0, 0, W, 0, H // 2 - 1)

            if printLosses:
                print("Loss for shape",shape,"for",ar1,ar2," = ",addloss*areainfo[2])
            loss += addloss*areainfo[2]
            
    #relation
    relations = configuration.get('Relations')
    if relations[0]:
        for rel in relation:
            shape1,relat,shape2 = rel[0][0],rel[1][0],rel[2][0]
            l1 = ifXthenYatRelation(output_tensor, class_values[shape2], class_values[shape1], relat)*relations[1]
            if relations[2]: 
                rlLos = ifXthenYatRelation(output_tensor, class_values[shape1], class_values[shape2], relat,'not')
                l2 = rlLos*relations[3]

            if printLosses:
                print("loss for hard relation:",shape1,relat,shape2," = ",l1,'(soft)',l2,'(not)')
            tot_loss = l1 + l2
            loss += tot_loss
            

    #softRelation
    softrelations = configuration.get('SoftRelations')
    if softrelations[0]:
        for softrel in soft_relation:
            shape1,relat,shape2 = softrel[0][0],softrel[1][0],softrel[2][0]
            addloss = ifXthenYatRelation(output_tensor, class_values[shape2], class_values[shape1], relat[5:])
            if printLosses:
                print("loss for soft relation",shape1,relat,shape2,"=",addloss*softrelations[1])
            loss += addloss*softrelations[1]

    #point
    points = configuration.get('Point')
    if points[0]:
        for p in point:
            shape,x,y = p[0][0],int(p[1][0]),int(p[2][0])
            coord = [[x,y]]  
            addloss = scribble_loss(output_tensor,coord,class_values[shape],"all")
            if printLosses:
                print("loss for 1 point for shape",shape,' = ',addloss*points[1])
            loss += addloss*points[1]

    #adjacency
    adjacencies = configuration.get("Adjacency")
    if adjacencies[0]:
        presentAdjacencies = set()
        if len(adjacency) > 0:
            for adj in adjacency:
                shape1, shape2 = adj[0][0],adj[1][0]
                presentAdjacencies.add(tuple(sorted([class_values[shape1], class_values[shape2]])))
                addloss = adjacency_loss(output_tensor, class_values[shape1], class_values[shape2])
                if printLosses:
                    print("loss for adjacency between",shape1, shape2," = ",addloss*adjacencies[2])
                loss += addloss*adjacencies[2]

        #Not adjacent --> implied constraint
        if adjacencies[1]:
            for c1 in range(1,3):
                for c2 in range(c1+1,4):
                    if (c1,c2) not in presentAdjacencies:
                        addloss = adjacency_loss(output_tensor,c1,c2,'not')
                        if printLosses:
                            print('loss for NO adjacency between',names_from_classes[c1],names_from_classes[c2],"=",addloss*adjacencies[3])
                        loss += addloss*adjacencies[3]
    
    return loss





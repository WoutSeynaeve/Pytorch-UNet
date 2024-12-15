import os
from LogicLossVOC.LogicConstraints import adjacency, ifXthenXadjecent, ifXthenYatRelation, scribble, image_level_label, about_p_percent_is_class, about_p_percent_is_class_in_bounding_box
import torch.nn.functional as F
import torch

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


def calculateLogicLossD(output_tensor,weaklabels):
    #example file_path = '../../datasetExtraction/Dataset1/2011_002953.txt'
    output_tensor = output_tensor[0,:,:,:]
    output_tensor = F.softmax(output_tensor, dim=0)
    print(output_tensor,output_tensor.shape)
    adjacencies, relations, scribbles, image_level, bboxes = weaklabels[0]

    loss = 0
    #print(adjacencies, relations, scribbles, image_level,bboxes)
    for i in adjacencies:
        i = i[0]
        objects = i.split(',')
        #print(class_values[objects[0]],class_values[objects[1]])
        #loss += adjacency(output_tensor, class_values[objects[0]], class_values[objects[1]])
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
        #loss += scribble(output_tensor,scribbleCoords,class_values[objectString])
    for i in image_level:
        i = i[0]
        info = i.split(',')
        objects = info[0::2]
        percentages = info[1::2]
        for i in range(len(objects)):
            #print([class_values[objects[i]]],int(percentages[i].replace('%','')))
            loss += about_p_percent_is_class(output_tensor,[class_values[objects[i]]],int(percentages[i].replace('%',''))/100)
    for i in bboxes:
        i = i[0]
        info = i.split(',')
        objectt = info[0]
        x1,x2,y1,y2 = info[1:-1]
        percentage = int(info[-1].replace('%',''))/100
        #print(class_values[objectt],percentage,x1,x2,y1,y2)
        #loss += about_p_percent_is_class_in_bounding_box(output_tensor,[class_values[objectt]],percentage,x1,x2,y1,y2)
    
    #GLOBAL SMOOTHNESS CONSTRAINT
    globalsmoothness = False   #set to True to enable !!!
    if globalsmoothness:
        for i in range(0,21):
            loss += ifXthenXadjecent(output_tensor, i)
    
    #IMAGE LEVEL LABEL NOT for all classes which should not be in the image
    # --> to do !! 
    #but: Maybe this forces too much prediction of background?   

    return loss

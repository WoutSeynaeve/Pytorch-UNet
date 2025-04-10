import torch
import torch.nn.functional as F
import numpy as np

def image_level_label(normalized_tensor, I, NOT = None):

    class_I_probs = normalized_tensor[I]  #Select probabilities for class I
    class_I_probs = torch.clamp(class_I_probs, 0, 1 - 1e-7)

    log_probs = torch.log1p(-class_I_probs) #log probability of at least one pixel being class I

    if NOT:
        log_probability = log_probs.sum() #log probability of no pixel being class I
    else:
        log_probability = torch.log1p(-torch.exp(log_probs.sum()))  
    
    logicLoss = -log_probability

    return logicLoss


def bounding_box_loss(normalized_tensor,x1,x2,y1,y2,I, Option = None):
    class_probs = normalized_tensor[I, :, :]
    class_probs = torch.clamp(class_probs, 0, 1 - 1e-7)
    bbox_class_probs = class_probs[y1:y2+1, x1:x2+1]

    #print("Extracted bounding box probabilities: \n",bbox_class_probs)

    if Option == "all":
        log_probs = torch.log(bbox_class_probs) #log probability of pixel being class I
        log_probability = log_probs.sum() #log probability of all pixels being class I

    elif Option == "not":
        log_probs = torch.log1p(-bbox_class_probs)
        log_probability = log_probs.sum() #log probability of no pixel being class I
    else:
        log_probs = torch.log1p(-bbox_class_probs)
        log_probability = torch.log(1-torch.exp(log_probs.sum())) #log probability of at least one pixel being class I

    logicLoss = -log_probability

    return logicLoss

def scribble_loss(normalized_tensor, scribble_coords, target_class, Option = None):
    scribble_coords = np.array(scribble_coords)
    class_probs = normalized_tensor[target_class, :, :]
    class_probs = torch.clamp(class_probs, 1e-7, 1 - 1e-7)
    scribble_probs = class_probs[scribble_coords[:, 1], scribble_coords[:, 0]] 
    
    if Option == "none":
        log_probs = torch.log1p(-scribble_probs)
    elif Option == "all":
        log_probs = torch.log(scribble_probs)  #optional: add epsilon for stability

    log_probability = log_probs.sum()  

    logicLoss = -log_probability

    return logicLoss

def adjacency_loss(normalized_tensor, class_I, class_J,option = 'yes'):
    
    # Extract probabilities for class I and class J
    probs_I = normalized_tensor[class_I, :, :]  # Shape: (H, W)
    probs_J = normalized_tensor[class_J, :, :]  # Shape: (H, W)
    probs_I = torch.clamp(probs_I, min=1e-7, max=1-1e-7)
    probs_J = torch.clamp(probs_J, min=1e-7, max=1-1e-7)


    # Define adjacency kernel (3x3 neighborhood excluding center)
    adjacency_kernel = torch.tensor([[1, 1, 1],
                                     [1, 0, 1],
                                     [1, 1, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    # Compute log(1 - P_J)
    log_one_minus_probs_J = torch.log1p(-probs_J)
    adjacency_kernel = adjacency_kernel.to(log_one_minus_probs_J.dtype).to(log_one_minus_probs_J.device)

    # Convolve log(1 - P_J) with adjacency kernel to sum over neighbors
    log_sum_neighbors = F.conv2d(
        log_one_minus_probs_J.unsqueeze(0).unsqueeze(0),  # Add batch and channel dimensions
        adjacency_kernel,
        padding=1  # Ensure the output has the same spatial dimensions as input
    ).squeeze(0).squeeze(0)  # Remove batch and channel dimensions
    # Compute probabilities for no adjacencies
    pixelwise_adjacency = torch.exp(torch.log(probs_I) + torch.log1p(-torch.exp(log_sum_neighbors)))
    # Compute log(pixelwise_no_adjacency)
    log_pixelwise_no_adjacency = torch.log1p(-pixelwise_adjacency)

    # Sum over all pixels to compute log(global_no_adjacency)
    log_global_no_adjacency = torch.sum(log_pixelwise_no_adjacency)

    if option == 'not':
        log_probability = log_global_no_adjacency
    elif option == 'yes':
        log_probability = torch.log1p(-torch.exp(log_global_no_adjacency))
    else:
        print("invalid option for adjacency")

    logicLoss = -log_probability

    return logicLoss

def ifXthenYatRelation(normalized_tensor, X, Y, relation,NOT = None):
    device = normalized_tensor.device
    probs_I = normalized_tensor[X, :, :]  # Shape: (H, W)
    probs_J = normalized_tensor[Y, :, :]  # Shape: (H, W)
    probs_I = torch.clamp(probs_I, min=0, max=1-1e-6)
    probs_J = torch.clamp(probs_J, min=0, max=1-1e-6)
    scalingFactor = 10
    if scalingFactor > 1:
        start = np.random.randint(0, scalingFactor)  # Randomly choose 0, 1, or 2
        probs_I = probs_I[start::scalingFactor, start::scalingFactor]  # Apply offset
        probs_J = probs_J[start::scalingFactor, start::scalingFactor]

    # Preprocess based on the specified relation
    if relation == "left":
        new_probs_I = probs_I
        new_probs_J = probs_J
    elif relation == "right":
        new_probs_I = torch.flip(probs_I, dims=[1])  # Flip columns
        new_probs_J = torch.flip(probs_J, dims=[1])
    elif relation == "under":
        new_probs_I = torch.flip(probs_I.T, dims=[1])  # Transpose + flip horizontally
        new_probs_J = torch.flip(probs_J.T, dims=[1])
    elif relation == "above":
        new_probs_I = probs_I.T  # Transpose for rows/columns
        new_probs_J = probs_J.T
        
    else:
        raise ValueError("Invalid relation specified. Choose from 'left', 'right', 'above', 'under'.")

    H, W = new_probs_I.shape  # Shape of the output tensor
    dp_log = torch.zeros(W-1) 

    for n in range(0, W-1):
        # Log probability of no J in column n
        no_J_in_column_n = torch.log1p(-new_probs_J[:, n])
        
        # Update dp_log[n] using the sum of log probabilities (product of probabilities in original space)
        dp_log[n] = torch.sum(no_J_in_column_n)  # Log of probability of no dog in the n-th column
        
        # For columns before the last one, accumulate log probabilities (equivalent to multiplying probabilities)
        if n > 0:
            dp_log[n] += dp_log[n-1]  # Add previous column's log probability for cumulative effect

    prob_J_and_I = torch.zeros(W-1, dtype=normalized_tensor.dtype, device=normalized_tensor.device)

    for i in range(0,W-1):  # Loop through all but last column

        prob_no_j_in_0_to_ith_col = torch.exp(dp_log[i]) 
        #print("probability of there being atleast one J in the first", i+1, "collum(s)",1-prob_no_j_in_0_to_ith_col.item())
        logprob_I = torch.sum(torch.log1p((-new_probs_I[:, i+1])))
        #print("probability of there being atleast one I in collum", i+2, 1-torch.exp(logprob_I).item())
        if NOT:
            prob_J_and_I[i] = (1-torch.exp(logprob_I))*(1-prob_no_j_in_0_to_ith_col)
        else:
            prob_J_and_I[i] = (1-torch.exp(logprob_I))*(prob_no_j_in_0_to_ith_col)
        #print(f"probability of there being no J in the first {i+1} and atleast one I in the {i+2} collum", prob_J_and_I[i].item())

    if NOT:
        prob_J_and_I = torch.clamp(prob_J_and_I, max=1 - 1e-6) 
        logprobability_constraint = torch.sum(torch.log1p(-prob_J_and_I))
    else:
        logprobability_constraint = torch.sum(torch.log1p(-prob_J_and_I))+torch.sum(torch.log1p((-new_probs_I[:, 0])))
    logicLoss = -logprobability_constraint
    
    return logicLoss

    

def ifXthenXadjecent(normalized_tensor, class_I):
    # Extract probabilities for class I and class J
    probs_I = normalized_tensor[class_I, :, :]  # Shape: (H, W)
    probs_I = torch.clamp(probs_I, 1e-7, 1 - 1e-7)
    _, H, W = normalized_tensor.shape

    # Define adjacency kernel (3x3 neighborhood excluding center)
    adjacency_kernel = torch.tensor([[1, 1, 1],
                                     [1, 0, 1],
                                     [1, 1, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    # Compute log(1 - P_J)
    log_one_minus_probs_J = torch.log1p(-probs_I)
    adjacency_kernel = adjacency_kernel.to(log_one_minus_probs_J.dtype).to(log_one_minus_probs_J.device)

    # Convolve log(1 - P_J) with adjacency kernel to sum over neighbors
    log_sum_neighbors = F.conv2d(
        log_one_minus_probs_J.unsqueeze(0).unsqueeze(0),  # Add batch and channel dimensions
        adjacency_kernel,
        padding=1  # Ensure the output has the same spatial dimensions as input
    ).squeeze(0).squeeze(0)  # Remove batch and channel dimensions
    #probability of pixel being I and no adjecent pixel being I
    pixelwise_no_adjacency = torch.exp(torch.log(probs_I) + log_sum_neighbors)
    log_pixelwise_no_adjacency = torch.log1p(-pixelwise_no_adjacency)
    # Sum over all pixels to compute log(global_no_adjacency)
    log_probability = torch.sum(log_pixelwise_no_adjacency)
    logicLoss = -log_probability

    return logicLoss/(H*W)

    
def about_p_percent_is_class_in_bounding_box(normalized_tensor,classesList,p,x1,x2,y1,y2):
    bounding_box_tensor = normalized_tensor[:,y1:y2+1, x1:x2+1]
    return about_p_percent_is_class(bounding_box_tensor,classesList,p)

def atleast_p_percent_is_class_in_bounding_box(normalized_tensor,classesList,p,x1,x2,y1,y2):
    bounding_box_tensor = normalized_tensor[:,y1:y2+1, x1:x2+1]
    return atleast_p_percent_is_class(bounding_box_tensor,classesList,p)

def about_p_percent_is_class(normalized_tensor,classesList,p,single=None):
    assert(p <= 1)
    ExpectedPixels = 0
    for classs in classesList:
        ExpectedPixels += normalized_tensor[classs].sum()
    if single:
        _,totalPixels = normalized_tensor.shape
    else:
        _, H, W = normalized_tensor.shape
        totalPixels = H*W


    
    maxloss = 100
    NumberOfPixels = p*totalPixels
    loss = maxloss*torch.abs(ExpectedPixels-NumberOfPixels)/totalPixels #REMOVED SQUARE!!!!!
    return loss

def atleast_p_percent_is_class(normalized_tensor,classesList,p):
    assert(p <= 1)
    ExpectedPixels = 0
    for classs in classesList:
        ExpectedPixels += normalized_tensor[classs].sum()

    _, H, W = normalized_tensor.shape

    maxloss = 100

    totalPixels = H*W
    NumberOfPixels = p*totalPixels
    if ExpectedPixels >= NumberOfPixels:
        return 0
    else:
        loss = maxloss*torch.abs(ExpectedPixels-NumberOfPixels)/totalPixels
        return loss
    
def atmost_p_percent_is_class(normalized_tensor,classesList,p):
    assert(p <= 1)
    ExpectedPixels = 0
    for classs in classesList:
        ExpectedPixels += normalized_tensor[classs].sum()

    _, H, W = normalized_tensor.shape

    maxloss = 100

    totalPixels = H*W
    NumberOfPixels = p*totalPixels
    if ExpectedPixels <= NumberOfPixels:
        return 0
    else:
        loss = maxloss*torch.abs(ExpectedPixels-NumberOfPixels)/totalPixels
        return loss
    
def atmost_p_percent_is_class_in_bounding_box(normalized_tensor,classesList,p,x1,x2,y1,y2):
    bounding_box_tensor = normalized_tensor[:,y1:y2+1, x1:x2+1]
    return atmost_p_percent_is_class(bounding_box_tensor,classesList,p)


def onehot(normalized_tensor):
    C, H, W = normalized_tensor.shape  # Get tensor dimensions
    normalized_tensor = torch.clamp(normalized_tensor, min=1e-5, max=1-1e-5)
    results = []
    for i in range(C):
        new_probs = torch.zeros(C, H, W, device=normalized_tensor.device)  # Ensure device consistency
        for j in range(C):
            if i != j:
                new_probs[j] = torch.log(1-normalized_tensor[j])
            else:
                new_probs[j] = torch.log(normalized_tensor[j])
        result = new_probs.sum(dim=0)
        results.append(torch.log1p(-torch.exp(result)))
    
    summed_results = torch.stack(results).sum(dim=0)
    ll = -torch.log1p(-torch.exp(summed_results))
 
    return ll.sum()/(H*W)

def onehot2(normalized_tensor):
    C, H, W = normalized_tensor.shape  # Get tensor dimensions
    normalized_tensor = torch.clamp(normalized_tensor, min=1e-6, max=1-1e-6)
    result = 0
    for c1 in range(C):
        for c2 in range(C):
            if c1 != c2:
                result += torch.log1p(-torch.exp(torch.log(normalized_tensor[c1])+torch.log(normalized_tensor[c2])))
   
    sum_to_one_penalty = torch.log1p(-torch.exp(torch.log(1-normalized_tensor).sum()))
    result += sum_to_one_penalty
    
    return -result.sum()/(H*W)
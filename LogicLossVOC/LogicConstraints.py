import torch
import torch.nn.functional as F
import numpy as np

def dyn_progr(normalized_tensor, I, percentage, mode='exact'):
    _, H, W = normalized_tensor.shape
    class_probs = normalized_tensor[I, :, :]
    scalingFactor = 8
    if scalingFactor > 1:
        start = np.random.randint(0, scalingFactor)  # Randomly choose 0, 1, or 2
        class_probs = class_probs[start::scalingFactor, start::scalingFactor]  # Apply offset
    total_pixels = H * W
    target_pixels = int(percentage * total_pixels)

    # Initialize DP table
    dp = torch.zeros((total_pixels + 1,), device=normalized_tensor.device)
    dp[0] = 1

    # Flatten the probabilities
    flat_probs = class_probs.flatten()

    # Dynamic programming update
    for prob in flat_probs:
        dp_new = dp.clone()
        dp_new[1:] = dp[1:] * (1 - prob) + dp[:-1] * prob
        dp = dp_new

    if mode == 'exact':
        return dp[target_pixels]
    elif mode == 'atleast':
        return dp[target_pixels:].sum()
    elif mode == 'atmost':
        return dp[:target_pixels + 1].sum()
    else:
        raise ValueError("mode must be 'exact', 'atleast', or 'atmost'")
    


def newBounding_box(normalized_tensor, x1, x2, y1, y2, I):
    # Extract the probabilities of the bounding box and of class I
    class_probs = normalized_tensor[I, :, :]
    bbox_class_probs = class_probs[y1:y2, x1:x2]
    # Clamp probabilities to avoid numerical issues
    bbox_class_probs = torch.clamp(bbox_class_probs, min=1e-6, max=1-1e-6)
    # For each column, calculate the probability of there being no class I in that column
    no_class_I_in_columns = torch.log1p(-bbox_class_probs).sum(dim=0)

    # For each row, calculate the probability of there being no class I in that row
    no_class_I_in_rows = torch.log1p(-bbox_class_probs).sum(dim=1)

    # Probability that there is at least one class I in each column
    at_least_one_class_I_in_columns = 1 - torch.exp(no_class_I_in_columns)

    # Probability that there is at least one class I in each row
    at_least_one_class_I_in_rows = 1 - torch.exp(no_class_I_in_rows)

    # logProbability that for all of the columns there is at least one class I
    prob_all_columns_have_class_I = torch.sum(torch.log(at_least_one_class_I_in_columns))
    # logProbability that for all of the rows there is at least one class I
    prob_all_rows_have_class_I = torch.sum(torch.log(at_least_one_class_I_in_rows))
    return -(prob_all_columns_have_class_I+prob_all_rows_have_class_I)



def patch_level_label(normalized_tensor, I, patch_size=10):
    """
    For each center pixel, computes the probability that it and all neighbors in a patch_size x patch_size
    region are class I. Then returns a logic loss enforcing that such a patch must exist somewhere.

    Args:
        normalized_tensor: Tensor of shape (C, H, W)
        I: class index
        patch_size: size of the square patch (default: 5)

    Returns:
        A scalar logic loss
    """
    class_I_probs = normalized_tensor[I]  # (H, W)
    class_I_probs = torch.clamp(class_I_probs, min=1e-6, max=1.0 - 1e-6)

    # Work in log-space for numerical stability
    log_probs = torch.log(class_I_probs).unsqueeze(0).unsqueeze(0)  # shape: (1, 1, H, W)

    # Create kernel of ones to compute log-product (i.e., sum of logs)
    kernel = torch.ones((1, 1, patch_size, patch_size), dtype=torch.float32, device=class_I_probs.device)

    # No padding — only compute over valid center pixels with full neighborhood
    log_patch_product = F.conv2d(log_probs, kernel, padding=0)  # shape: (1, 1, H - k + 1, W - k + 1)
    patch_probs = torch.exp(log_patch_product).squeeze(0).squeeze(0)  # shape: (H-k+1, W-k+1)
    patch_probs = torch.clamp(patch_probs, min=1e-6, max=1.0 - 1e-6)

    # Compute the probability that no such patch exists (i.e., all are < full patch)
    no_patch_probs = 1.0 - patch_probs
    log_no_patch_probs = torch.log(no_patch_probs)
    log_total_no_patch = torch.sum(log_no_patch_probs)  # log of product over all (1 - patch_probs)
    prob_no_patch = torch.exp(log_total_no_patch)

    # Then the loss is: -log(1 - prob_no_patch) = encourage at least one full patch to exist
    prob_at_least_one_patch = 1.0 - prob_no_patch
    prob_at_least_one_patch = torch.clamp(prob_at_least_one_patch, min=1e-6, max=1.0)

    logic_loss = -torch.log(prob_at_least_one_patch)

    return logic_loss

def patch_level_label2(normalized_tensor, I):
    """
    For each pixel, computes the probability that itself and all 8-connected neighbors (3x3 patch) are class I.

    Args:
        normalized_tensor: Tensor of shape (C, H, W), where C is the number of classes.
        I: Class index to check for.

    Returns:
        Tensor of shape (H, W), where each value is the probability that the corresponding
        pixel and all its neighbors are class I.
    """
    class_I_probs = normalized_tensor[I]  # Shape: (H, W)
    class_I_probs = torch.clamp(class_I_probs, 1e-6, 1)

    # Prepare for convolution: reshape to (1, 1, H, W)
    input_tensor = class_I_probs.unsqueeze(0).unsqueeze(0)
    print(input_tensor)
    # Define a 3x3 kernel filled with ones to compute product over neighborhood
    kernel = torch.ones((1, 1, 5, 5), dtype=torch.float32, device=class_I_probs.device)

    # Apply convolution in log-space for numerical stability:
    log_input = torch.log(class_I_probs).unsqueeze(0).unsqueeze(0)  # shape (1, 1, H, W)
    log_sum = F.conv2d(log_input, kernel, padding=0)  # log of product over 3x3 patches
    print(log_sum)
    product = torch.exp(log_sum).squeeze()  # shape (H, W), probability that all 9 values are class I
    product = torch.clamp(product, 0, 1-1e-6)
    print(product)
    res1 = torch.log(1-product)
    res2 = res1.sum()
    print(res2)
    #-log (1-(exp sum log (1- exp sum log p, N(p) )
    return -torch.log1p(-torch.exp(res2))

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
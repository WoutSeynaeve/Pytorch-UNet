import logging
import os
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
from collections import Counter

from utils.data_loading import WeakLabelDatasetCLEVR
from unet import UNet
from utils.utils import plot_img_and_mask

# Define paths
input_dir = "../../datasetCLEVRaug/ImagesValidation"
output_dir = "outputPredictionsCLEVR"
os.makedirs(output_dir, exist_ok=True)

# Load the first 10 images from the directory
image_filenames = sorted(os.listdir(input_dir))[:20]
in_files = [os.path.join(input_dir, f) for f in image_filenames]
out_files = [os.path.join(output_dir, f"{os.path.splitext(f)[0]}_OUT.png") for f in image_filenames]

# Load model
#model_path = "./DebugCheckpoints/checkpoint_epoch1.pth"
model_path = "./checkpoints/checkpoint_epoch190.pth"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = UNet(n_channels=3, n_classes=4, bilinear=True)
net.to(device=device)

logging.info(f'Loading model {model_path}')
state_dict = torch.load(model_path, map_location=device)
mask_values = state_dict.pop('mask_values', [0, 1])
net.load_state_dict(state_dict)
logging.info('Model loaded!')

# Function to predict mask
def predict_img(net, full_img, device, scale_factor=1, out_threshold=0.5):
    net.eval()
    img = torch.from_numpy(WeakLabelDatasetCLEVR.preprocess(None, full_img, scale_factor, is_mask=False))
    img = img.unsqueeze(0).to(device=device, dtype=torch.float32)
    
    with torch.no_grad():
        output = net(img).cpu()
        output = F.interpolate(output, (full_img.size[1], full_img.size[0]), mode='bilinear')
        output = F.softmax(output, dim=1)
        mask = output.argmax(dim=1)
        max_probs, _ = output.max(dim=1)
        #mask[max_probs <= 0.2] = 0
    
    return mask[0].long().squeeze().numpy()

# Function to save mask
def mask_to_image(mask: np.ndarray):
    colors = [
        (0, 0, 0), (0, 0, 255), (0, 255, 0), (255, 0, 0)
    ]
    h, w = mask.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    for cls, color in enumerate(colors):
        out[mask == cls] = color
    return Image.fromarray(out)

# Process images
for i, filename in enumerate(in_files):
    logging.info(f'Predicting image {filename} ...')
    img = Image.open(filename)
    mask = predict_img(net, img, device)
    result = mask_to_image(mask)
    result.save(out_files[i])
    logging.info(f'Mask saved to {out_files[i]}')

import logging
import os
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np

from utils.data_loading import WeakLabelDatasetCLEVR
from unet import UNet
from utils.utils import plot_img_and_mask


#gif maker: https://ezgif.com/

# Define paths
input_dir = "../../datasetCLEVRaug/ImagesValidation"
output_dir = "outputMaskEvolutionsCLEVR"
checkpoint_dir = "./checkpoints"
tot_en_met_epoch = 228
os.makedirs(output_dir, exist_ok=True)

# Load the first image from the directory
# image_filenames = sorted(os.listdir(input_dir))[6]  #4th image from validation set
image_filenames = 'CLEVR_val_013355n.png'
image_filenames = 'CLEVR_val_014847n.png'
print(image_filenames)
in_file = os.path.join(input_dir, image_filenames)

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = UNet(n_channels=3, n_classes=4, bilinear=True)
net.to(device=device)

def load_model(epoch):
    model_path = os.path.join(checkpoint_dir, f"checkpoint_epoch{epoch}.pth")
    if os.path.exists(model_path):
        logging.info(f'Loading model {model_path}')
        state_dict = torch.load(model_path, map_location=device)
        net.load_state_dict(state_dict, strict=False)
        logging.info('Model loaded!')
    else:
        logging.warning(f'Model checkpoint {model_path} not found! Skipping.')

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
    return mask[0].long().squeeze().numpy()

# Function to save mask
def mask_to_image(mask: np.ndarray):
    colors = [(0, 0, 0), (0, 0, 255), (0, 255, 0), (255, 0, 0)]
    h, w = mask.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    for cls, color in enumerate(colors):
        out[mask == cls] = color
    return Image.fromarray(out)

# Process the single image for every checkpoint in increments of 2 epochs
img = Image.open(in_file)

for epoch in range(1, tot_en_met_epoch+1): 
    load_model(epoch)
    mask = predict_img(net, img, device)
    result = mask_to_image(mask)
    out_file = os.path.join(output_dir, f"{os.path.splitext(image_filenames[0])[0]}_epoch{epoch}.png")
    result.save(out_file)   
    logging.info(f'Mask saved to {out_file}')

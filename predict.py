import argparse
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from matplotlib import colormaps
from collections import Counter

from utils.data_loading import BasicDataset
from unet import UNet
from utils.utils import plot_img_and_mask

class_values = {
    0: "background",
    1: "aeroplane",
    2: "bicycle",
    3: "bird",
    4: "boat",
    5: "bottle",
    6: "bus",
    7: "car",
    8: "cat",
    9: "chair",
    10: "cow",
    11: "diningtable",
    12: "dog",
    13: "horse",
    14: "motorbike",
    15: "person",
    16: "pottedplant",
    17: "sheep",
    18: "sofa",
    19: "train",
    20: "tvmonitor"
}


def predict_img(net,
                full_img,
                device,
                scale_factor=1,
                out_threshold=0.5):
    net.eval()
    img = torch.from_numpy(BasicDataset.preprocess(None, full_img, scale_factor, is_mask=False))
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)
    with torch.no_grad():
        output = net(img).cpu()
        output = F.interpolate(output, (full_img.size[1], full_img.size[0]), mode='bilinear')

        #remove background: output = output[:, 1:, :, :]
        #softmax for probs: output = F.softmax(output, dim=1)
        if net.n_classes > 1:
            mask = output.argmax(dim=1)    
        else:
            mask = torch.sigmoid(output) > out_threshold

    return mask[0].long().squeeze().numpy()


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model', '-m', default='MODEL.pth', metavar='FILE',
                        help='Specify the file in which the model is stored')
    parser.add_argument('--input', '-i', metavar='INPUT', nargs='+', help='Filenames of input images', required=True)
    parser.add_argument('--output', '-o', metavar='OUTPUT', nargs='+', help='Filenames of output images')
    parser.add_argument('--viz', '-v', action='store_true',
                        help='Visualize the images as they are processed')
    parser.add_argument('--no-save', '-n', action='store_true', help='Do not save the output masks')
    parser.add_argument('--mask-threshold', '-t', type=float, default=0.5,
                        help='Minimum probability value to consider a mask pixel white')
    parser.add_argument('--scale', '-s', type=float, default=1, #changed from 0.5
                        help='Scale factor for the input images')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')
    
    return parser.parse_args()


def get_output_filenames(args):
    def _generate_name(fn):
        return f'{os.path.splitext(fn)[0]}_OUT.png'

    return args.output or list(map(_generate_name, args.input))


def mask_to_image(mask: np.ndarray, mask_values):
    # If the mask is one-hot encoded, convert it to class indices
    if mask.ndim == 3:
        mask = np.argmax(mask, axis=0)

    # Count class frequencies in the mask
    class_counts = Counter(mask.flatten())
    top_classes = class_counts.most_common(5)
    print(top_classes)
    # Print top 5 classes
    print("Top 5 classes in the image (class: count):")
    for cls, count in top_classes:
        print(f"Class {class_values[cls]}: {count} pixels")

    manual_colors = [
    (0, 0, 0),  # Class 0: Background (black)
    (255, 0, 0),  # Class 1: Red
    (0, 255, 0),  # Class 2: Green
    (0, 0, 255),  # Class 3: Blue
    (255, 255, 0),  # Class 4: Yellow
    (255, 0, 255),  # Class 5: Magenta
    (0, 255, 255),  # Class 6: Cyan
    (128, 128, 128),  # Class 7: Gray
    (255, 165, 0),  # Class 8: Orange
    (75, 0, 130),  # Class 9: Indigo
    (255, 20, 147),  # Class 10: DeepPink
    (0, 128, 0),  # Class 11: DarkGreen
    (0, 0, 139),  # Class 12: DarkBlue
    (255, 99, 71),  # Class 13: Tomato
    (160, 82, 45),  # Class 14: Sienna
    (255, 255, 240),  # Class 15: Ivory
    (0, 255, 127),  # Class 16: SpringGreen
    (255, 228, 196),  # Class 17: Bisque
    (255, 218, 185),  # Class 18: PeachPuff
    (255, 240, 245),  # Class 19: LavenderBlush
    (50, 80, 110),  
    ]

    # Ensure there are enough colors for all classes (21 classes)

    # Map class labels to RGB colors
    mask_values.pop()
    color_map = {cls: manual_colors[i] for i, cls in enumerate(mask_values)}

    # Create an RGB output array
    h, w = mask.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)

    # Assign colors to each class in the mask
    for cls, color in color_map.items():
        out[mask == cls] = color

    return Image.fromarray(out)

if __name__ == '__main__':
    args = get_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    in_files = args.input
    out_files = get_output_filenames(args)

    net = UNet(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Loading model {args.model}')
    logging.info(f'Using device {device}')

    net.to(device=device)
    state_dict = torch.load(args.model, map_location=device)
    mask_values = state_dict.pop('mask_values', [0, 1])
    net.load_state_dict(state_dict)

    logging.info('Model loaded!')

    for i, filename in enumerate(in_files):
        logging.info(f'Predicting image {filename} ...')
        img = Image.open(filename)

        mask = predict_img(net=net,
                           full_img=img,
                           scale_factor=args.scale,
                           out_threshold=args.mask_threshold,
                           device=device)

        if not args.no_save:
            out_filename = out_files[i]
            result = mask_to_image(mask, mask_values)
            result.save(out_filename)
            logging.info(f'Mask saved to {out_filename}')

        if args.viz:
            logging.info(f'Visualizing results for image {filename}, close to continue...')
            plot_img_and_mask(img, mask)

import torch

net = torch.hub.load('milesial/Pytorch-UNet', 'unet_carvana', pretrained=True, scale=0.5)

# Save the model
torch.save(net.state_dict(), 'MODEL.pth')

#python predict.py -i ../../datasetPascalVOC/JPEGImages/2007_001834.jpg -o output.jpg --classes 22 -m './checkpoints/checkpoint_epoch1.pth'

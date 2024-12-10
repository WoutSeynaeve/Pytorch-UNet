import torch

net = torch.hub.load('milesial/Pytorch-UNet', 'unet_carvana', pretrained=True, scale=0.5)

# Save the model
torch.save(net.state_dict(), 'MODEL.pth')

#python predict.py -i image.jpg -o output.jpg -m './checkpoints/checkpoint_epoch1.pth' --classes 22

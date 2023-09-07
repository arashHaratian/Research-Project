import torch

test = torch.hub.load('facebookresearch/dino:main', 'dino_resnet50')
vicreg = torch.hub.load('facebookresearch/vicreg:main', 'resnet50')
test = torch.load("dinov2_vits14_linear4_head.pth")

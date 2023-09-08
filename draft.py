import torch
import torchvision.datasets as torch_datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
import torch.nn.functional as F


dino = torch.hub.load('facebookresearch/dino:main', 'dino_resnet50')
vicreg = torch.hub.load('facebookresearch/vicreg:main', 'resnet50')
# dinov2 =torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14')
# dinov2 =torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
dinov2.eval()
dinov2.cuda()
# dinov2 = torch.load("dinov2_vits14_linear4_head.pth")



data = torch_datasets.CelebA("./dataset", download=True, transform=ToTensor()) # if not working, try download files manually off the internet (mostly works in the morning XD)
data = DataLoader(data)
sample_data = next(iter(data))[0]
# plt.imshow(sample_data.squeeze(0).permute(1, 2, 0))
# plt.show()
# plt.close()


# vicreg(sample_data)
# dino(sample_data)

padded_sample_data = F.pad(sample_data)
padded_sample_data = F.pad(sample_data, (2, 2, 3, 3)).to("cuda") # Images should be multiple of 14
dinov2(padded_sample_data)
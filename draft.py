import torch
import torchvision.datasets as torch_datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
import torch.nn.functional as F
from RankMe import RankMe
from ignite.metrics import FID, InceptionScore


dino = torch.hub.load('facebookresearch/dino:main', 'dino_resnet50')
vicreg = torch.hub.load('facebookresearch/vicreg:main', 'resnet50')
# dinov2 =torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').eval().cuda()  ## Small network
## dinov2 = torch.load("dinov2_vits14_linear4_head.pth")



# data = torch_datasets.CelebA("./dataset", download=True, transform=ToTensor()) # if not working, try download files manually off the internet (mostly works in the morning XD)
data = torch_datasets.CelebA("./dataset", download=False, transform=ToTensor()) 
# data = torch_datasets.SUN397("./dataset", download=True, transform=ToTensor()) 
data_loader = DataLoader(data, 32)
data_iter = iter(data_loader)

sample_data = next(data_iter)[0]
plt.imshow(sample_data[0,:,:,:].permute(1, 2, 0))
plt.show()
plt.close()


# vicreg(sample_data)
# dino(sample_data)
# padded_sample_data = F.pad(sample_data, (2, 2, 3, 3)).to("cuda") # Images should be multiple of 14
# dinov2(padded_sample_data)

x = RankMe()(vicreg(sample_data).detach())
y = RankMe()(dino(sample_data).detach())

torch.asarray(x).shape

# torch.cat(torch.empty([1,]),x.reshape([1,]))  
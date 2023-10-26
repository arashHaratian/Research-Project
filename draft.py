import torch
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
import torch.nn.functional as F
from RankMe import RankMe
from ignite.metrics import FID, InceptionScore
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.fid import NoTrainInceptionV3



dino = torch.hub.load('facebookresearch/dino:main', 'dino_resnet50')
vicreg = torch.hub.load('facebookresearch/vicreg:main', 'resnet50')
# dinov2 =torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').eval().cuda()  ## Small network
## dinov2 = torch.load("dinov2_vits14_linear4_head.pth")
# inception = NoTrainInceptionV3(name="inception-v3-compat", features_list=["logits_unbiased"])



data = datasets.CelebA("./dataset", download=True, split='all', transform=ToTensor()) # if not working, try download files manually off the internet (mostly works in the morning XD)
# data = datasets.SUN397("./dataset", download=True, transform=ToTensor()) 
data_loader = DataLoader(data, 256)
data_iter = iter(data_loader)

sample_data = next(data_iter)[0]
plt.imshow(sample_data[0,:,:,:].permute(1, 2, 0))
plt.show()
plt.close()


# vicreg(sample_data)
# dino(sample_data)
# padded_sample_data = F.pad(sample_data, (2, 2, 3, 3)).to("cuda") # Images should be multiple of 14
# dinov2(padded_sample_data)

# x = RankMe(vicreg).evaluate_with_size(sample_data)
# y = RankMe(dino)(sample_data)




target_idx = 20 # male attribute

male_data = []
female_data = []

for i, (x, y) in enumerate(data_loader):
    m_idxs = y[:,target_idx] == 0
    f_idxs =  ~m_idxs
    male_data.append(x[m_idxs, :, :, :])
    female_data.append(x[f_idxs, :, :, :])

    if i == 2:
        break
 

male_data_tensor = torch.cat(male_data[0:7]).type(torch.uint8)
female_data_tensor = torch.cat(female_data[0:25]).type(torch.uint8)



fid = FrechetInceptionDistance()

fid.update(male_data_tensor, True)
fid.update(female_data_tensor, False)
fid.compute()



male_data_tensor = torch.cat(male_data[0:5])
female_data_tensor = torch.cat(female_data[0:5])

data_loader = DataLoader(data, 113)
data_iter = iter(data_loader)
both_data_tensor = next(data_iter)[0]

x = RankMe(vicreg)(male_data_tensor)
y = RankMe(vicreg)(female_data_tensor)
z = RankMe(vicreg)(both_data_tensor)

x = RankMe(dino)(male_data_tensor)
y = RankMe(dino)(female_data_tensor)
z = RankMe(dino)(both_data_tensor)
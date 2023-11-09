import torch
import torch.nn.functional as F
from torchvision import datasets
from torchvision.transforms import ToTensor, Compose
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from RankMe import RankMe
from torchmetrics.image.inception import InceptionScore
from torchmetrics.image.fid import NoTrainInceptionV3

DEVICE = "cuda"

def generate_bias_data(data_loader, target_attr = 20, early_break = None):
    target_data = []
    nontarget_data = []

    for i, (x, y) in enumerate(data_loader):
        t_idxs = torch.any(y[:,target_attr] == 1, 1)
        nt_idxs =  ~t_idxs
        target_data.append(x[t_idxs, :, :, :])
        nontarget_data.append(x[nt_idxs, :, :, :])

        if early_break:
            if i == early_break:
                print(f"\nFinished generating after {i} iterations\n")
                break
    
    target_data = torch.cat(target_data)
    nontarget_data = torch.cat(nontarget_data)
    
    return target_data, nontarget_data



## Loading the reperesentation network
vicreg = torch.hub.load('facebookresearch/vicreg:main', 'resnet50')
inception = NoTrainInceptionV3(name="inception-v3-compat", features_list=["logits_unbiased"])

## Loading the dataset
data = datasets.CIFAR10("./dataset", download=True, transform=ToTensor(),
                         target_transform=lambda x: F.one_hot(torch.Tensor([x]).to(torch.int64), 10).squeeze(0)) 
data_loader = DataLoader(data, 256)


## Create biased data
target_idx = torch.randint(0, 10, (5,))
male_data, female_data = generate_bias_data(data_loader, target_idx)



## Making sample sizes equal 
male_size = male_data.shape[0]
female_size = female_data.shape[0]
min_size = min(male_size, female_size)
male_data = male_data[:min_size,:, :, :].to(DEVICE)
female_data = female_data[:min_size,:, :, :].to(DEVICE)


## Creating unbiased data
data_loader = DataLoader(data, batch_size = min_size, shuffle = True)
data_iter = iter(data_loader)
unbiased_data = next(data_iter)[0]


## Setting the device of the tensors
vicreg =vicreg.to(DEVICE)
inception =inception.to(DEVICE)
unbiased_data = unbiased_data.to(DEVICE)
female_data = female_data.to(DEVICE)
male_data = male_data.to(DEVICE)


## Creating the RankMe objects
rankme_general = RankMe(vicreg, DEVICE)
rankme_male = RankMe(vicreg, DEVICE)
rankme_female = RankMe(vicreg, DEVICE)

## Evaluating RankMe on the data
rankme_general(unbiased_data, save_eval=True)
rankme_male(male_data, save_eval=True)
rankme_female(female_data, save_eval=True)


rankme_general = RankMe(inception, DEVICE)
rankme_male = RankMe(inception, DEVICE)
rankme_female = RankMe(inception, DEVICE)

# rankme_general(unbiased_data.type(torch.uint8), True)
# rankme_male(male_data.type(torch.uint8), True)
# rankme_female(female_data.type(torch.uint8), True)

# --------------- FID with vicreg --------------------------
from ignite.metrics import FID, InceptionScore
from ignite.engine import Engine

metrics = FID(2048, vicreg)

def eval_step(engine, batch):
    return batch

default_evaluator = Engine(eval_step)
metrics.attach(default_evaluator, "fid")

state = default_evaluator.run([[male_data, female_data]])
print(state.metrics["fid"])

# ------------------- FID with inceptionV3 network ---------------------

from torchmetrics.image.fid import FrechetInceptionDistance
fid = FrechetInceptionDistance()

fid.update(male_data.type(torch.uint8), True)
fid.update(female_data.type(torch.uint8), False)
fid.compute()


# --------------- Inception network -----------------------
inception = InceptionScore(vicreg)
inception.update(unbiased_data)
inception.compute()
inception = InceptionScore(vicreg)
inception.update(male_data)
inception.compute()
inception = InceptionScore(vicreg)
inception.update(female_data)
inception.compute()
# inception(female_data)

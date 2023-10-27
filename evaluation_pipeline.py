import torch
import torch.nn.functional as F
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from ignite.engine import Engine
from ignite.metrics import FID
from torchmetrics.image.inception import InceptionScore
from torchmetrics.image.fid import FrechetInceptionDistance
from RankMe import RankMe
import gc 
from datetime import datetime
from os import path

DEVICE = "cpu"

if path.isfile("pipeline_results.txt"):
    result_file = open("pipeline_results.txt", "a")
else:
    result_file = open("pipeline_results.txt", "a+")
    result_file.write("date,target_rankme,nontarget_rankme,unbiased_rankme,target_inception,nontarget_inception,unbiased_inception,target_nontarget_fid,target_unbiased_fid,nontarget_unbiased_fid\n")


def generate_bias_data(data_loader, target_attr = 20, early_break = None):
    target_data = []
    nontarget_data = []

    for i, (x, y) in enumerate(data_loader):
        t_idxs = torch.any(y[:,target_attr] == 1, 1)
        nt_idxs =  ~t_idxs
        target_data.append(x[t_idxs, :, :, :])
        nontarget_data.append(x[nt_idxs, :, :, :])
        if early_break:
            if early_break == i:
                print(f"\nFinished generating after {i} iterations\n")
                break
    
    target_data = torch.cat(target_data)
    nontarget_data = torch.cat(nontarget_data)
    
    return target_data, nontarget_data



## Loading the reperesentation network
# vicreg = torch.hub.load('facebookresearch/vicreg:main', 'resnet50')
vicreg = torch.hub.load('facebookresearch/dino:main', 'dino_resnet50')


torch.manual_seed(5)

## Create biased data

# if path.isfile("target_data.pt"):
#     ## Load the saved tensors
#     target_data = torch.load('target_data.pt')
#     nontarget_data = torch.load('nontarget_data.pt')
# else:
#     ## Loading the dataset
#     data = datasets.CelebA("./dataset", download=True, split='all', transform=ToTensor()) # if not working, try download files manually off the internet (mostly works in the morning XD)
#     data_loader = DataLoader(data, 256, shuffle=True)
#     target_idx = 20 # male attribute
#     target_data, nontarget_data = generate_bias_data(data_loader, target_idx, 500)
#     torch.save(target_data, 'target_data.pt')
#     torch.save(nontarget_data, 'nontarget_data.pt')

data = datasets.CelebA("./dataset", download=True, split='all', transform=ToTensor()) # if not working, try download files manually off the internet (mostly works in the morning XD)
data_loader = DataLoader(data, 256, shuffle=True)
# target_idx = torch.Tensor([20]).type(torch.int) # male attribute
target_idx = torch.Tensor([3, 7, 13, 20]).type(torch.int)
target_data, nontarget_data = generate_bias_data(data_loader, target_idx, 500)



## Make the batch sizes a common divider of 256 and save the data
min_size = 51200
target_data = target_data[:min_size,:, :, :]
nontarget_data = nontarget_data[:min_size,:, :, :]

torch.save(target_data, 'target_data.pt')
torch.save(nontarget_data, 'nontarget_data.pt')




## Setting the device of the models
vicreg = vicreg.to(DEVICE)

## Creating the RankMe objects
rankme_general = RankMe(vicreg, 256, DEVICE)
rankme_target = RankMe(vicreg, 256, DEVICE)
rankme_nontarget = RankMe(vicreg, 256, DEVICE)

## Creating FID metric and needed function
metrics = FID(2048, vicreg)
def eval_step(engine, batch):
    ## The engine needed for ignite 
    return batch

def inifite_iterator(data_loaders):
    ## The data to be passed to the engine
    for x, y in zip(data_loaders[0],data_loaders[1]):
        yield [x, y]

default_evaluator = Engine(eval_step)
metrics.attach(default_evaluator, "fid")

## =============================================================== BIASED DATA =========================================================


## Evaluating RankMe on the biased data 
gc.collect()
target_rankme_val = rankme_target(target_data, save_eval = True)
nontarget_rankme_val = rankme_nontarget(nontarget_data, save_eval = True)


## Checking of rankme results are reasonable
## We expect to have an increasing score as we get more data
# rankme_target.evaluate_with_size(target_data, torch.arange(0, min_size, 256),True)
# rankme_nontarget.evaluate_with_size(nontarget_data, torch.arange(0, min_size, 256),True)

## Evaluating InceptionScore on the biased data 
inception = InceptionScore(vicreg)
target_data_loader = DataLoader(target_data, 256)
for batch, x in enumerate(target_data_loader):
    x = x.to(DEVICE)
    inception.update(x)
target_is_val = inception.compute()


inception = InceptionScore(vicreg)
nontarget_data_loader = DataLoader(nontarget_data, 256)
for batch, x in enumerate(nontarget_data_loader):
    x = x.to(DEVICE)
    inception.update(x)
nontarget_is_val = inception.compute()


state = default_evaluator.run(inifite_iterator([target_data_loader, nontarget_data_loader]))
target_nontarget_fid_val = state.metrics["fid"]


## =============================================================== UNBIASED DATA =========================================================

## Removing biased data to make space
del(target_data)
del(nontarget_data)
del(target_data_loader)
del(nontarget_data_loader)
gc.collect()
## Creating unbiased data
data = datasets.CelebA("./dataset", download=True, split='all', transform=ToTensor()) # if not working, try download files manually off the internet (mostly works in the morning XD)
data_loader = DataLoader(data, batch_size= min_size, shuffle = True)
data_iter = iter(data_loader)
unbiased_data = next(data_iter)[0]


## Evaluating RankMe on the unbiased data
unbiased_rankme_val = rankme_general(unbiased_data, save_eval = True)

## Checking of rankme results are reasonable
## We expect to have an increasing score as we get more data
# rankme_general.evaluate_with_size(unbiased_data, torch.arange(0, min_size, 1000),True)


## Evaluating InceptionScore on the unbiased data
inception = InceptionScore(vicreg)
unbiased_data_loader = DataLoader(unbiased_data, 256)
for batch, x in enumerate(unbiased_data_loader):
    x = x.to(DEVICE)
    inception.update(x)

unbiased_is_val = inception.compute()

## Load the target data and calculate fid
target_data = torch.load('target_data.pt')
target_data_loader = DataLoader(target_data, 256)
default_evaluator = Engine(eval_step)
metrics.attach(default_evaluator, "fid")
state = default_evaluator.run(inifite_iterator([target_data_loader, unbiased_data_loader]))
target_unbiased_fid_val = state.metrics["fid"]
## Remove the target to free memory
del(target_data)
del(target_data_loader)


## Load the nontarget data and calculate fid
nontarget_data = torch.load('nontarget_data.pt')
nontarget_data_loader = DataLoader(nontarget_data, 256)
default_evaluator = Engine(eval_step)
metrics.attach(default_evaluator, "fid")
state = default_evaluator.run(inifite_iterator([nontarget_data_loader, unbiased_data_loader]))
nontarget_unbiased_fid_val = state.metrics["fid"]
## Remove the nontarget to free memory
del(nontarget_data)
del(nontarget_data_loader)







# ------------------- FID with inceptionV3 network ---------------------

# fid = FrechetInceptionDistance()

# fid.update(target_data.type(torch.uint8), True)
# fid.update(nontarget_data.type(torch.uint8), False)
# fid.compute()



result_file.write(f"{str([datetime.now(), target_rankme_val, nontarget_rankme_val, unbiased_rankme_val, target_is_val, nontarget_is_val, unbiased_is_val, target_nontarget_fid_val, target_unbiased_fid_val, nontarget_unbiased_fid_val])}\n")
result_file.close()
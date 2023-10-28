import torch
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from ignite.engine import Engine
from ignite.metrics import FID
from torchmetrics.image.inception import InceptionScore
from RankMe import RankMe


import gc 
from datetime import datetime
import argparse
from os import path


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', default="vicreg", type=str)
    parser.add_argument('-b', '--batch-size', default="256", type=int)
    parser.add_argument('-s', '--sample-iter', default="500", type=int)
    parser.add_argument('--min-size', default="51200", type=int)
    parser.add_argument('-f', '--file', default="./pipeline_results.txt", type=str)
    parser.add_argument('-t', '--target-idx', default="20", type=str)
    parser.add_argument('--seed', default="1", type=int)

    return parser.parse_args()


def load_model(model_string):
    if model_string.lower() == "dino":
        model = torch.hub.load('facebookresearch/dino:main', 'dino_resnet50')
    elif model_string.lower() == "vicreg":
        model = torch.hub.load('facebookresearch/vicreg:main', 'resnet50')
    else:
        raise Exception("Valid model names are 'dino' or 'vicreg'")
    
    return model


def load_data(batch_size):
    try:
        data = datasets.CelebA("./dataset", download=True, split='all', transform=ToTensor())
    except:
        print("Data cannot be downloaded. Try download files manually off the internet.")

    data_loader = DataLoader(data, batch_size, shuffle=True)

    return data_loader


def check_min_size(args, data_shapes):
    if min(data_shapes) < args.min_size:
        args.min_size = min(data_shapes) 
        print(f"The sample-iter is too low. the min-size is now set to minimum number of sample size which is { min(data_shapes) }")

    if args.min_size % args.batch_size != 0:
        ## Make the batch sizes a common divider of batch_size
        args.min_size -= (args.min_size % args.batch_size)
        print(f"min-size is not the common divider of batch-size.\n setting the min-size to {args.min_size}")

    if args.min_size <= 0:
        raise Exception("the min-size is zero or lower. If you did not choose a zero min-size then increase the sample-iter or lower the batch-size")


def generate_bias_data(data_loader, target_attr = 20, early_break = None):
    target_data = []
    nontarget_data = []

    for i, (x, y) in enumerate(data_loader):
        # t_idxs = torch.any(y[:, target_attr] == 1, 1)
        # nt_idxs =  ~t_idxs

        t_idxs = torch.all(y[:, target_attr] == 1, 1)
        nt_idxs = torch.all(y[:, target_attr] == 0, 1)

        target_data.append(x[t_idxs, :, :, :])
        nontarget_data.append(x[nt_idxs, :, :, :])
        if early_break:
            if early_break == i:
                print(f"\nFinished generating after {i} iterations\n")
                break
    
    target_data = torch.cat(target_data)
    nontarget_data = torch.cat(nontarget_data)
    
    return target_data, nontarget_data


def evaluate_inception_score(data_loader, model):
    inception = InceptionScore(model)
    for batch, x in enumerate(data_loader):
        inception.update(x)
    score = inception.compute()
    return score


def evaluate_fid(data_loader_1, data_loader_2, metrics):
    default_evaluator = Engine(eval_step)
    metrics.attach(default_evaluator, "fid")
    state = default_evaluator.run(inifite_iterator([data_loader_1, data_loader_2]))
    fid = state.metrics["fid"]
    return fid


## Utility functions for FID
def eval_step(engine, batch):
    ## The engine needed for ignite 
    return batch


def inifite_iterator(data_loaders):
    ## The data to be passed to the engine
    for x, y in zip(data_loaders[0],data_loaders[1]):
        yield [x, y]


if __name__ == '__main__':

    args = arg_parser()
    print(args)
    torch.manual_seed(args.seed)


    if path.isfile(args.file):
        result_file = open(args.file, "a")
    else:
        result_file = open(args.file, "a+")
        result_file.write("date,target_rankme,nontarget_rankme,unbiased_rankme,target_inception,nontarget_inception,unbiased_inception,target_nontarget_fid,target_unbiased_fid,nontarget_unbiased_fid\n")


    model = load_model(args.model)
    data_loader = load_data(args.batch_size)

    ## Create biased data
    target_idx = args.target_idx.split(",")
    target_idx = [int(idx) for idx in target_idx]
    target_idx = torch.Tensor(target_idx).type(torch.int)
    target_data, nontarget_data = generate_bias_data(data_loader, target_idx, args.sample_iter)

    ## Save the data for later use
    torch.save(target_data, 'target_data.pt')
    torch.save(nontarget_data, 'nontarget_data.pt')

    data_shapes = [target_data.shape[0], nontarget_data.shape[0]]
    check_min_size(args, data_shapes)

    target_data = target_data[:args.min_size,:, :, :]
    nontarget_data = nontarget_data[:args.min_size,:, :, :]


    ## Creating the RankMe objects
    rankme_general = RankMe(model, args.batch_size)
    rankme_target = RankMe(model, args.batch_size)
    rankme_nontarget = RankMe(model, args.batch_size)

    ## Creating FID metric and needed function
    metrics = FID(2048, model)


    ## =============================================================== BIASED DATA =========================================================
    ## Evaluating RankMe on the biased data 
    target_rankme_val = rankme_target(target_data, save_eval = True)
    nontarget_rankme_val = rankme_nontarget(nontarget_data, save_eval = True)

    ## Checking of rankme results are reasonable
    ## We expect to have an increasing score as we get more data
    # rankme_target.evaluate_with_size(target_data, torch.arange(0, min_size, 256),True)
    # rankme_nontarget.evaluate_with_size(nontarget_data, torch.arange(0, min_size, 256),True)

    ## Making the data loaders
    target_data_loader = DataLoader(target_data,  args.batch_size)
    nontarget_data_loader = DataLoader(nontarget_data,  args.batch_size)

    ## Evaluating InceptionScore on the biased data 
    target_is_val = evaluate_inception_score(target_data_loader, model)
    nontarget_is_val = evaluate_inception_score(nontarget_data_loader, model)

    # Evaluating FID on target and nontarget data
    target_nontarget_fid_val = evaluate_fid(target_data_loader, nontarget_data_loader, metrics)





    ## =============================================================== UNBIASED DATA =========================================================
    ## Creating unbiased data
    unbiased_data = torch.concat([target_data[:args.min_size//2, :], nontarget_data[:args.min_size//2, :]])

    # data_loader = load_data(args.min_size)
    # data_iter = iter(data_loader)
    # unbiased_data = next(data_iter)[0]
    

    ## Removing biased data to make space
    del(target_data)
    del(nontarget_data)
    del(data_loader)
    del(target_data_loader)
    del(nontarget_data_loader)
    gc.collect()

    ## Evaluating RankMe on the unbiased data
    unbiased_rankme_val = rankme_general(unbiased_data, save_eval = True)

    ## Checking of rankme results are reasonable
    ## We expect to have an increasing score as we get more data
    # rankme_general.evaluate_with_size(unbiased_data, torch.arange(0, min_size, 1000),True)


    ## Evaluating InceptionScore on the unbiased data
    unbiased_data_loader = DataLoader(unbiased_data, args.batch_size)
    unbiased_is_val = evaluate_inception_score(unbiased_data_loader, model)

    ## Load the target data and calculate fid
    target_data = torch.load('target_data.pt')
    target_data_loader = DataLoader(target_data,  args.batch_size)
    target_unbiased_fid_val = evaluate_fid(target_data_loader, unbiased_data_loader, metrics)

    ## Remove the target to free memory
    del(target_data)
    del(target_data_loader)


    ## Load the nontarget data and calculate fid
    nontarget_data = torch.load('nontarget_data.pt')
    nontarget_data_loader = DataLoader(nontarget_data,  args.batch_size)
    nontarget_unbiased_fid_val = evaluate_fid(nontarget_data_loader, unbiased_data_loader, metrics)

    ## Remove the nontarget to free memory
    del(nontarget_data)
    del(nontarget_data_loader)


    ## Writing down the results

    result_file.write(f"{str([datetime.now(), target_rankme_val, nontarget_rankme_val, unbiased_rankme_val, target_is_val, nontarget_is_val, unbiased_is_val, target_nontarget_fid_val, target_unbiased_fid_val, nontarget_unbiased_fid_val])}\n")
    result_file.close()
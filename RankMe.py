import torch
from torch import tensor
from torch.utils.data import DataLoader

class RankMe:
    def __init__(self, model, batch_size, device = "cpu") -> None:
        self.model = model
        self.device = device
        self.evaluations = torch.empty((1,)).to(self.device)
        self.batch_size = batch_size
    
    def __call__(self, data:tensor, representation_dim = 2048, save_eval = False) -> tensor:

        model_output = torch.zeros((data.shape[0], representation_dim))
        data_loader = DataLoader(data, self.batch_size)
        with torch.no_grad():
            for batch, x in enumerate(data_loader):
                x = x.to(self.device)
                model_output[(batch*self.batch_size):((batch+1)*self.batch_size), :] = self.model(x)
            
        sigma = self._calc_singular(model_output)
        eps = torch.finfo(model_output.dtype).eps
        # sum_range = torch.min(model_output.shape)

        # p_ks = (sigma / torch.linalg.norm(sigma, ord = 1) ) + eps
        p_ks = (sigma / torch.abs(sigma).sum() ) + eps
        rank = torch.exp(-torch.sum(p_ks * torch.log(p_ks)))
        rank = rank.unsqueeze(0)
        rank = rank.to(self.device)
        if save_eval:
            self.evaluations = torch.cat((self.evaluations, rank))
            
        return rank


    def _calc_singular(self, x):
        _, singular_values, _ = torch.linalg.svd(x)
        return singular_values
    
    def save_evaluation(self, result)-> None:
        self.evaluations = torch.cat((self.evaluations, result))
    

    def evaluate_with_size(self, data, size = None, save_eval = False):
        if size == None:
            size = range(data.shape[0])

        eval_result = torch.zeros(size.shape[0])
        
        for idx, sample_size in enumerate(size):
            eval_result[idx] = self(data[0:sample_size,:, :, :], save_eval = save_eval)
        
        return eval_result

    def del_evaluation(self, index = None):
        if index:
            self.evaluations = self.evaluations[index:]
        else:
            self.evaluations = torch.empty((1,))
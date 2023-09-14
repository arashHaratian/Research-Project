import torch
from torch import tensor


class RankMe:
    def __init__(self, model, device = "cpu") -> None:
        self.evaluations = torch.empty((1,)).to(device)
        self.model = model
    
    def __call__(self, data:tensor, save_eval = False) -> tensor:
        with torch.no_grad():
            model_output = self.model(data)
            
        sigma = self._calc_singular(model_output)
        eps = torch.finfo(model_output.dtype).eps
        # sum_range = torch.min(model_output.shape)

        # p_ks = (sigma / torch.linalg.norm(sigma, ord = 1) ) + eps
        p_ks = (sigma / torch.abs(sigma).sum() ) + eps
        rank = torch.exp(-torch.sum(p_ks * torch.log(p_ks)))
        rank = rank.unsqueeze(0)

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
        eval_result = torch.zeros(data.shape[0])
        for i in size:
            eval_result[i] = self(data[0:i,:, :, :], save_eval)
        
        return eval_result

    def del_evaluation(self, index = None):
        if index:
            self.evaluations = self.evaluations[index:]
        else:
            self.evaluations = torch.empty((1,))
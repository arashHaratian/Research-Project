import torch
from torch import tensor


class RankMe:
    def __init__(self) -> None:
        self.evaluations = torch.empty((1,))
    
    def __call__(self, model_output:tensor, save_eval = False) -> tensor:

        sigma = self._calc_singular(model_output)
        eps = torch.finfo(model_output.dtype).eps
        # sum_range = torch.min(model_output.shape)

        p_ks = (sigma / torch.linalg.norm(sigma, ord = 1) ) + eps
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
        

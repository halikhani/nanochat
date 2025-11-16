"""
Borrowed from modded-nanogpt. By Keller, @vagrawal, et al.
Not a general optimizer! But works for our specific use.
"""
# TODO (Hamidreza): get back to this later for better understanding 
import torch
import torch.distributed as dist
from torch import Tensor

class DistAdamW(dist.optim.Optimizer):
    """
    Distributed AdamW optimizer.
    In the style of ZeRO-2, i.e. sharded optimizer states and gradient reduction
    """
    def __init__(self, param_groups, lr: float = 1e-3, betas: tuple[float, float] = (0.9, 0.999), eps: float = 1e-8, weight_decay: float = 0.01):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(param_groups, defaults)

    @torch.compile
    @torch.no_grad()
    def step(self):
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        reduce_scatter_futures: list[torch.Future] = [] # for the gradient sharding
        all_reduce_futures: list[torch.Future] = [] # for re-gathering the updated parameters.
        grad_slices = [] # we’ll store each rank’s gradient slice per parameter here.

        # Pass 1: shard gradients (reduce-scatter)
        # this assumes:
        # 1. Every param is at least 1-D (has .shape[0]).
        # 2. First dimension is divisible by world_size:
        for group in self.param_groups:
            params: list[Tensor] = group["params"]
            for base_i in range(len(params)):
                grad = params[base_i].grad
                rank_size = grad.shape[0] // world_size
                grad_slice = torch.empty_like(grad[:rank_size])
                reduce_scatter_futures.append(dist.reduce_scatter_tensor(grad_slice, grad, op=dist.ReduceOp.AVG, async_op=True).get_future())
                grad_slices.append(grad_slice)

        # Pass 2: apply AdamW on the shard and all-gather parameters 
        
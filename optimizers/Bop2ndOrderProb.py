import copy
import math
import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer, required
from torch import Tensor
from typing import List, Optional
import torch.optim._functional as F


class Bop2ndOrderProb(Optimizer):
    def __init__(
        self, 
        params, 
        gamma: float = 1e-4,
        threshold: float = 1e-8,
        sigma: float = 1e-2,
        name="Bop2ndOrderProb", 
        **kwargs
    ):
        if gamma < 0:
            raise ValueError(
                'Invalid gamma value: {}'.format(gamma)
            )
        if threshold < 0:
            raise ValueError(
                'Invalid threshold value: {}'.format(threshold)
            )
        if sigma < 0:
            raise ValueError(
                'Invalid sigma value: {}'.format(sigma)
            )

        defaults = dict(
            gamma=gamma,
            threshold=threshold,
            sigma=sigma
        )

        super(Bop2ndOrderProb, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        i = 0
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                i += 1
                # Bop2ndOrder optimizer
                if hasattr(p,'org'):
                    grad = p.grad.data

                    state = self.state[p]
                    if len(state) == 0:
                        state['step'] = 0

                    if not hasattr(p,'m'):
                        p.m = torch.zeros_like(p, memory_format=torch.preserve_format)
                    if not hasattr(p,'v'):
                        p.v = torch.zeros_like(p, memory_format=torch.preserve_format)
                    if not hasattr(p, 'bigger'):
                        p.bigger = []
                    if not hasattr(p, 'bigger_same'):
                        p.bigger_same = []
                    if not hasattr(p, 'smaller_same'):
                        p.smaller_same = []
                    if not hasattr(p, 'bigger_same_min_max_mean'):
                        p.bigger_same_min_max_mean = []
                    if not hasattr(p, 'smaller_same_min_max_mean'):
                        p.smaller_same_min_max_mean = []
                    if not hasattr(p, 'bigger_flip_num'):
                        p.bigger_flip_num = []
                    if not hasattr(p, 'smaller_flip_num'):
                        p.smaller_flip_num = []
                    if not hasattr(p, 'epochs'):
                        p.epochs = 1

                    gamma= group['gamma']
                    threshold = group['threshold']
                    sigma = group['sigma']
                    m = p.m
                    v = p.v
                    state['step'] += 1

                    m_t = m.add_(gamma * (grad - m))
                    v_t = v.add_(sigma * (grad*grad - v))
                    delta = abs(m_t) / (v_t.sqrt() + 1e-10)
                    delta = delta/threshold
                    # print(f'delta:{delta}, total elememnts: {delta.shape}, number of elements > 1:  {torch.sum(delta>1)}')
                    # import sys
                    # sys.exit(0)
                    p.bigger.append((torch.sum(delta>1)/delta.numel()).item())
                    p.bigger_same.append((torch.sum((delta > 1) & (torch.sign(m_t) == torch.sign(p.data))) / delta.numel()).item())
                    p.smaller_same.append((torch.sum((delta < 1) & (torch.sign(m_t) == torch.sign(p.data))) / delta.numel()).item())

                    # threshold = 1 - pow(0.5, state['step'])/2 # resnet18 44.29 4gpu
                    # threshold = 1 - pow(0.9, state['step'])/2 # resnet18 44.16 4gpu

                    # threshold = 0.8
                    threshold = 1 - pow(0.5, p.epochs)/2
                    bigger_threshold = 1/threshold

                    # handle bigger ones
                    indices_largethan1 = torch.where((delta >= bigger_threshold) & (torch.sign(m_t) == torch.sign(p.data)))
                    p.data[indices_largethan1] = -p.data[indices_largethan1]

                    indices_largethan1 = torch.where((bigger_threshold > delta) & (delta > 1) & (torch.sign(m_t) == torch.sign(p.data)))
                    not_all_zeros = False
                    for v in indices_largethan1:
                        if v.numel() != 0:
                            not_all_zeros = True
                    if not_all_zeros:
                        p.bigger_same_min_max_mean.append(torch.min(delta[indices_largethan1]).item())
                        p.bigger_same_min_max_mean.append(torch.max(delta[indices_largethan1]).item())
                        p.bigger_same_min_max_mean.append(torch.mean(delta[indices_largethan1]).item())
                        temp = torch.sign(2*torch.bernoulli(1/delta[indices_largethan1]) - 1)
                        p.bigger_flip_num.append(torch.sum(temp == -1).item())
                        p.data[indices_largethan1] = torch.sign(p.data[indices_largethan1]) * temp
                    else:
                        p.bigger_same_min_max_mean = [-1, -1, -1]
                        p.bigger_flip_num.append(0)

                    # p.smaller_same_min_max_mean = [-1, -1, -1]
                    # p.smaller_flip_num.append(0)
                    #handle smaller ones.
                    indices_lessthan1 = torch.where((threshold < delta) & (delta < 1) & (torch.sign(m_t) == torch.sign(p.data)))
                    not_all_zeros = False
                    for v in indices_lessthan1:
                        if v.numel() != 0:
                            not_all_zeros = True
                    if not_all_zeros:
                        # print(f'indices_lessthan1: {indices_lessthan1}')
                        p.bigger_same_min_max_mean.append(torch.min(delta[indices_lessthan1]).item())
                        p.bigger_same_min_max_mean.append(torch.max(delta[indices_lessthan1]).item())
                        p.bigger_same_min_max_mean.append(torch.mean(delta[indices_lessthan1]).item())
                        temp = torch.sign(-2*torch.bernoulli(delta[indices_lessthan1]) + 1)
                        p.smaller_flip_num.append(torch.sum(temp == -1).item())
                        p.data[indices_lessthan1] = torch.sign(p.data[indices_lessthan1]) * temp
                    else:
                        p.smaller_same_min_max_mean = [-1, -1, -1]
                        p.smaller_flip_num.append(0)
                    
                    if state['step'] == 391:
                        p.epochs += 1
                    # p.data = torch.sign(torch.sign(-torch.sign(p.data.mul(temp) - threshold).mul(p.data)).add(0.1))

        return loss

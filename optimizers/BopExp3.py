import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer, required
from torch import Tensor
from typing import List, Optional
import torch.optim._functional as F
import math
import sys

class BopExp3(Optimizer):
    def __init__(
        self, 
        params, 
        gamma: float = 1e-4,
        threshold: float = 1e-8,
        sigma: float = 1e-2,
        alpha: float = 0.5,
        name="BopExp3", 
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
            sigma=sigma,
            alpha=alpha
        )

        super(BopExp3, self).__init__(params, defaults)

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
                        p.m = torch.zeros_like(p, memory_format=torch.preserve_format).cuda()
                    if not hasattr(p,'v'):
                        p.v = torch.zeros_like(p, memory_format=torch.preserve_format).cuda()
                    if not hasattr(p, 'flip_num'):
                        p.flip_num = []
                    if not hasattr(p, 'epochs'):
                        p.epochs = 1

                    if not hasattr(p, 'each_flip_num'):
                        # p.each_flip_num = torch.zeros_like(p, memory_format=torch.preserve_format)+1
                        p.each_flip_num = torch.zeros_like(p, memory_format=torch.preserve_format)

                    gamma= group['gamma']
                    threshold = group['threshold']
                    sigma = group['sigma']
                    alpha = group['alpha']
                    m = p.m
                    v = p.v
                    state['step'] += 1

                    #print("flip_num_start: ", torch.sum(abs(p.data-p.pre_binary_data)/2))

                    # print(f'grad: {grad}, m: {m}, gamma: {gamma}')
                    m_t = m.add_(gamma * (grad - m))
                    v_t = v.add_(sigma * (grad*grad - v))
                    delta = abs(m_t) / (v_t.sqrt() + 1e-10)
                    delta = delta/threshold
                    out_c = delta.shape[0]
                    # print(f'delta.shape: {delta.shape}')
                    # print(f'p.data: {p.data}, m_t: {m_t}')
                    thresholds = 1 - pow(alpha, p.epochs)/2
                    bigger_thresholds = (1/thresholds)


                    indices_largethan1 = torch.where((delta >= bigger_thresholds) & (torch.sign(m_t) == torch.sign(p.data)))
                    p.data[indices_largethan1] = -p.data[indices_largethan1]
                    p.each_flip_num[indices_largethan1] +=1

                    indices = torch.where((torch.sign(m_t) == torch.sign(p.data)) & (delta>=thresholds) & (delta<bigger_thresholds))
                    # indices = torch.where((torch.sign(m_t) == torch.sign(p.data)))
                    # print("indices: ",len(indices))
                    # print("ratio: ",len(indices)/p.data.numel())
                    delta = delta[indices]
                    # print(f'mean of delta: {torch.mean(delta)}')
                    # EPS = math.sqrt(1e-3)
                    # mu = torch.randn(delta.size()).cuda()*EPS
                    # print(f'delta: {delta}, mu: {mu}')
                    # import sys
                    # sys.exit(0)
                    # delta += mu
                    # num_batches = 391
                    # prob = torch.exp(-delta*int(num_batches/state['step']))
                    # print(f"state['step']: {state['step']}, epochs: {p.epochs}")
                    # prob = torch.exp(-delta/(p.epochs * pow(p.each_flip_num[indices], 2)))
                    # prob = torch.exp(-delta/(p.epochs * p.each_flip_num[indices] * 2))
                    # prob = torch.exp(-delta/(p.each_flip_num[indices]*1.1*pow(1.01, out_c/128)))
                    # prob = torch.exp(-delta/(p.each_flip_num[indices]*pow(1.1, out_c/128)))
                    prob = torch.exp(-delta/(p.each_flip_num[indices]+1e-10))
                    # prob = 1.0/(delta/(p.each_flip_num[indices]+ 1e-10))
                    #print("mean_prob",torch.mean(1-prob))
                    #print("max_prob",1-torch.min(prob))
                    #print("min_prob",1-torch.max(prob))
                    # prob = torch.exp(-delta)

                    
                    # prob = torch.exp(-delta/pow(p.each_flip_num[indices],2))
                    # prob = torch.exp(-delta)
                    # prob = torch.exp(-delta)/p.each_flip_num[indices]
                    # prob = torch.exp(-delta/(p.epochs * p.each_flip_num[indices]))

                    tmp = torch.sign(2*torch.bernoulli(prob) - 1)

                    prev_data = p.data[indices]
                    p.data[indices] *= tmp
                    p.each_flip_num[indices] += (1-tmp)/2
                    # import sys
                    # sys.exit(0)                    
                    p.flip_num.append(torch.sum(tmp == -1).item())
                    if state['step'] == 196:
                    #if state['step'] == 391:
                        p.epochs += 1
        return loss
        
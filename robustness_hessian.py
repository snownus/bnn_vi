from utils import * # get the dataset
from pyhessian import hessian # Hessian computation

import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import copy

criterion = nn.MSELoss()
def compute_hessian(model, inputs, targets, cuda=False):
    # create the hessian computation module
    hessian_comp = hessian(model, criterion, data=(inputs, targets), cuda=cuda)
    # Now let's compute the top eigenvalue. This only takes a few seconds.
    top_eigenvalues, top_eigenvector = hessian_comp.eigenvalues(top_n=2)
    print("The top Hessian eigenvalue of this model is %.4f"%top_eigenvalues[-1])

    # This is a simple function, that will allow us to perturb the model paramters and get the result
    def get_params(model_orig,  model_perb, direction, alpha):
        model_orig_list = [param for param in model_orig.parameters() if param.requires_grad is True]
        model_perb_list = [param for param in model_perb.parameters() if param.requires_grad is True]
        for m_orig, m_perb, d in zip(model_orig_list, model_perb_list, direction):
            m_perb.data = m_orig.data + alpha * d
        return model_perb

    # lambda is a small scalar that we use to perturb the model parameters along the eigenvectors 
    lams = np.linspace(-0.5, 0.5, 21).astype(np.float32)
    sgd_loss_list_rand = []

    # create a copy of the model
    model_perb = copy.deepcopy(model)
    model_perb.eval()

    for lam in lams:
        if cuda is True:
            inputs = inputs.cuda()
            targets = targets.cuda()
        model_perb = get_params(model, model_perb, top_eigenvector[0], lam)
        sgd_loss_list_rand.append(criterion(model_perb(inputs), targets).item())
    
    return lams, sgd_loss_list_rand
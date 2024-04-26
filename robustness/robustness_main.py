from robustness_data import generate_complete_graph_data, generate_random_graph_data
from robustness_train import train, train_sdp
from robustness_mlp import models
from robustness_hessian import compute_hessian

import torch

import pandas as pd

def save_losses(dataset, sgd_train_losses, adam_train_losses, SDP_train_losses, 
                sgd_test_losses, adam_test_losses, SDP_test_losses, lr):
    data = {
        "SGD": sgd_train_losses,
        "Adam": adam_train_losses, 
        "Ours": SDP_train_losses
    }
    df = pd.DataFrame(data)
    df.to_csv(dataset + '_' + str(lr) + '_' + 'train.csv', index=False)

    data = {
        "SGD": sgd_test_losses,
        "Adam": adam_test_losses, 
        "Ours": SDP_test_losses
    }
    df = pd.DataFrame(data)
    df.to_csv(dataset  + '_' + str(lr) + '_' +  'test.csv', index=False)


if __name__ == '__main__':
    hidden_channels = 500
    lr = 0.01
    K, L, scale = 4, 40, 100
    complete_data, random_data = generate_complete_graph_data(), generate_random_graph_data()

    for dataset, dataloader in zip(['complete', 'random'], [complete_data, random_data]):
        train_data, test_data, input_channels = dataloader
        
        BASE1 = models(dataset, "BASE", hidden_channels, -1, -1, input_channels=input_channels)
        sgd_optimizer = torch.optim.SGD(BASE1.parameters(), lr=lr)
        sgd_train_losses, sgd_test_losses = train(BASE1, sgd_optimizer, train_data, test_data)

        BASE2 = models(dataset, "BASE", hidden_channels, -1, -1, input_channels=input_channels)
        adam_optimizer = torch.optim.SGD(BASE1.parameters(), lr=lr)
        adam_train_losses, adam_test_losses = train(BASE1, sgd_optimizer, train_data, test_data)

        SDP_model = models(dataset, "SDP", hidden_channels, K, scale, input_channels=input_channels)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        SDP_model.to(device)
        sdp_optimizer = torch.optim.SGD(SDP_model.parameters(), lr=lr)
        SDP_train_losses, SDP_test_losses = train_sdp(SDP_model, sdp_optimizer, train_data, test_data)

        save_losses(dataset, sgd_train_losses, adam_train_losses, SDP_train_losses, 
                    sgd_test_losses, adam_test_losses, SDP_test_losses, lr)

        i = 0
        for inputs, targets in test_data:
            lams, sgd_loss_list = compute_hessian(BASE1, inputs, targets, cuda=False)
            lams, adam_loss_list = compute_hessian(BASE2, inputs, targets, cuda=False)
            lams, SDP_loss_list = compute_hessian(SDP_model, inputs, targets, cuda=True)
            
            if i == 0:
                sgd_final_loss, adam_final_loss, SDP_final_loss = sgd_loss_list, adam_loss_list, SDP_loss_list
            else:
                sgd_final_loss = [sgd_final_loss[j] + sgd_loss_list[j] for j in range(len(sgd_loss_list))]
                adam_final_loss = [adam_final_loss[j] + adam_loss_list[j] for j in range(len(adam_loss_list))]
                SDP_final_loss = [SDP_final_loss[j] + SDP_loss_list[j] for j in range(len(SDP_loss_list))]
        
        sgd_final_loss = [sgd_final_loss[j]/len(test_data) for j in range(len(sgd_final_loss))]
        adam_final_loss = [adam_final_loss[j]/len(test_data) for j in range(len(adam_final_loss))]
        SDP_final_loss = [SDP_final_loss[j]/len(test_data) for j in range(len(SDP_final_loss))]

        data = {
            "SGD": sgd_final_loss,
            "Adam": adam_final_loss, 
            "Ours": SDP_final_loss,
            "pertubation": lams
        }
        df = pd.DataFrame(data)
        df.to_csv(dataset  + '_' + str(lr) + '_' +  'hessian.csv', index=False)
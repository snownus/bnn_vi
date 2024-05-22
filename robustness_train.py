import time
import torch
import torch.nn as nn

from math import cos, pi

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
criterion = nn.MSELoss()
def train(model, optimizer, dataloader, test_dataloader, init_lr, epochs):
    start_time = time.time()
    train_loss, test_loss = AverageMeter(), AverageMeter()
    train_losses, test_losses = [], []
    for i in range(epochs):
        # Run the training batches
        for b, (X_train, y_train) in enumerate(dataloader):
            adjust_learning_rate_cos(optimizer, i, b, epochs, len(dataloader), init_lr)
            #X_train = X_train.reshape(1,2)
            # Apply the model
            y_pred = model(X_train) 
            loss = criterion(y_pred, y_train)
            train_loss.update(loss.item(), X_train.size(0))

            # Update parameters
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        train_losses.append(train_loss.avg)

        # Run the testing batches
        with torch.no_grad():
            for b, (X_test, y_test) in enumerate(test_dataloader):
                y_val = model(X_test)
                loss = criterion(y_val, y_test)
                test_loss.update(loss.item(), X_test.size(0))
        test_losses.append(test_loss.avg)

    print(f'\nDuration: {time.time() - start_time:.0f} seconds') # print the time elapsed
    return train_losses, test_losses


def sample_uniform_int(a, b):
    # Sample a single integer from a uniform distribution between a and b (inclusive)
    sampled_number = torch.randint(a, b + 1, (1,))
    
    return sampled_number.item()


def train_sdp(model, optimizer, dataloader, test_dataloader, init_lr, epochs):
    start_time = time.time()
    train_loss, test_loss = AverageMeter(), AverageMeter()
    train_losses, test_losses = [], []
    L = 40
    for i in range(epochs):
        # Run the training batches
        for b, (X_train, y_train) in enumerate(dataloader):
            adjust_learning_rate_cos(optimizer, i, b, epochs, len(dataloader), init_lr)

            X_train, y_train = X_train.cuda(), y_train.cuda()
            output = model(X_train)
            loss = criterion(output, y_train)
            loss.backward()
            total_loss = loss
            total_output = output

            optimizer.step()
            optimizer.zero_grad()
            y_pred = model(X_train) 
            loss = criterion(y_pred, y_train)
            train_loss.update(loss.item(), X_train.size(0))

        train_losses.append(train_loss.avg)
        # Run the testing batches
        with torch.no_grad():
            for b, (X_test, y_test) in enumerate(test_dataloader):
                X_test, y_test = X_test.cuda(), y_test.cuda()
                with torch.no_grad():
                    output = model(X_train)
                    loss = criterion(output, y_train)
                    total_loss = loss
                    total_output = output

                    for j in range(L-1):
                        output = model(X_train)
                        loss = criterion(output, y_train)
                        total_loss += loss
                        total_output += output
                    total_loss /= L
                    total_output /= L
                    test_loss.update(total_loss.item(), X_test.size(0))
        test_losses.append(test_loss.avg)
    print(f'\nDuration: {time.time() - start_time:.0f} seconds') # print the time elapsed

    return train_losses, test_losses


def adjust_learning_rate_cos(optimizer, epoch, step, total_epoch, len_epoch, init_lr):
    # first 5 epochs for warmup
    warmup_iter = 5 * len_epoch
    current_iter = step + epoch * len_epoch
    max_iter = total_epoch * len_epoch
    lr = init_lr * (1 + cos(pi * (current_iter - warmup_iter) / (max_iter - warmup_iter))) / 2
    if epoch < 5:
        lr = init_lr * current_iter / warmup_iter

    for i, param_group in enumerate(optimizer.param_groups):
        param_group['lr'] = lr
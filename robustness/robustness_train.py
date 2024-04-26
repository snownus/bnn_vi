import time
import torch
import torch.nn as nn

# optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
criterion = nn.MSELoss()
def train(model, optimizer, dataloader, test_dataloader):
    start_time = time.time()

    epochs = 100
    train_losses = []
    test_losses = []
    for i in range(epochs):
        # Run the training batches
        for b, (X_train, y_train) in enumerate(dataloader):
            #X_train = X_train.reshape(1,2)
            b+=1
            
            # Apply the model
            y_pred = model(X_train)  # we don't flatten X-train here
            loss = criterion(y_pred, y_train)
            
            # Update parameters
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        train_losses.append(loss.item())
        print("Train Loss:", loss.item())
            
        # Run the testing batches
        with torch.no_grad():
            for b, (X_test, y_test) in enumerate(test_dataloader):
                y_val = model(X_test)

        loss = criterion(y_val, y_test)
        test_losses.append(loss.item())
        print("Test Loss:", loss.item())
            
    print(f'\nDuration: {time.time() - start_time:.0f} seconds') # print the time elapsed
    return train_losses, test_losses


def sample_uniform_int(a, b):
    # Sample a single integer from a uniform distribution between a and b (inclusive)
    sampled_number = torch.randint(a, b + 1, (1,))
    
    return sampled_number.item()


def train_sdp(model, optimizer, dataloader, test_dataloader):
    start_time = time.time()

    epochs = 100
    train_losses = []
    test_losses = []

    K = 4
    L = 40
    for i in range(epochs):
        # Run the training batches
        for b, (X_train, y_train) in enumerate(dataloader):
            X_train, y_train = X_train.cuda(), y_train.cuda()
            num = sample_uniform_int(0, L + K * 2 - 1)
            if num < L:
                output = model(X_train)
                loss = criterion(output, y_train)
                loss.backward()
            elif L <= num < L + K:
                j = num - L
                params = model.named_parameters()
                for name, param in params:
                    # print(f'name: {name}, param.shape: {param.shape}')
                    if 'sample' in name and 'downsample' not in name:
                        param.zero_()
                        param[0][j] = 1
                output = model(X_train)
                loss = criterion(output, y_train) 
                loss.backward()
            else:
                j = num - L - K
                params = model.named_parameters()
                for name, param in params:
                    # print(f'name: {name}, param.shape: {param.shape}')
                    if 'sample' in name and 'downsample' not in name:
                        param.zero_()
                        param[0][j] = -1
                output = model(X_train)
                loss = criterion(output, y_train) 
                loss.backward()
                total_loss = loss
                total_output = output

                optimizer.step()
                optimizer.zero_grad()

            params = model.named_parameters()
            for name, param in params:
                if 'sample' in name and 'downsample' not in name:
                    param.zero_()
            #X_train = X_train.reshape(1,2)
            y_pred = model(X_train) 
            loss = criterion(y_pred, y_train)
    
            
        train_losses.append(loss.item())
        print("Train Loss:", loss.item())
            
        # Run the testing batches
        with torch.no_grad():
            for b, (X_test, y_test) in enumerate(test_dataloader):
                X_test, y_test = X_test.cuda(), y_test.cuda()
                output = model(X_test)
                loss = criterion(output, y_test)
                total_loss = loss
                total_output = output

                for j in range(L-1):
                    output = model(X_test)
                    loss = criterion(output, y_test)
                    total_loss += loss
                    total_output += output

                for j in range(K):
                    params = model.named_parameters()
                    for name, param in params:
                        # print(f'name: {name}, param.shape: {param.shape}')
                        if 'sample' in name and 'downsample' not in name:
                            param.zero_()
                            param[0][j] = 1
                    output = model(X_test)
                    total_loss += criterion(output, y_test)
                    total_output += output

                for j in range(K):
                    params = model.named_parameters()
                    for name, param in params:
                        # print(f'name: {name}, param.shape: {param.shape}')
                        if 'sample' in name and 'downsample' not in name:
                            param.zero_()
                            param[0][j] = -1
                    output = model(X_test)
                    total_loss += criterion(output, y_test)
                    total_output += output
                
                total_loss /= (L + K * 2)
                total_output /= (L + K * 2)
                y_val = total_output
                
        loss = criterion(y_val, y_test)
        test_losses.append(loss.item())
        #test_correct2.append(tst_corr)
        print(f"Test loss:{loss.item()} at epoch: {i}")
        print("<--------------------------------------->")
    
    print(f'\nDuration: {time.time() - start_time:.0f} seconds') # print the time elapsed
    return train_losses, test_losses
import numpy as np
import torch.nn as nn
import torch

train_on_gpu=torch.cuda.is_available()

if(train_on_gpu):
    print('Training on GPU.')
else:
    print('No GPU available, training on CPU.')


def train(net,train_loader,valid_loader,test_loader,criterion,optimizer):
    epochs = 4 
    counter = 0
    print_every = 100
    clip=5 

    if(train_on_gpu):
        net.cuda()

    net.train()
    for e in range(epochs):
        h = net.init_hidden(batch_size)
        for inputs, labels in train_loader :
            counter += 1
            if(train_on_gpu):
                inputs, labels = inputs.cuda(), labels.cuda()
            h = tuple([each.data for each in h])

            net.zero_grad()

            output, h = net(inputs, h)

            loss = criterion(output.squeeze(), labels.float())
            loss.backward()

            nn.utils.clip_grad_norm_(net.parameters(), clip)
            optimizer.step()
            if counter % print_every == 0 :
                val_h = net.init_hidden(batch_size)
                val_losses = []
                net.eval()
                for inputs, labels in valid_loader :
                    val_h = tuple([each.data for each in val_h])
                    if(train_on_gpu):
                        inputs, labels = inputs.cuda(), labels.cuda()
                    output, val_h = net(inputs, val_h)
                    val_loss = criterion(output.squeeze(), labels.float())
                    val_losses.append(val_loss.item())
                net.train() 
        return net
            
    
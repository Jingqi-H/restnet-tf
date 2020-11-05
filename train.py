import torch
import numpy as np
import matplotlib.pyplot as plt


def train(net,
          train_loader,
          train_num,
          loss_function,
          optimizer):
    running_loss = []
    acc = 0.0
    lr_list = []
    for step, data in enumerate(train_loader, start=0):
        images, labels = data
        optimizer.zero_grad()
        logits = net(images.cuda())
        loss = loss_function(logits, labels.cuda())

        predict_y = torch.max(logits, dim=1)[1]
        acc += (predict_y == labels.cuda()).sum().item()

        loss.backward()
        # scheduler.step(loss)
        optimizer.step()
        # scheduler.step()
        lr_list.append(optimizer.state_dict()['param_groups'][0]['lr'])

        running_loss.append(loss.item())

    print("lr：{}".format(optimizer.param_groups[0]['lr']))

    loss_ave = np.mean(running_loss)
    train_ave = acc / train_num
    return loss_ave, train_ave

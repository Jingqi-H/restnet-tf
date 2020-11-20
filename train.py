import torch
import numpy as np
import matplotlib.pyplot as plt


def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))


def adjust_learning_rate(optimizer, i_iter, base_lr, total_niters, lr_power):
    lr = lr_poly(base_lr, i_iter, total_niters, lr_power)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10
    return lr


def train(net,
          base_lr,
          epoch,
          num_epoch,
          train_num,
          train_loader,
          loss_function,
          optimizer):
    running_loss = []
    acc = 0.0
    lr_list = []

    total_niters = num_epoch * len(train_loader)
    # lr_power = 0.9

    for step, data in enumerate(train_loader, start=0):
        images, labels = data

        current_idx = epoch * len(train_loader) + step
        optimizer.zero_grad()
        lr = adjust_learning_rate(optimizer, current_idx, base_lr, total_niters, lr_power=0.9)

        logits = net(images.cuda())
        loss = loss_function(logits, labels.cuda())

        predict_y = torch.max(logits, dim=1)[1]
        acc += (predict_y == labels.cuda()).sum().item()

        loss.backward()
        optimizer.step()
        lr_list.append(lr)

        running_loss.append(loss.item())

        # print('current_idx={} | lr={}'.format(current_idx, lr))
    print('lr=', lr)
    loss_ave = np.mean(running_loss)
    train_ave = acc / train_num
    return loss_ave, train_ave, lr_list

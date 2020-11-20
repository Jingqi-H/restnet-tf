import torch
import numpy as np
from utils.metrics import pred_prob2pred_label, metrics_score


def eval(net, loss_function, validation_loader):
    running_loss = []
    pred_prob_all, pred_label_all, gt_label = [], [], []
    with torch.no_grad():
        for step, val_data in enumerate(validation_loader, start=0):
            val_images, val_labels = val_data
            outputs = net(val_images.cuda())  # eval model only have last output layer

            pred_prob, pred_label = pred_prob2pred_label(outputs)
            # print('val_labels:', val_labels)
            # print('pred_label:', pred_label)
            # print('pred_prob:\n', pred_prob)
            # print('outputs from network:\n', outputs)

            pred_label_all.append(pred_label)
            pred_prob_all.append(pred_prob)
            gt_label.append(val_labels)

            loss = loss_function(outputs, val_labels.cuda())
            running_loss.append(loss.item())

        pred_prob_all = np.concatenate(pred_prob_all)
        pred_label_all = np.concatenate(pred_label_all)
        gt_labels = np.concatenate(gt_label)

        loss_ave = np.mean(running_loss)

        # acc, recall, precision, auc, f1 = metrics_score(gt_labels, pred_label_all)
    # return loss_ave, acc, recall, precision, auc, f1
    return loss_ave, pred_prob_all, pred_label_all, gt_labels

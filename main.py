"""
这个main.py文件，用的是迁移学习方法一中的，加载预训练模型，训练全部参数
"""

import os
import torch
import pandas as pd
from torch import nn
from torchvision import transforms, datasets
import numpy as np
import torch.utils.data as DATA
import torch.optim as optim
import time
import matplotlib.pyplot as plt
from train import train
from config import configs
from models.model import resnet34, resnet101
from utils.util import gain_index, mkfile
from utils.visulization import plot_pd
from validate import eval


def cross_valid(
        date,
        num_epoch,
        dataset,
        k_fold,
        batch_size,
        num_classes,
        learning_rate,
        weight_decay,
        model_weight_path):
    total_size = len(dataset)
    fraction = 1 / k_fold
    seg = int(total_size * fraction)

    shuffle_dataset = True
    dataset_size = len(dataset)
    print(dataset_size)
    indices = list(range(dataset_size))
    print(indices)
    if shuffle_dataset:
        np.random.seed(42)
        np.random.shuffle(indices)
    print(indices)
    kfold_result = pd.DataFrame(columns=('Accurate', 'Recall', 'Precision', 'AUC', 'F1'))

    for i in range(k_fold):

        ################# 用迁移学习，改这个模块 ########################

        net = resnet34(num_classes=num_classes)
        net.cuda()

        ################################################################

        criterion = nn.CrossEntropyLoss()

        # optimizers = optim.SGD(net.parameters(),
        #                        lr=learning_rate,
        #                        momentum=0.9,
        #                        dampening=0,  # 动量的抑制因子，默认为0
        #                        weight_decay=weight_decay,  # 默认为0，有值说明用作正则化
        #                        nesterov=True, )  # 使用Nesterov动量，默认为False

        optimizers = optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)

        train_result = pd.DataFrame(columns=('Loss', 'Accurate'))
        val_result = pd.DataFrame(columns=('Loss', 'Accurate', 'Recall', 'Precision', 'AUC', 'F1'))

        print('\n', '*' * 10, 'Fold {}'.format(i + 1), '*' * 10)
        train_indices, val_indices = gain_index(i, seg, total_size, indices)
        print()
        print('train_indices\n', train_indices)
        print('val_indices\n', val_indices)

        train_sampler = DATA.sampler.SubsetRandomSampler(train_indices)
        valid_sampler = DATA.sampler.SubsetRandomSampler(val_indices)

        train_len, val_len = len(train_sampler), len(valid_sampler)
        print(train_len, val_len)
        print()

        train_loader = DATA.DataLoader(dataset,
                                       batch_size=batch_size,
                                       sampler=train_sampler,
                                       num_workers=4)
        validation_loader = DATA.DataLoader(dataset,
                                            batch_size=batch_size,
                                            sampler=valid_sampler,
                                            drop_last=True,
                                            num_workers=4)

        for epoch in range(num_epoch):
            print('\n-- Epoch {} --'.format(epoch + 1))

            net.train()
            train_loss, train_acc = train(net=net,
                                          train_loader=train_loader,
                                          train_num=train_len,
                                          loss_function=criterion,
                                          optimizer=optimizers)

            # 保存结果到DataFrame里面
            train_result = train_result.append(pd.DataFrame({'Loss': [train_loss],
                                                             'Accurate': [train_acc]}), ignore_index=True)

            net.eval()
            val_loss, val_acc, recall, precision, auc, f1 = eval(net=net,
                                                                 loss_function=criterion,
                                                                 validation_loader=validation_loader)

            print('train_loss:{} | train_acc:{}'.format(train_loss, train_acc))
            print('val_loss:{} | val_acc:{}'.format(val_loss, val_acc))
            # 保存结果到DataFrame里面
            val_result = val_result.append(pd.DataFrame({'Loss': [val_loss],
                                                         'Accurate': [val_acc],
                                                         'Recall': [recall],
                                                         'Precision': [precision],
                                                         'AUC': [auc],
                                                         'F1': [f1]}), ignore_index=True)

        kfold_result = val_result.append(pd.DataFrame({'Accurate': [val_acc],
                                                       'Recall': [recall],
                                                       'Precision': [precision],
                                                       'AUC': [auc],
                                                       'F1': [f1]}), ignore_index=True)

        save_dir = os.path.join(os.path.join('./result', date), 'main')
        mkfile(save_dir)
        train_file_name = 'K' + str(i + 1) + 'TrainScore.csv'
        val_file_name = 'K' + str(i + 1) + 'ValScore.csv'

        train_result.to_csv(os.path.join(save_dir, train_file_name))
        val_result.to_csv(os.path.join(save_dir, val_file_name))

    return train_result, val_result, kfold_result


if __name__ == '__main__':
    opt = configs()
    print("start time:", time.asctime(time.localtime(time.time())))
    # print(opt.introduction, '\n')
    data_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.2, 1), ratio=(0.5, 2)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15, resample=False, expand=False, center=None),
        transforms.ColorJitter(brightness=0, contrast=0.5, hue=0),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    datasets = datasets.ImageFolder(root=opt.data_path_train,
                                    transform=data_transform)

    train_score, val_score, kfold_score = cross_valid(date=opt.date,
                                         num_epoch=opt.num_epoch,
                                         dataset=datasets,
                                         k_fold=opt.kfold,
                                         batch_size=opt.batch_size,
                                         num_classes=opt.num_classes,
                                         learning_rate=opt.lr,
                                         weight_decay=opt.weight_decay,
                                         model_weight_path=opt.model_weight_path)

    # 只是print出第k折的结果
    print('train_result:\n', train_score)
    print('train describe:\n', train_score.describe())
    print('val_result:\n', val_score)
    print('val describe:\n', val_score.describe())
    # 存了k折每一折的最后一个epoch的平均准确率
    print('kfold_result:\n', kfold_score)
    print('kfold describe:\n', kfold_score.describe())

    f = plot_pd(train_score, val_score)
    save_dir = './result'
    mkfile(save_dir)
    # plt.savefig(os.path.join(save_dir, 'figure.png'))

    print("\nend time:", time.asctime(time.localtime(time.time())))

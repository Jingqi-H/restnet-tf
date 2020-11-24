import torch
import pandas as pd
from utils.metrics import pred_prob2pred_label
from torchvision import transforms
import torch.utils.data as DATA
import numpy as np
from torch import nn
from models.model import resnet34
from utils.util import gain_index
from dataset.dataset import CustomImageFolder

def eval(net, loss_function, validation_loader):
    running_loss = []
    pred_prob_all, pred_label_all, gt_label, name_all = [], [], [], []
    with torch.no_grad():
        for step, val_data in enumerate(validation_loader, start=0):
            val_images, val_labels, names = val_data
            outputs = net(val_images.cuda())  # eval model only have last output layer

            pred_prob, pred_label = pred_prob2pred_label(outputs)
            # print('val_labels:', val_labels)
            # print('pred_label:', pred_label)
            # print('pred_prob:\n', pred_prob)
            # print('outputs from network:\n', outputs)

            pred_label_all.append(pred_label)
            pred_prob_all.append(pred_prob)
            gt_label.append(val_labels)
            name_all.append(names)

            loss = loss_function(outputs, val_labels.cuda())
            running_loss.append(loss.item())

        pred_prob_all = np.concatenate(pred_prob_all)
        pred_label_all = np.concatenate(pred_label_all)
        gt_labels = np.concatenate(gt_label)
        name_all = np.concatenate(name_all)

        loss_ave = np.mean(running_loss)

        # acc, recall, precision, auc, f1 = metrics_score(gt_labels, pred_label_all)
    # return loss_ave, acc, recall, precision, auc, f1
    return loss_ave, pred_prob_all, pred_label_all, gt_labels, name_all

if __name__ == '__main__':

    num_epoch = 20
    seed = 42
    batch_size = 32
    k_fold = 3
    num_classes = 5
    pre_tained = False
    model_weight_path = '/home/huangjq/PyCharmCode/4_project/6_ResNet/5_ResNetTF34/checkpoints/1118Lr1e-5_Sgd/K3CP1000.pth'
    data_path_train = '/home/huangjq/PyCharmCode/1_dataset/1_glaucoma/v12/700_2100'
    data_transform = transforms.Compose([
        # transforms.RandomHorizontalFlip(),
        # AddPepperNoise(0.98, p=0.5),
        transforms.Resize((256, 512)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    dataset = CustomImageFolder(data_path_train, transform=data_transform)

    total_size = len(dataset)
    fraction = 1 / k_fold
    seg = int(total_size * fraction)

    shuffle_dataset = True
    dataset_size = len(dataset)
    print(dataset_size)
    indices = list(range(dataset_size))
    # print(indices)
    if shuffle_dataset:
        np.random.seed(seed)
        np.random.shuffle(indices)



    for i in range(k_fold):
        criterion = nn.CrossEntropyLoss().cuda()
        ################# 用迁移学习，改这个模块 ########################

        optim_param = []
        net = resnet34()
        missing_keys, unexpected_keys = net.load_state_dict(torch.load(model_weight_path), strict=False)
        for param in net.layer1.parameters():
            param.requires_grad = False

        for p in net.layer2.parameters():  # 将fine-tuning 的参数的 requires_grad 设置为 True
            p.requires_grad = False
            optim_param.append(p)
        for p in net.layer3.parameters():  # 将fine-tuning 的参数的 requires_grad 设置为 True
            p.requires_grad = True
            optim_param.append(p)
        for p in net.layer4.parameters():  # 将fine-tuning 的参数的 requires_grad 设置为 True
            p.requires_grad = True
            optim_param.append(p)

        inchannel = net.fc.in_features  # inchannel=512
        net.fc = nn.Sequential(
            nn.Linear(inchannel, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes),
        )
        net.cuda()

        ################################################################

        train_indices, val_indices = gain_index(i, seg, total_size, indices)

        train_sampler = DATA.sampler.SubsetRandomSampler(train_indices)
        valid_sampler = DATA.sampler.SubsetRandomSampler(val_indices)

        train_len, val_len = len(train_sampler), len(valid_sampler)
        print(train_len, val_len, '\n')

        train_loader = DATA.DataLoader(dataset,
                                       batch_size=batch_size,
                                       sampler=train_sampler,
                                       num_workers=4)
        validation_loader = DATA.DataLoader(dataset,
                                            batch_size=batch_size,
                                            sampler=valid_sampler,
                                            drop_last=True,
                                            num_workers=4)

        # start validation
        net.eval()
        val_loss, pred_prob, pred_labels, gt_labels, val_names = eval(net=net,
                                                                      loss_function=criterion,
                                                                      validation_loader=validation_loader)
        print('pred_labels:{} \n gt_labels:{} \n val_names:{}'.format(pred_labels, gt_labels, val_names))

        # 将list转为DataFrame
        val_img_names = pd.DataFrame({
            'img_name': val_names,
            'gt_index': gt_labels,
            'pred_index': pred_labels
        })
        print('val_csv', val_img_names)

        # val_img_names = val_img_names.append()
        # train_result = train_result.append(pd.DataFrame({'Loss': [train_loss],
        #                                                  'Accurate': [train_acc]}), ignore_index=True)
        break

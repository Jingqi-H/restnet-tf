import numpy as np
from itertools import cycle
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import pandas as pd
import matplotlib.pyplot as plt
import os
from utils.util import mkfile



import matplotlib.pyplot as plt

def plot_lr(lr_list):
    # lr_list = [9.827660231920668e-06, 9.647126956713013e-06, 9.466217490987827e-06, 9.284923027753324e-06, 9.103234357811095e-06, 8.921141842035945e-06, 8.738635381071262e-06, 8.555704382133257e-06, 8.372337722572737e-06, 8.1885237097908e-06, 8.004250037043245e-06, 7.819503734595651e-06, 7.63427111560465e-06, 7.448537715997826e-06, 7.262288227501316e-06, 7.075506422815729e-06, 6.888175071761446e-06, 6.700275846996265e-06, 6.511789217641672e-06, 6.3226943288262385e-06, 6.132968864749162e-06, 5.942588892362033e-06, 5.751528682133769e-06, 5.559760501563623e-06, 5.3672543760884065e-06, 5.1739778107212e-06, 4.979895464061975e-06, 4.784968764098667e-06, 4.589155452276317e-06, 4.392409038373951e-06, 4.194678143388581e-06, 3.995905700283114e-06, 3.796027972205331e-06, 3.5949733332371227e-06, 3.3926607356998547e-06, 3.1889977570015337e-06, 2.9838780721377813e-06, 2.777178125315112e-06, 2.568752658212602e-06, 2.358428560937516e-06, 2.1459961829028056e-06, 1.9311966493358395e-06, 1.713702604478028e-06, 1.493087514809159e-06, 1.2687736014565799e-06, 1.0399359634069778e-06, 8.053045997236955e-07, 5.626781045616927e-07, 3.0729964689139947e-07, 1.759429800836174e-08]

    # print(lr_list)
    x = []
    for i in range(len(lr_list)):
        x.append(i + 1)
    plt.figure(figsize=(20, 10))
    plt.plot(x, lr_list, label='loss', linewidth=2.0)

    plt.legend()
    plt.xlabel('num', fontsize=24)
    plt.ylabel('loss', fontsize=24)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    # plt.title('name', fontsize=24)
    plt.grid(linestyle='-.')

    # plt.savefig('./test.png')
    # plt.show()



def plot_dataframe(train_file, val_file):
    """
    一张画布上有两个图，
    :param train_file:
    :param val_file:
    :return:
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    ax1 = axes[0]
    ax2 = axes[1]

    # “.loc[:3000]”可以删掉，这个是用来限制行数的
    ax1.plot(train_file['Accurate'].loc[:3000], '-', label='train')
    ax1.plot(val_file['Accurate'].loc[:3000], '--', label='val')
    ax1.set_xlabel('Epoch', fontsize=15)
    ax1.set_ylabel('Accurate', fontsize=15)
    # ax1.legend('train', fontsize=15)
    ax1.grid(linestyle=":")
    ax1.tick_params(labelsize=12)

    ax2.plot(train_file['Loss'].loc[:3000], '-', label='train')
    ax2.plot(val_file['Loss'].loc[:3000], '--', label='val')
    ax2.set_xlabel('Epoch', fontsize=15)
    ax2.set_ylabel('Loss', fontsize=15)
    # ax2.legend(legend_train, fontsize=15)
    ax2.grid(linestyle=":")
    ax2.tick_params(labelsize=12)
    # plt.show()

    return fig


def plot_trainval_lossacc(dir_main32_root):
    train_file = []
    val_file = []
    legend_train, legend_val = [], []

    # 获得每个csv文件的DataFrame，存到list中
    for i in range(3):
        name_train = 'K' + str(i + 1) + 'TrainScore.csv'
        name_val = 'K' + str(i + 1) + 'ValScore.csv'
        train_file.append(pd.read_csv(os.path.join(dir_main32_root, name_train), encoding='gbk'))
        val_file.append(pd.read_csv(os.path.join(dir_main32_root, name_val), encoding='gbk'))
        legend_train.append('k' + str(i + 1) + '_train')
        legend_val.append('k' + str(i + 1) + '_val')

    # 输出k折的平均结果
    kflod_file = pd.read_csv(os.path.join(dir_main32_root, 'KfoldScore.csv'), encoding='gbk')
    print(kflod_file.describe())

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    ax1 = axes[0]
    ax2 = axes[1]

    # “.loc[:3000]”可以删掉，这个是用来限制行数的
    ax1.plot(train_file[0]['Accurate'].loc[:3000], '-')
    ax1.plot(train_file[1]['Accurate'].loc[:3000], '-')
    ax1.plot(train_file[2]['Accurate'].loc[:3000], '-')
    ax1.plot(val_file[0]['Accurate'].loc[:3000], '--')
    ax1.plot(val_file[1]['Accurate'].loc[:3000], '--')
    ax1.plot(val_file[2]['Accurate'].loc[:3000], '--')
    ax1.set_xlabel('Epoch', fontsize=15)
    ax1.set_ylabel('Accurate', fontsize=15)
    legend_train += legend_val
    ax1.legend(legend_train, fontsize=15)
    ax1.grid(linestyle=":")
    ax1.tick_params(labelsize=12)

    ax2.plot(train_file[0]['Loss'].loc[:3000], '-')
    ax2.plot(train_file[1]['Loss'].loc[:3000], '-')
    ax2.plot(train_file[2]['Loss'].loc[:3000], '-')
    ax2.plot(val_file[0]['Loss'].loc[:3000], '--')
    ax2.plot(val_file[1]['Loss'].loc[:3000], '--')
    ax2.plot(val_file[2]['Loss'].loc[:3000], '--')
    ax2.set_xlabel('Epoch', fontsize=15)
    ax2.set_ylabel('Loss', fontsize=15)
    ax2.legend(legend_train, fontsize=15)
    ax2.grid(linestyle=":")
    ax2.tick_params(labelsize=12)
    # plt.show()

    return fig


def plot_confusion_matrix(k, cm, save_results, epoch, title='Confusion Matrix'):
    classes = ['N1', 'N2', 'N3', 'N4', 'W']
    plt.cla()
    plt.close('all')
    plt.figure(figsize=(12, 8), dpi=100)
    np.set_printoptions(precision=2)

    # 在混淆矩阵中每格的概率值
    ind_array = np.arange(len(classes))
    x, y = np.meshgrid(ind_array, ind_array)
    # 每个框里的数字
    for x_val, y_val in zip(x.flatten(), y.flatten()):
        c = cm[y_val][x_val]
        if c > 0.001:
            plt.text(x_val, y_val, "%0.0f" % (c,), color='black', fontsize=15, va='center', ha='center')

    #     cmap 设置混淆矩阵的颜色
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    #     plt.imshow(confusion, cmap=plt.cm.Blues)
    # plt.title(title)
    plt.colorbar()
    xlocations = np.array(range(len(classes)))
    # plt.xticks(xlocations, classes, rotation=45)
    plt.xticks(xlocations, classes, rotation=0)
    plt.yticks(xlocations, classes)
    plt.ylabel('Predicted Labels')
    plt.xlabel('True Labels')

    # offset the tick
    tick_marks = np.array(range(len(classes))) + 0.5
    plt.gca().set_xticks(tick_marks, minor=True)
    plt.gca().set_yticks(tick_marks, minor=True)
    plt.gca().xaxis.set_ticks_position('none')
    plt.gca().yaxis.set_ticks_position('none')
    plt.grid(True, which='minor', linestyle='-')
    plt.gcf().subplots_adjust(bottom=0.15)

    #    show confusion matrix
    save_name = 'K' + str(k) + 'CP' + str(epoch) + '_confusion_matrix' + '.png'
    plt.savefig(os.path.join(save_results, save_name))
    # plt.show()


def plot_roc(k, y_score, y_test, save_results, epoch):
    """
    根据sklearn参考文档
    :param y_score:是得到预测结果，他是概率值，并且是array
    :param y_test:是gt
    :param save_results: 保存路径
    :return:
    """

    # y_score = torch.rand([30, 5]).numpy()
    # # y = torch.tensor([1, 0, 0, 4, 1, 3, 0, 3, 4, 4, 3, 2, 0, 2, 3, 4, 1, 1, 1, 4, 3, 0, 0, 0,
    # #         1, 1, 0, 0, 2, 2])
    # y_test = torch.tensor([1, 1, 1, 4, 1, 4, 0, 3, 4, 4, 2, 2, 0, 3, 3, 3, 1, 1, 1, 4, 3, 0, 0, 0,
    #                        1, 0, 0, 0, 3, 2])

    y_test = label_binarize(y_test, classes=[0, 1, 2, 3, 4])
    n_classes = y_test.shape[1]
    # print(n_classes)

    # y_test是二值，y_score是概率
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        # print(i)
        # print(y_test[:, i])
        # print(y_score[:, i])
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        #     fpr[i], tpr[i], _ = roc_curve(y_test, y_score)
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    plt.cla()
    plt.close('all')
    plt.figure()
    # micro：多分类；macro：计算二分类metrics的均值，为每个类给出相同权重的分值。
    plt.plot(fpr["micro"], tpr["micro"],
             label='average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    # plt.plot(fpr["macro"], tpr["macro"],
    #          label='macro-average ROC curve (area = {0:0.2f})'
    #                ''.format(roc_auc["macro"]),
    #          color='navy', linestyle=':', linewidth=4)

    colors = cycle(['gold', '#1E90FF', '#FF6347', '#9370DB', '#228B22'])
    classes = ['N1', 'N2', 'N3', 'N4', 'W']
    lw = 2
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of {0} (area = {1:0.2f})'
                       ''.format(classes[i], roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    # plt.title('Some extension of Receiver operating characteristic to multi-class')
    plt.legend(loc="lower right")

    save_name = 'K' + str(k) + 'CP' + str(epoch) + '_roc' + '.png'
    plt.savefig(os.path.join(save_results, save_name))
    # plt.show()


'''

def plot_pd(train_df, val_df):
    """

    df:验证分数，DataFrame
    """

    fig, axes = plt.subplots(3, 2, figsize=(7, 7))
    ax1 = axes[0, 0]
    ax2 = axes[0, 1]
    ax3 = axes[1, 0]
    ax4 = axes[1, 1]
    ax5 = axes[2, 0]
    ax6 = axes[2, 1]
    ax1.plot(val_df['Loss'], label='val loss')
    ax1.plot(train_df['Loss'], 'g', label='train loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('loss')
    ax1.legend()  # 加图例

    ax2.plot(val_df['Accurate'], label='val acc')
    ax2.plot(train_df['Accurate'], 'g',  label='train acc')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accurate')
    ax2.legend()

    ax3.plot(val_df['Recall'])
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Recall')

    ax4.plot(val_df['Precision'])
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Precision')

    ax5.plot(val_df['AUC'])
    ax5.set_xlabel('Epoch')
    ax5.set_ylabel('AUC')

    ax6.plot(val_df['F1'])
    ax6.set_xlabel('Epoch')
    ax6.set_ylabel('F1')

    plt.show()
    # return fig
    
'''

if __name__ == '__main__':
    dir_main32_root = '/home/huangjq/PyCharmCode/4_project/6_ResNet/5_ResNetTF34/result/1118Lr1e-5_Sgd/main3.2'
    figure = plot_trainval_lossacc(dir_main32_root)

    # save_dir = os.path.join(os.path.join('./result', opt.date), 'main321')
    save_dir = os.path.join(os.path.join('./result', 'demo'), 'main321')
    mkfile(save_dir)
    plt.savefig(os.path.join(save_dir, 'trainval_lossacc.png'))

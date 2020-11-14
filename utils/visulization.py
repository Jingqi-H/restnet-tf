import pandas as pd
import matplotlib.pyplot as plt
import os
from utils.util import mkfile


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
    ax1.legend() # 加图例

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

if __name__ == '__main__':

    val_root = '/home/huangjq/PyCharmCode/4_project/6_ResNet/5_ResNetTF34/result/1112Lr1e-6wh_300-900/main3.2/K3ValScore.csv'
    train_root = '/home/huangjq/PyCharmCode/4_project/6_ResNet/5_ResNetTF34/result/1112Lr1e-6wh_300-900/main3.2/K3TrainScore.csv'
    train_file = pd.read_csv(train_root, encoding='gbk')
    val_file = pd.read_csv(val_root, encoding='gbk')
    f = plot_pd(train_file, val_file)

    # save_dir = '../result'
    # mkfile(save_dir)
    # plt.savefig(os.path.join(save_dir, 'figure.png'))



    # dir_root = '/home/huangjq/PyCharmCode/4_project/6_ResNet/5_ResNetTF34/result/'
    # for i in range(10):
    #     file_name = 'K' + str(i+1) + 'ValScore.csv'
    #     file_root = os.path.join(dir_root, file_name)
    #
    #     file = pd.read_csv(file_root, encoding='gbk')
    #     file.plot()
    #     plt.show()
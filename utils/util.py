import os


def gain_index(k, seg, total_size, indices):
    """
    tr:train,val:valid; r:right,l:left;  eg: trrr: right index of right side train subset
    index: [trll,trlr],[vall,valr],[trrl,trrr]
    :param k:
    :param seg:
    :param total_size:
    :param indices:
    :return:
    """
    trll = 0
    trlr = k * seg
    vall = trlr
    valr = k * seg + seg
    trrl = valr
    trrr = total_size

    print("train indices: [%d,%d),[%d,%d), val indices: [%d,%d)"
          % (trll, trlr, trrl, trrr, vall, valr))
    train_indices = indices[trll:trlr] + indices[trrl:trrr]
    val_indices = indices[vall:valr]

    return train_indices, val_indices


# 若文件夹不存在，则创建新的文件夹
def mkfile(file):
    if not os.path.exists(file):
        os.makedirs(file)

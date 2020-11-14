


class configs(object):

    kfold = 3
    data_path_train = '/home/huangjq/PyCharmCode/1_dataset/1_glaucoma/v12/700_2100'
    validation_split = .2
    shuffle_dataset = True

    lr = 1e-5
    weight_decay = 0.0005  # 缓解过拟合
    # lr_decay = 0.99995
    # patience = 50
    # weight_decay = 0.0

    batch_size = 32

    num_epoch = 1000
    # num_epoch = 2

    num_classes = 5

    model_weight_path = "./resnet34-pre.pth"
    date = '1114Lr1e-5wh_300-900'

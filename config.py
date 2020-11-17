


class configs(object):

    kfold = 3
    data_path_train = '/home/huangjq/PyCharmCode/1_dataset/1_glaucoma/v12/change_to_same_angle'
    validation_split = .2
    shuffle_dataset = True

    lr = 1e-6
    weight_decay = 0.0005  # 缓解过拟合
    # lr_decay = 0.99995
    # patience = 50
    # weight_decay = 0.0

    batch_size = 32
    num_epoch = 3000
    # num_epoch = 2

    num_classes = 5

    model_weight_path = "./resnet34-pre.pth"
    date = '1116Lr1e-6_noaugumentation_epoch3_wh300-900'

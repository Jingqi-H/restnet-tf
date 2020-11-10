


class configs(object):

    kfold = 3
    data_path_train = '/home/huangjq/PyCharmCode/4_project/paper/A_Grading_Method_of_Glaucoma_based_on_Anterior_Chamber_Angle_Image/data_demo/1106_classification'
    validation_split = .2
    shuffle_dataset = True

    lr = 1e-6
    weight_decay = 0.0005  # 缓解过拟合
    # lr_decay = 0.99995
    # patience = 50
    # weight_decay = 0.0

    batch_size = 32
    num_epoch = 1000
    num_classes = 5

    model_weight_path = "./resnet34-pre.pth"
    date = '1106Lr1e-6newdata'
    # save_path = "./checkpoints"
    # save_epoch = 50

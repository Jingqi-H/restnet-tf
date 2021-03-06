class configs(object):
    kfold = 5
    # data_path_train = '/home/huangjq/PyCharmCode/1_dataset/1_glaucoma/v13/classification_data/data'
    data_path_train = '/home/huangjq/PyCharmCode/1_dataset/1_glaucoma/v13/classification_data/origin/data_from_yolo'
    # data_path_train = '/home/huangjq/PyCharmCode/1_dataset/1_glaucoma/v12/700_2100'
    seed = 42

    lr = 1e-5
    num_epoch = 3000
    date = '1210Lr1e-5_Sgd+K5_DatafromYOLO'

    num_classes = 5
    batch_size = 32
    weight_decay = 0.0005  # 缓解过拟合

    # ImageNet在resnet上的预训练网络
    model_weight_path = "/home/huangjq/PyCharmCode/4_project/frequently_used/pretrained_model/resnet34-pre.pth"

    # 判断是否要在自己训练好的网络上继续训练，k=3折，每次都要拿对应折的网络接着训练，随机种子也必须一致
    pre_tained = False  # 是否要使用自己预训练的网络，不用的话要改成False
    pre_tained_model = '/home/huangjq/PyCharmCode/4_project/6_ResNet/5_ResNetTF34/checkpoints/1202Lr1e-5_Sgd_NewData+K5'
    # pre_tained_model = '/home/huangjq/PycharmProjects/5_ResNetTF34/pretained_model/from60-1118Lr1e-5_Sgd'
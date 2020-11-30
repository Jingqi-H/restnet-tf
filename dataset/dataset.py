from torchvision.datasets import ImageFolder
from torchvision import transforms
import torch.utils.data as DATA
import matplotlib.pyplot as plt
import numpy as np
from utils.util import gain_index

class CustomImageFolder(ImageFolder):
    """
        为了获得batchsize的每个图片名字
    """

    def __init__(self, root, transform=None):
        super(CustomImageFolder, self).__init__(root, transform)

    def __getitem__(self, index):
        path = self.imgs[index][0]  # 此时的self.imgs等于self.samples，即内容为[(图像路径, 该图像对应的类别索引值),(),...]
        label = self.imgs[index][1]
        #         print('image name: ', path.split("/")[-1])
        #         print('image name: ', path)

        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)

        return img, label, path.split("/")[-1]


if __name__ == '__main__':


    num_epoch = 20
    seed = 42
    batch_size = 32
    k_fold = 3

    data_path_train = '/home/huangjq/PyCharmCode/1_dataset/1_glaucoma/v12/700_2100'
    data_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        # AddPepperNoise(0.98, p=0.5),
        transforms.Resize((256, 512)),
        transforms.ToTensor(),
        #     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # dataset = datasets.ImageFolder(data_path_train, transform=data_transform)
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

        train_indices, val_indices = gain_index(i, seg, total_size, indices)

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
            for step, data in enumerate(validation_loader, start=0):
                images, labels, names = data
                #         print(transforms.ToPILImage()(images[0]))
                plt.imshow(transforms.ToPILImage()(images[0]))
                plt.show()
                print(labels)
                print(names)

            #         break

            break



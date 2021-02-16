from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import random
import torch
from PIL import Image
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from dataPrepare.my_prepare_data import calculate_pitch_yaw_roll


class MyDataset(Dataset):
    def __init__(self, dataset_dir, seed=None, mode="train", train_val_ratio=0.9, trans=None):
        if seed is None:
            seed = random.randint(0, 65536)
        random.seed(seed)
        self.dataset_dir = dataset_dir
        self.mode = mode
        if mode=="val":
            mode = "train"
        self.img_dir = os.path.join(dataset_dir, mode)  # 图片存储的文件夹
        img_list_txt = os.path.join(dataset_dir, mode+".txt")  # 储存图片位置的列表
        label_csv = os.path.join(dataset_dir, mode+".csv")  # 储存标签的数组文件
        self.img_list = []
        self.label = np.loadtxt(label_csv, delimiter=',')  # 读取标签数组文件
        # 读取图片位置文件
        with open(img_list_txt, 'r') as f:
            for line in f.readlines():
                self.img_list.append(line.strip())
        # 在mode=train或val时， 将数据进行切分
        # 注意在mode="val"时，传入的随机种子seed要和mode="train"相同
        self.num_all_data = len(self.img_list)
        all_ids = list(range(self.num_all_data))
        num_train = int(train_val_ratio*self.num_all_data)
        if self.mode == "train":
            self.use_ids = all_ids[:num_train]
        elif self.mode == "val":
            self.use_ids = all_ids[num_train:]
        else:
            self.use_ids = all_ids

        # 储存数据增广函数
        self.trans = trans

    def __len__(self):
        return len(self.use_ids)

    def __getitem__(self, item):
        id = self.use_ids[item]
        label = torch.tensor(self.label[id, :])
        img_path = self.img_list[id]
        img = Image.open(img_path)
        if self.trans is None:
            trans = transforms.Compose([
                transforms.Resize((112, 112)),
                transforms.ToTensor(),
            ])
            img = trans(img)
        else:
            trans = self.trans
            img, label = trans(img, label)  # 图像预处理&数据增广
        # os.system("pause")
        # transforms.ToPILImage()(img).show()  # for debug
        # print(label)
        return img, label


class DataAugmentation(object):
    """
    用于数据增广的类
    """
    def __init__(self):
        """
        :param methods: list[str], 传入数据增广的方法列表，用字符串描述，可选参数如下：
                "random_horizontal_flip" : 随机水平翻转
                self._random_horizontal_flip,
                self._get_angles,`
        """
        self.methods = [

                        self._my_resize,
                        self._my_totensor]

    def __call__(self, img, labels):
        """
        调用数据增广列表的函数
        :param img: 待处理图像
        :param labels:  对应的样本标签
        :return: 处理后的图像和标签
        """
        for method in self.methods:
            img, labels = method(img, labels)
        return img, labels

    @staticmethod
    def _random_horizontal_flip(img, labels, p=0.5):
        """
        概率p=0.5，随机水平翻转，并处理标签
        :param img: PIL.Image类型, 输入图像
        :param labels: 图像对应的标签(已归一化)
        :param p: 水平翻转概率
        :return: 处理后的img和labels
        """
        isFlip = (random.random() > p)
        if isFlip:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            kps = labels[:196]
            kps = kps.reshape(-1, 2)
            kps[:, 0] = 1 - kps[:, 0]
            kps = kps.reshape(-1)
            labels[:196] = kps
            if False:
                imgdraw = ImageDraw.Draw(img)
                imgsize = img.size[0]
                for i in range(kps.shape[0]//2):
                    x = int(imgsize*kps[i*2])
                    y = int(imgsize*kps[i*2+1])
                    imgdraw.point((x,y), (255,0,0))
        return img, labels

    @staticmethod
    def _my_resize(img, labels):
        img = img.resize((112,112), Image.BICUBIC)
        return img, labels

    @staticmethod
    def _my_totensor(img, labels):
        img = F.to_tensor(img)
        return img, labels

    @staticmethod
    def _get_angles(img, labels):
        """
        获取经过图像预处理后的欧拉角
        """
        kps = labels[:196]
        kps = kps.reshape(-1, 2)
        kps = kps.data.numpy()
        angles = calculate_pitch_yaw_roll(kps, n_points=98)
        labels[206] = angles[0]
        labels[207] = angles[1]
        labels[208] = angles[2]
        return img, labels

if __name__ == '__main__':
    dataset_dir = "I:\Dataset\WFLW\WFLW_for_PFLD"
    dataset = MyDataset(dataset_dir)
    dataloader = DataLoader(dataset, 1)
    for i in enumerate(dataloader):
        input("press enter to continue")
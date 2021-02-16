import torch
import torch.nn as nn
from numpy import pi
from util import DepthwiseConv, InvertedBottleneck, Flatten, Conv2d_withBN


class MyNet(nn.Module):
    """
    PFLDNet
    """
    def __init__(self, n_points=98):
        """
        :param n_points: 关键点个数，默认98
        """
        super(MyNet, self).__init__()
        self.n_points = n_points
        self.backbone1 = nn.Sequential(
            Conv2d_withBN(in_channels=3, out_channels=64, kernel_size=3, stride=2, padding=1, bias=True),
            DepthwiseConv(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True),

            InvertedBottleneck(in_channels=64, out_channels=64, kernel_size=3, stride=2, t_factor=2, padding=1, bias=True),
            InvertedBottleneck(in_channels=64, out_channels=64, kernel_size=3, stride=1, t_factor=2, padding=1,
                               bias=True),
            InvertedBottleneck(in_channels=64, out_channels=64, kernel_size=3, stride=1, t_factor=2, padding=1,
                               bias=True),
            InvertedBottleneck(in_channels=64, out_channels=64, kernel_size=3, stride=1, t_factor=2, padding=1,
                               bias=True),
            InvertedBottleneck(in_channels=64, out_channels=64, kernel_size=3, stride=1, t_factor=2, padding=1,
                               bias=True),
        )

        self.backbone2 = nn.Sequential(
            InvertedBottleneck(in_channels=64, out_channels=128, kernel_size=3, stride=2, t_factor=2, padding=1,
                               bias=True),

            InvertedBottleneck(in_channels=128, out_channels=128, kernel_size=3, stride=1, t_factor=4, padding=1,
                               bias=True),
            InvertedBottleneck(in_channels=128, out_channels=128, kernel_size=3, stride=1, t_factor=4, padding=1,
                               bias=True),
            InvertedBottleneck(in_channels=128, out_channels=128, kernel_size=3, stride=1, t_factor=4, padding=1,
                               bias=True),
            InvertedBottleneck(in_channels=128, out_channels=128, kernel_size=3, stride=1, t_factor=4, padding=1,
                               bias=True),
            InvertedBottleneck(in_channels=128, out_channels=128, kernel_size=3, stride=1, t_factor=4, padding=1,
                               bias=True),
            InvertedBottleneck(in_channels=128, out_channels=128, kernel_size=3, stride=1, t_factor=4, padding=1,
                               bias=True),

            InvertedBottleneck(in_channels=128, out_channels=16, kernel_size=3, stride=1, t_factor=2, padding=1,
                               bias=True),
        )

        self.avgpool1 = nn.AvgPool2d(14)
        self.S1 = nn.Sequential(
            Conv2d_withBN(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1, bias=True),
        )
        self.avgpool2 = nn.AvgPool2d(7)

        self.S2 = nn.Sequential(
            # nn.Conv2d(in_channels=32, out_channels=128, kernel_size=7, stride=1, padding=0, bias=True),
            # nn.ReLU(inplace=True)
            Conv2d_withBN(in_channels=32, out_channels=128, kernel_size=7, stride=1, padding=0, bias=True)
        )
        self.backbone_fc = nn.Linear(in_features=176, out_features=n_points*2, bias=True)  # 128+32+16=4832

        self.auxiliary_net = nn.Sequential(
            Conv2d_withBN(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1, bias=True),
            Conv2d_withBN(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=True),
            Conv2d_withBN(in_channels=128, out_channels=32, kernel_size=3, stride=2, padding=1, bias=True),
            Conv2d_withBN(in_channels=32, out_channels=128, kernel_size=7, stride=1, padding=0, bias=True),
            Flatten(),
            nn.Linear(in_features=128, out_features=32, bias=True),
            nn.Linear(in_features=32, out_features=3, bias=True)
        )

    def forward(self, inputs):
        n_samples = inputs.shape[0]
        x = self.backbone1(inputs)
        self.aux_result = self.auxiliary_net(x)
        x = self.backbone2(x)
        s1_result = self.avgpool1(x)
        x = self.S1(x)
        s2_result = self.avgpool2(x)
        s3_result = self.S2(x)
        s1_result = s1_result.reshape(n_samples, -1)
        s2_result = s2_result.reshape(n_samples, -1)
        s3_result = s3_result.reshape(n_samples, -1)
        x = torch.cat((s1_result, s2_result, s3_result), dim=1)
        self.backbone_result = self.backbone_fc(x)
        return self.backbone_result

    def calculate_loss(self, labels):
        """
        计算训练loss
        :param labels: (batchsize, 209), 0~195 98个关键点坐标， 196~199 bbox, 200~205 6个属性， 206~208 3个欧拉角(角度制)
        :return: loss数值
        """
        batchsize = labels.shape[0]
        id1 = self.n_points*2
        id2 = id1 + 4
        gt_kps = labels[:, :id1]
        gt_attrs = labels[:, id2:id2+6]
        gt_angle = labels[:, -3:]/180*pi
        self.backbone_result = self.backbone_result.double()
        self.aux_result = self.aux_result.double()
        l2_distance = torch.sum((self.backbone_result - gt_kps)**2, dim=1)  # 计算关键点的L2loss, (bs, 1)
        weight_angle = torch.sum(1-torch.cos(self.aux_result-gt_angle), dim=1).reshape(batchsize, 1)  # 计算 sum(1-cos theta^k), (bs, 1)
        # 计算每个属性在该batchsize中所占的比例
        ratio = torch.mean(gt_attrs, dim=0).reshape(1, 6)
        ratio[ratio==0] = 1.0
        ratio = 1.0 / ratio  # (1, 6)
        weight_attrs = torch.sum(torch.mul(gt_attrs, ratio), dim=1)  # (bs, 1)
        loss = torch.mean(weight_angle * weight_attrs * l2_distance)
        return loss

    def calculate_metric(self, preds, labels):
        preds = preds.double()
        labels = labels[:, :(self.n_points*2)]
        l2_distance = torch.mean(torch.sum((preds-labels)**2, dim=1))
        return l2_distance


if __name__ == '__main__':
    import numpy as np
    x = torch.zeros(5,3,112,112)
    net = MyNet()
    a = net(x)
    labels = torch.zeros(5, 205)
    loss = net.calculate_loss(labels)
    print(loss)
    print(a.shape)

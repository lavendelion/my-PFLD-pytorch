import os
from my_arguments import Args
import torch
from torch.utils.data import DataLoader

from model import MyNet
from data import MyDataset
import torchvision.transforms as transforms
from PIL import Image
import cv2


class TestInterface(object):
    def __init__(self, opts):
        self.opts = opts
        print("=======================Start inferring.=======================")

    def main(self):
        opts = self.opts
        img_list = os.listdir(opts.dataset_dir)
        trans = transforms.Compose([
            transforms.Resize((112, 112)),
            transforms.ToTensor(),
        ])
        model = torch.load(opts.weight_path)
        if opts.use_GPU:
            model.to(opts.GPU_id)
        for img_name in img_list:
            img_path = os.path.join(opts.dataset_dir, img_name)
            img = Image.open(img_path).convert('RGB')
            img = trans(img)
            img = torch.unsqueeze(img, dim=0)
            if opts.use_GPU:
                img = img.to(opts.GPU_id)
            preds = model(img)
            res_img = cv2.imread(img_path)
            img_size = res_img.shape[0]
            for i in range(preds.shape[1]//2):
                point = (int(img_size*preds[0,i*2]), int(img_size*preds[0,i*2+1]))
                if point[0]<0 or point[1]<0:
                    print("wrong (x,y)=(%d, %d) in point %d"%(point[0], point[1], i))
                cv2.circle(res_img, point, 2, (255,0,0), 2)
            cv2.imshow("res_img", res_img)
            cv2.waitKey(0)


if __name__ == '__main__':
    args = Args()
    args.set_test_args()
    test_interface = TestInterface(args.get_opts())
    test_interface.main()
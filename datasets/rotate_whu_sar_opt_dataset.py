import cv2
import os
from torch.utils.data import Dataset,DataLoader
import random
import torch
import numpy as np
from .utils import *
from skimage import io, color


class RotateWHUDataset(Dataset):
    def __init__(self,data_file,size=(320,320),stride=8, aug=True):
        self.data_file = data_file
        with open(data_file, 'r') as f:
            self.train_data = f.readlines()

        self.size = size
        self.aug = aug
        self.stride = stride # for generating gt-mask needed to compute local-feature loss
        self.opt_mean = np.array([0.41045055, 0.37915975, 0.31014156],dtype=np.float32).reshape(3,1,1)
        self.opt_std = np.array([0.04182153, 0.04993042, 0.06234434],dtype=np.float32).reshape(3,1,1)
        self.sar_mean = np.array([0.21184826, 0.21184082, 0.21182437],dtype=np.float32).reshape(3,1,1)
        self.sar_std = np.array([0.15666042, 0.15666236, 0.15666533],dtype=np.float32).reshape(3,1,1)


    def _read_file_paths(self,data_dir):
        assert os.path.isdir(data_dir), "%s should be a dir which contains images only"%data_dir
        file_paths = os.listdir(data_dir)
        return file_paths

    def __getitem__(self, index: int):
        sar = self.train_data[index].strip('\n')
        #sar = sar.replace('trans', 'trans2')
        sar_img = np.array(Image.open(sar).convert('RGB'))
        opt_img = np.array(Image.open(sar.replace('_sar', '_opt')).convert('RGB'))
        opt_img_src = opt_img.copy()
        #os.makedirs('test_images', exist_ok=True)
        #cv2.imwrite(f"test_images/{index}_sar.jpg", cv2.cvtColor(sar_img, cv2.COLOR_RGB2BGR))
        #cv2.imwrite(f"test_images/{index}_opt.jpg", cv2.cvtColor(opt_img, cv2.COLOR_RGB2BGR))

        gt = arr = np.loadtxt(f'{os.path.dirname(sar)}/mat.txt', delimiter=',').squeeze()
        query = sar_img.transpose(2,0,1)
        refer = opt_img.transpose(2,0,1)

        query = ((query / 255.0) - self.sar_mean) / self.sar_std
        refer = ((refer / 255.0) - self.opt_mean) / self.opt_std

        refer_src = opt_img_src.transpose(2,0,1)
        refer_src = ((refer_src / 255.0) - self.opt_mean) / self.opt_std

        sample = {
            "refer":refer,
            "query":query,
            "refer_src":refer_src,
            "H_gt": gt

            # "M": M,
            # "Mr": Mr,
            # "Mq": Mq
        }
        return sample

            
    def __len__(self):
        return len(self.train_data)



def build_Rotate_WHU(
        train_data_file,
        test_data_file,
        size,
        stride):
    train_data = RotateWHUDataset(
        train_data_file,
        size=(320, 320),
        stride=8,
        aug=True)
    test_data = RotateWHUDataset(
        test_data_file,
        size=(320, 320),
        stride=8,
        aug=False)

    return train_data, test_data



if __name__ == "__main__":
    from utils import _transform_inv,draw_match
    size = (320,320)
    dataloader = DataLoader(
        RotateWHUDataset("/home/ly/Documents/zkj/dataset/whu-opt-sar/whu_train.txt",size=size, aug=True),
        batch_size=4,
        shuffle=True,
        num_workers=0,
        pin_memory=True)
    print(len(dataloader))
    opt_mean = np.array([0.41045055, 0.37915975, 0.31014156],dtype=np.float32).reshape(3,1,1)
    opt_std = np.array([0.04182153, 0.04993042, 0.06234434],dtype=np.float32).reshape(3,1,1)
    sar_mean = np.array([0.21184826, 0.21184082, 0.21182437],dtype=np.float32).reshape(3,1,1)
    sar_std = np.array([0.15666042, 0.15666236, 0.15666533],dtype=np.float32).reshape(3,1,1)
    check_index = 0
    num = 0
    while 1:
        for sample in dataloader:
            query,refer,label_matrix = sample["query"],sample["refer"],sample["gt_matrix"]
            query0 = query.detach().cpu().numpy()[check_index]
            refer0 = refer.detach().cpu().numpy()[check_index]
            label_matrix0 = label_matrix.detach().cpu().numpy()[check_index]
            query1 = query.detach().cpu().numpy()[check_index+1]
            refer1 = refer.detach().cpu().numpy()[check_index+1]
            label_matrix1 = label_matrix.detach().cpu().numpy()[check_index+1]

            sq0 = _transform_inv(query0,sar_mean,sar_std)
            sr0 = _transform_inv(refer0,opt_mean,opt_std)
            out0 = draw_match(label_matrix0>0,sq0,sr0).squeeze()
            sq1 = _transform_inv(query1,sar_mean,sar_std)
            sr1 = _transform_inv(refer1,opt_mean,opt_std)
            out1 = draw_match(label_matrix1>0,sq1,sr1).squeeze()
            cv2.imwrite(f"images/match_img0_{num}.jpg",out0)
            cv2.imwrite(f"images/match_img1_{num}.jpg",out1)
            num = num+ 1


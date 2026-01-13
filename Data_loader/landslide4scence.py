import numpy as np
from torch.utils import data
import h5py
import torch
from torch.utils.data.dataloader import DataLoader
from Data_loader.Data_augmentation import DataTransform

class Landslide4DataSet(data.dataloader.Dataset):
    def __init__(self, data_dir, set = None, h_flip=True, v_flip=True, scale_random_crop=True, noise=False, rotate = False, erase = False):
        self.set = set
        self.mean = [-0.4914, -0.3074, -0.1277, -0.0625, 0.0439, 0.0803, 0.0644, 0.0802, 0.3000, 0.4082, 0.0823, 0.0516,
                     0.3338, 0.7819]
        self.std = [0.9325, 0.8775, 0.8860, 0.8869, 0.8857, 0.8418, 0.8354, 0.8491, 0.9061, 1.6072, 0.8848, 0.9232,
                    0.9018, 1.2913]
        self.n_class = 2
        self.transform = DataTransform(h_flip=h_flip, v_flip=v_flip, scale=scale_random_crop, noise=noise,
                                       rotate=rotate, erase=erase)

        # 读取数据到矩阵
        self.img_ids = [i_id.strip() for i_id in open(data_dir + '/' + set + '.txt', 'r')]
        self.files = []

        for name in self.img_ids:
            img_file = data_dir + '/img/' + name
            label_file = data_dir + '/mask/' + name.replace('image', 'mask')
            self.files.append({
                'img': img_file,
                'label': label_file,
                'name': name
            })

    def __len__(self):
        # 返回数据集的长度
        return len(self.files)

    def __getitem__(self, index):
        # 根据索引返回一条数据
        datafiles = self.files[index]
        name = datafiles['name']
        # 读取.h5文件存至image和label
        with h5py.File(datafiles['img'], 'r') as hf:
            image = hf['img'][:]
        with h5py.File(datafiles['label'], 'r') as hf:
            label = hf['mask'][:]

        image = np.asarray(image, np.float32)
        label = torch.as_tensor(np.asarray(label, np.longlong), dtype=torch.uint8)
        image = torch.as_tensor(image).permute(2,0,1)
        # for i in range(len(self.mean)):
        #     image[i, :, :] -= self.mean[i]
        #     image[i, :, :] /= self.std[i]
        if self.set == 'train':
            label = label.unsqueeze(0)
            image, label = self.transform(image, label)
            label = label.squeeze(0)
        label = label.long()
        # onehot_label = F.one_hot(label, self.n_class).permute(2, 0, 1)
        return image, label, name





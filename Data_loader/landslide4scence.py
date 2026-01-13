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


if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt
    import cv2
    from scipy import ndimage
    import os

    folder_path = r"E:\landslide4scen\mask"  # 替换为你的文件夹路径

    # 获取所有文件名
    all_files = os.listdir(folder_path)

    # 筛选出所有 .png 文件
    mask_files = [file for file in all_files if file.endswith(".h5")]
    file_path = []

    # 遍历并处理每个 .h5 文件
    landslide_area = []
    for mask in mask_files:
        full_path = os.path.join(folder_path, mask)
        file_path.append(full_path)
    # 初始化长宽列表
    widths = []
    heights = []
    for i, file in enumerate(file_path):
        with h5py.File(file, 'r') as hf:
            label_image = hf['mask'][:]
            label_image = np.asarray(label_image)
        # 使用连通区域标记
        labeled_array, num_features = ndimage.label(label_image)
        # 遍历每个连通区域
        for i in range(1, num_features + 1):
            # 提取当前区域的坐标
            slip_y, slip_x = np.where(labeled_array == i)
            if len(slip_y) > 0 and len(slip_x) > 0:
                # 计算长宽
                min_x, max_x = np.min(slip_x), np.max(slip_x)
                min_y, max_y = np.min(slip_y), np.max(slip_y)
                slip_height = max_y - min_y + 1
                slip_width = max_x - min_x + 1
                widths.append(slip_width)
                heights.append(slip_height)
    # 绘制长宽的散点图
    fig, ax = plt.subplots(figsize=(6, 6))
    _, _, _, img = ax.hist2d(widths, heights, bins=32, cmap='Blues')
    # 添加坐标轴标签和标题
    ax.set_xlabel('width')
    ax.set_ylabel('height')
    ax.set_title('landslide4Scence')
    # 设置坐标轴范围
    ax.set_xlim(0, 128)
    ax.set_ylim(0, 128)

    # 添加颜色条
    cb = fig.colorbar(img, ax=ax)
    cb.set_label('Count')
    plt.tight_layout()
    plt.show()
#     train_dataset = Landslide4DataSet(data_dir=r'E:\landslide4scen', set = 'train')
#     train_loader = DataLoader(dataset=train_dataset, batch_size=2, pin_memory=True)
#     image, label, _ = train_dataset.__getitem__(0)
#     print(train_dataset.__len__())
#     from skimage import measure
#     import os
#
#     # 目标文件夹路径
#     folder_path = r"E:\landslide4scen\mask"  # 替换为你的文件夹路径
#
#     # 获取所有文件名
#     all_files = os.listdir(folder_path)
#
#     # 筛选出所有 .h5 文件
#     h5_files = [file for file in all_files if file.endswith(".h5")]
#     file_path =[]
#
#     # 遍历并处理每个 .h5 文件
#     landslide_area = []
#     for h5_file in h5_files:
#         full_path = os.path.join(folder_path, h5_file)
#         file_path.append(full_path)
#     for i,file in  enumerate(file_path):
#         with h5py.File(file, 'r') as hf:
#             label = hf['mask'][:]
#             np.asarray(label)
#             labels = measure.label(label, background=0)
#             regions = measure.regionprops(labels)
#             for i, region in enumerate(regions):
#                 landslide_area.append(region.area)
#     print(len(landslide_area))
#     import matplotlib.pyplot as plt
#     import seaborn as sns
#
#     # 计算累积分布函数（CDF）
#     # 计算累积分布函数（CDF）
#     sorted_data = np.sort(landslide_area)
#     cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
#
#     # 绘制CDF图
#     plt.figure(figsize=(10, 6))
#     plt.plot(sorted_data, cdf, label='CDF')
#
#     # 标注面积为500的点及其概率值
#     area = 500
#     index = np.searchsorted(sorted_data, area)
#     cdf_value = cdf[index]
#
#     plt.scatter(area, cdf_value, color='red')  # 标注点
#     plt.annotate(f'({area}, {cdf_value:.2f})',
#                  xy=(area, cdf_value),
#                  xytext=(area + 100, cdf_value + 0.05),  # 调整文本位置,
#                  fontsize=10,
#                  ha='center')
#
#     # 添加标题和标签
#     plt.title('Cumulative distribution of landslide area (CDF)')
#     plt.xlabel('Area')
#     plt.ylabel('Cumulative Probability')
#     plt.grid(True)
#     plt.legend()
#
#     # 显示图表
#     plt.show()



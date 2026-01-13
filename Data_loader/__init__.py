import importlib
import torch.utils.data as data_utils
from Data_loader.landslide4scence import Landslide4DataSet
from Data_loader.LuDing_dataset import Luding_Landslide_Dataset



class Struct(dict):
    def __getattr__(self, item):
        try:
            value = self[item]
            if type(value) == type({}):
                return Struct(value)
            return value
        except KeyError:
            raise AttributeError(item)

    def set_cd_cfg_from_file(cfg_path=r'configs/configs.py'):
        module_spec = importlib.util.spec_from_file_location('cfg_file', cfg_path)
        module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)
        cfg = module.cfg
        cfg = Struct(cfg)
        return cfg


cfg = Struct.set_cd_cfg_from_file()


class DataLoader:
    def __init__(self, dataset_name):
        self.dataset       = dataset_name
        self.landslide4scence_dataset = Landslide4DataSet
        self.luding_dataset = Luding_Landslide_Dataset
        self.Train_dataset = None
        self.Val_dataset   = None
        self.Test_dataset  = None
        if dataset_name == 'landslide4':
            landslide4_dir = r'landslide4scen'
            self.Train_dataset = self.landslide4scence_dataset(data_dir=landslide4_dir, set=cfg.dataset.set[0])
            self.Val_dataset   = self.landslide4scence_dataset(data_dir=landslide4_dir, set=cfg.dataset.set[1])
            self.Test_dataset  = self.landslide4scence_dataset(data_dir=landslide4_dir, set=cfg.dataset.set[2])
        elif dataset_name == 'Luding':
            landslide4_dir = 'Luding_Landslide_Dataset'
            self.Train_dataset = self.luding_dataset(dir=landslide4_dir, set=cfg.dataset.set[0])
            self.Val_dataset   = self.luding_dataset(dir=landslide4_dir, set=cfg.dataset.set[1])
            self.Test_dataset  = self.luding_dataset(dir=landslide4_dir, set=cfg.dataset.set[2])


    def get_dataloader(self, batch_size=cfg.dataset.batch_size):
        train_loader = data_utils.DataLoader(self.Train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        val_loader   = data_utils.DataLoader(self.Val_dataset, batch_size=1)
        test_loader  = data_utils.DataLoader(self.Test_dataset, batch_size=1)
        return train_loader, val_loader, test_loader


# # dataset_list = ['Luding','landslide4']
# Data_loader = DataLoader(dataset_name=cfg.dataset.dataset_name)
# Train_loader, Val_loader, Test_loader = Data_loader.get_dataloader()


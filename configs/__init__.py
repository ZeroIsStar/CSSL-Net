import base_model
import landslide_model
from torch import nn
import torch
from torch.optim.lr_scheduler import LRScheduler
from torch import optim
from loss import TL_loss, focal_hausdorffErloss, Lovasz_ce_loss, DynamicWeightedCrossEntropyLoss,FocalLoss, DynamicFocalLoss,AutoBalanceWeightedLoss, hybridloss, AdaptiveSegmentationLoss,mix_loss
from pytorch_toolbelt.losses import DiceLoss
from loss.lovasz import LovaszSoftmaxLoss
from Data_loader import cfg


class WarmupCosineAnnealingLR(LRScheduler):
    def __init__(self, optimizer, warmup_epochs, total_epochs, eta_min=0.0, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # 预热阶段：线性提升
            alpha = self.last_epoch / self.warmup_epochs
            return [base_lr * alpha for base_lr in self.base_lrs]
        else:
            # 余弦退火阶段（返回标量列表）
            progress = (self.last_epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            cosine_decay = 0.5 * (1 + torch.cos(torch.tensor(progress * torch.pi)))
            return [self.eta_min + (base_lr - self.eta_min) * cosine_decay.item() for base_lr in self.base_lrs]



class Loader(dict):
    def __init__(self, model_type = None):
        self.cfg = cfg
        self.model_type = model_type
        self.loss_function = self.get_loss_function()
        self.model = self.get_segment_model()
        self.optimizer = self.get_optimizer()
        self.scheduler = self.get_lr_scheduler()

    def get_segment_model(self):
        # 定义模型名称到模型类的映射
        model_mapping = {
            'Unet': lambda: base_model.UNet(n_channels=self.cfg.dataset.in_channels, n_classes=self.cfg.dataset.Class, bilinear=True),
            'Unet++': lambda:  base_model.UNet_Nested(in_channels=self.cfg.dataset.in_channels,n_classes=self.cfg.dataset.Class),
            'FCN8s': lambda: base_model.FCN8s(num_classes=self.cfg.dataset.Class),
            'FCN16s': lambda: base_model.FCN16s(num_classes=self.cfg.dataset.Class),
            'FCN32s': lambda: base_model.FCN32s(num_classes=self.cfg.dataset.Class),
            'Deeplabv3+': lambda: base_model.DeepLabV3plus(in_channels=self.cfg.dataset.in_channels, num_classes=self.cfg.dataset.Class),
            'TransUnet': lambda: base_model.transNet(n_classes=self.cfg.dataset.Class),
            'A2FPN': lambda: base_model.A2FPN(band=self.cfg.dataset.in_channels,class_num=self.cfg.dataset.Class),
            'ABCnet': lambda: base_model.A2FPN(band=self.cfg.dataset.in_channels,class_num=self.cfg.dataset.Class),
            'CMTFNet': lambda: base_model.CMTFNet(num_classes=self.cfg.dataset.Class),
            'FTUNetFormer': lambda: base_model.CMTFNet(num_classes=self.cfg.dataset.Class),
            'MANet': lambda:  base_model.MANet(num_classes=self.cfg.dataset.Class),
            'DCswin': lambda:  base_model.dcswin_base(False,num_classes=self.cfg.dataset.Class),
            'resnet18': lambda:  base_model.new_model(CNN_backbone=base_model.CNN_backbone(),nums=self.cfg.dataset.Class),
            'resnet34': lambda:  base_model.new_model(CNN_backbone=base_model.CNN_backbone(),nums=self.cfg.dataset.Class),
            'swin_t': lambda:  base_model.new_model(swin_backbone=base_model.swin_t(),nums=self.cfg.dataset.Class),
            'swin_s': lambda:  base_model.new_model(swin_backbone=base_model.swin_s(), nums=self.cfg.dataset.Class),
            'double_backbone0': lambda:  base_model.new_model(backbone=base_model.CNN_backbone(), swin_backbone=base_model.swin_t(), nums=self.cfg.dataset.Class),
            'double_backbone1': lambda:  base_model.new_model(backbone=base_model.CNN_backbone(), swin_backbone=base_model.swin_s(), nums=self.cfg.dataset.Class),
            'double_backbone2': lambda:  base_model.new_model(backbone=base_model.CNN_backbone(backbone='resnet34'), swin_backbone=base_model.swin_t(), nums=self.cfg.dataset.Class),
            'double_backbone3': lambda:   base_model.new_model(backbone=base_model.CNN_backbone(backbone='resnet34'), swin_backbone=base_model.swin_s(), nums=self.cfg.dataset.Class),
            'SCDUNetPP': lambda:  base_model.SCDUNetPP(self.cfg.dataset.in_channels, num_class=self.cfg.dataset.Class),
            'swin_cnn': lambda:  base_model.swin_cnn(swin_backbone='swin_t',  nums=self.cfg.dataset.Class),
            'SegFormer': lambda:  base_model.SegFormer(num_classes=self.cfg.dataset.Class, phi='b5'),
            'R50_swin': lambda:  base_model.res_swin(swin_backbone='swin_l', nums=self.cfg.dataset.Class),
            '18_t': lambda:  base_model.shuffle_model(cnn_backbone= 'resnet18', swin_backbone='swin_t'),
            '18_l': lambda:  base_model.shuffle_model(cnn_backbone= 'resnet18', swin_backbone='swin_l'),
            '18_s': lambda:  base_model.shuffle_model(cnn_backbone= 'resnet18', swin_backbone='swin_s'),
            '18_b': lambda:  base_model.shuffle_model(cnn_backbone= 'resnet18', swin_backbone='swin_b'),
            '34_t': lambda:  base_model.shuffle_model(cnn_backbone= 'resnet34', swin_backbone='swin_t'),
            '34_l': lambda:  base_model.shuffle_model(cnn_backbone= 'resnet34', swin_backbone='swin_l'),
            '34_b': lambda:  base_model.shuffle_model(cnn_backbone='resnet34', swin_backbone='swin_b'),
            '34_s': lambda:  base_model.shuffle_model(cnn_backbone='resnet34', swin_backbone='swin_s'),
            'new_34_t': lambda:  base_model.New_model(cnn_backbone= 'resnet34', swin_backbone='swin_t', nums=self.cfg.dataset.Class),
            'new_34_s': lambda:  base_model.New_model(cnn_backbone= 'resnet34', swin_backbone='swin_s', nums=self.cfg.dataset.Class),
            'new_34_b': lambda:  base_model.New_model(cnn_backbone= 'resnet34', swin_backbone='swin_b', nums=self.cfg.dataset.Class),
            'new_34_l': lambda:  base_model.New_model(cnn_backbone= 'resnet34', swin_backbone='swin_l', nums=self.cfg.dataset.Class),
            'FLAnet': lambda:  base_model.compile_model(),
            'pyramidMamba': lambda:  base_model.PyramidMamba(),
            'EfficientPyramidMamba': lambda:  base_model.EfficientPyramidMamba(),
            'FreDsnet': lambda:  base_model.FDS(self.cfg.dataset.Class),
            'Attention_deeplabV3plus': lambda:  base_model.AttentionDeeplabV3plus(num_classes=self.cfg.dataset.Class),
            'Unet_former': lambda:  base_model.ft_unetformer(),
            'test_model': lambda:  base_model.test_model(cnn_backbone='resnet18', swin_backbone='swin_t',
                                                nums=self.cfg.dataset.Class),
            'CBSCnet': lambda:  base_model.CBSCnet(self.cfg.dataset.in_channels, n_classes=self.cfg.dataset.Class),
            'UFPN': lambda:  base_model.model(self.cfg.dataset.in_channels, num_classes=self.cfg.dataset.Class),
            'without': lambda:  base_model.without(self.cfg.dataset.in_channels, num_classes=self.cfg.dataset.Class),
            'Unet_MA': lambda:  base_model.UNet_MA(self.cfg.dataset.in_channels, num_classes=self.cfg.dataset.Class),
            'MV3FPN': lambda:  base_model.MobileNet_FPN(num_classes=self.cfg.dataset.Class),
            'shuffle_model': lambda:  test_model.shuffle_fpn(cnn_backbone='resnet18', swin_backbone='swin_l', in_channels=3,
                                                    nums=self.cfg.dataset.Class).cuda(),
            'resnet18_fpn': lambda:  test_model.resnet18_fpn(in_channels=self.cfg.dataset.in_channels, num_classes=self.cfg.dataset.Class),
            'resnet34_fpn': lambda:  test_model.resnet34_fpn(in_channels=self.cfg.dataset.in_channels, num_classes=self.cfg.dataset.Class),
            'trans18_fpn': lambda:  test_model.trans18_fpn(in_channels=self.cfg.dataset.in_channels, num_classes=self.cfg.dataset.Class),
            'trans34_fpn': lambda:  test_model.trans34_fpn(in_channels=self.cfg.dataset.in_channels, num_classes=self.cfg.dataset.Class),
            'sm_fpn': lambda:  test_model.sm_fpn(in_channels=self.cfg.dataset.in_channels, num_classes=self.cfg.dataset.Class),
            'deeplabv3_resnet50': lambda:  base_model.DeeplabV3_resnet50(in_channels=self.cfg.dataset.in_channels,
                                                                num_classes=self.cfg.dataset.Class),
            'Ukan': lambda:  base_model.UKAN(input_channels=self.cfg.dataset.in_channels, num_classes=self.cfg.dataset.Class),
            'MUkan': lambda:  base_model.MUKAN(input_channels=self.cfg.dataset.in_channels, num_classes=self.cfg.dataset.Class),
            'BisDnet': lambda:  base_model.BiSeNet_DeeNet(in_channels=self.cfg.dataset.in_channels, n_class=self.cfg.dataset.Class),
            'MS2LandsNet': lambda:  base_model.MS2LandsNet(input_channel=self.cfg.dataset.in_channels, n_classes=self.cfg.dataset.Class),
            'Unet3P':lambda: landslide_model.UNet3Plus_DeepSup(n_classes=self.cfg.dataset.Class,n_channels=self.cfg.dataset.in_channels),
            'swinmamba': lambda: landslide_model.SwinUMamba(in_chans=self.cfg.dataset.in_channels,out_chans=self.cfg.dataset.Class),
            'SCD': lambda : landslide_model.SCDUNetPP(self.cfg.dataset.in_channels, num_class=self.cfg.dataset.Class),
            'landslide_mamaba' : lambda :landslide_model.landslidemamba(self.cfg.dataset.in_channels),
            'dwUnet++' : lambda:landslide_model.UnetPP_dw.UNet_Nesteddw(self.cfg.dataset.in_channels),
            'RDSSA_net':lambda: landslide_model.RDSSA_net(self.cfg.dataset.in_channels),
            'A_net': lambda: landslide_model.A_net(self.cfg.dataset.in_channels),
            'Bisenet': lambda: base_model.BiSeNet(num_classes=self.cfg.dataset.Class),
            'BisenetV2': lambda: base_model.BiSeNetV2(num_classes=self.cfg.dataset.Class),
            'test': lambda: landslide_model.model(in_channels=self.cfg.dataset.in_channels,num_classes=self.cfg.dataset.Class),
            'RIPF_Unet': lambda: landslide_model.RiPF_UNet(n_channels=self.cfg.dataset.in_channels,n_classes=self.cfg.dataset.Class, bilinear=True),
            'MFFEnet': lambda: landslide_model.DeepLab(in_channels =self.cfg.dataset.in_channels,num_classes=self.cfg.dataset.Class),
            'TransUnet2': lambda: landslide_model.TransUNet2(num_classes=self.cfg.dataset.Class),
            'MiM_iSTD': lambda: base_model.MiM([2] * 3, [16, 32, 64, 128, 256], self.cfg.dataset.in_channels),
            'RFA_ResUnet': lambda: landslide_model.BFA_resunet(num_classes=self.cfg.dataset.Class),
            'bisednet': lambda: landslide_model.BiSeNet_DenseNet6(in_channels = self.cfg.dataset.in_channels,n_classes=self.cfg.dataset.Class),
            'Base_line': lambda: landslide_model.base_line(num_classes=self.cfg.dataset.Class),
            'ssce': lambda: landslide_model.base_line_SSCE(num_classes=self.cfg.dataset.Class),
            'LPA': lambda: landslide_model.base_line_LPA(),
            'FCSS': lambda: landslide_model.base_line_FCSS(),
            'ssce_fcss': lambda: landslide_model.base_line_SSCE_FCSS(),
            'fcss_lpa': lambda: landslide_model.base_line_FCSS_LPA(),
            'ssce_lpa': lambda: landslide_model.base_line_SSCE_LPA(),
            
            


            # 可以继续添加其他模型
        }
        # 检查模型名称是否存在于映射中
        if self.model_type not in model_mapping:
            available_models = ', '.join(model_mapping.keys())
            raise ValueError(f"Model {self.model_type} not found in model mapping. Available models are: {available_models}")
        return model_mapping[self.model_type]()

    def get_loss_function(self):
        # 初始化为None或默认损失函数
        loss_mapping = {
            'celoss': lambda : nn.CrossEntropyLoss(weight=self.cfg.train.loss_function_weight),
            'Tversky_loss_lovasz': lambda : TL_loss(alpha=0.5, beta=0.5, n_class = self.cfg.dataset.Class),
            'f-h-loss': lambda : focal_hausdorffErloss(),
            'lovasz_ce_loss': lambda : Lovasz_ce_loss(weight=self.cfg.train.loss_function_weight, n_class=self.cfg.dataset.Class),
            'lovasz_softmax': lambda : LovaszSoftmaxLoss,
            'DynamicWeightedCrossEntropyLoss': lambda : DynamicWeightedCrossEntropyLoss(),
            'focalloss': lambda: FocalLoss(),
            'dynamic_focal_loss': lambda:  DynamicFocalLoss(),
            'dice': lambda: DiceLoss(mode='multiclass'),
            'ASL': lambda: AutoBalanceWeightedLoss(classes=2),
            'hybridloss': lambda : hybridloss,
            'ASL_A':lambda : AdaptiveSegmentationLoss(num_classes=self.cfg.dataset.Class),
            'mix_loss':lambda : mix_loss(),
        }
        # 检查损失函数名称是否存在于映射中
        if self.cfg.train.loss_function not in loss_mapping:
            loss = ', '.join(loss_mapping.keys())
            raise ValueError(f"Model {self.cfg.train.loss_function} not found in model mapping. Available models are: {loss}")
        return loss_mapping[self.cfg.train.loss_function]()

    def get_optimizer(self):
        # 优化器
        # 定义优化器映射
        optimizer_mapping = {
            'SGD': lambda: optim.SGD(
                self.model.parameters(),
                lr=self.cfg.optimizer.base_lr,
                momentum=self.cfg.optimizer.momentum,
                weight_decay=self.cfg.optimizer.weight_decay
            ),
            'AdamW': lambda: optim.AdamW(
                self.model.parameters(),
                lr=self.cfg.optimizer.base_lr,
                weight_decay=self.cfg.optimizer.weight_decay
            ),
            'Adam': lambda: optim.Adam(
                self.model.parameters(),
                lr=self.cfg.optimizer.base_lr,
                weight_decay=self.cfg.optimizer.weight_decay
            )
        }

        # 检查优化器名称是否存在于映射中
        if self.cfg.optimizer.type not in optimizer_mapping:
            available_optimizers = ', '.join(optimizer_mapping.keys())
            raise ValueError(
                f"Optimizer {self.cfg.optimizer.type} not found in optimizer mapping. Available optimizers are: {available_optimizers}")

        # 返回对应的优化器实例
        return optimizer_mapping[self.cfg.optimizer.type]()

    def get_lr_scheduler(self):
        scheduler_mapping = {
            'Poly': lambda: optim.lr_scheduler.LambdaLR(
                self.optimizer,
                lr_lambda=lambda epoch: max(0.0, float(self.cfg.scheduler.epoch - epoch) / float(max(1,
                                                                                                     self.cfg.scheduler.epoch - self.cfg.scheduler.warmup_epoch))) if epoch >= self.cfg.scheduler.warmup_epoch else float(
                    epoch) / float(max(1, self.cfg.scheduler.warmup_epoch)),

            ),
            'step': lambda: optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.cfg.scheduler.step_size,
                gamma=self.cfg.scheduler.gamma
            ),
            'CosineAnnealingLR': lambda: optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.cfg.scheduler.epoch,
            ),
            'CosineAnnealingWarmRestarts': lambda: optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=8,
                T_mult=2,
                eta_min=1e-6,
            ),
            'normal': lambda: optim.lr_scheduler.LambdaLR(
                self.optimizer,
                lr_lambda=lambda epoch: 1
            ),
            'WarmupCosineAnnealingLR': lambda: WarmupCosineAnnealingLR(
                self.optimizer,
                warmup_epochs=self.cfg.scheduler.warmup_epoch,
                total_epochs=self.cfg.scheduler.epoch,
                eta_min=0,
            )
        }

        # 检查学习率调度器名称是否存在于映射中
        if self.cfg.scheduler.type not in scheduler_mapping:
            available_schedulers = ', '.join(scheduler_mapping.keys())
            raise ValueError(
                f"Scheduler {self.cfg.scheduler.type} not found in scheduler mapping. Available schedulers are: {available_schedulers}")

        return scheduler_mapping[self.cfg.scheduler.type]()











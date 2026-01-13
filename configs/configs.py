import torch

cfg = dict(
    segment_model = dict(
        type ='FCN8s',  # ['FCN8s','FCN16s','FCN32s','Unet','Deeplabv3+','SCDUNetPP']
    ),

    dataset = dict(
        set          = ['train', 'val', 'test'],
        dataset_name = 'Luding',
        batch_size   = 16,
        in_channels= 14,
        Class=2,
        # clip_grad_value_ = 5.0  # 模型梯度裁剪
    ),

    optimizer = dict(
        type         = 'AdamW',  # ['SGD','AdamW']
        base_lr      = 0.0005,
        min_lr       = 0,
        step_size    = 10,
        gamma        = 0.9,
        weight_decay = 5e-4,
        momentum     = 0.99
    ),
    train = dict(
        loss_function='mix_loss',  # ['celoss','Tversky_loss_lovasz','f-h-loss', 'lovasz_ce_loss','lovasz_softmax','DynamicWeightedCrossEntropyLoss', 'dynamic_focal_loss']
        loss_function_weight = torch.tensor([1.0, 1.0]),
    ),

    scheduler=dict(
        #['linear', 'step', 'CosineAnnealingLR'] 内置
        epoch        = 100,
        type         = 'WarmupCosineAnnealingLR',
        warmup_epoch = 10
    )

)

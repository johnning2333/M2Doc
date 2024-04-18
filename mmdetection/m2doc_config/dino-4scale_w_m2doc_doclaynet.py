model = dict(
    type='DINO_w_M2Doc',
    num_queries=900,  # num_matching_queries
    with_box_refine=True,
    as_two_stage=True,
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=1),
    Grid_layers=dict(
        auto_model_path='bert-base-multilingual-cased',
        embedding_dim=64,
        freeze_params=False,
        batch_max_num=128,
        select_layers=12,
        max_token_nums=48,
        with_line_proj=True,
    ),
    backbone=dict(
        type='ResNet_w_Early',
        depth=50,
        num_stages=4,
        out_indices=(1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50'),
        #grid representation channel dimmensions
		grid_dims=64),
    neck=dict(
        type='ChannelMapper',
        in_channels=[512, 1024, 2048],
        kernel_size=1,
        out_channels=256,
        act_cfg=None,
        norm_cfg=dict(type='GN', num_groups=32),
        num_outs=4),
    encoder=dict(
        num_layers=6,
        layer_cfg=dict(
            self_attn_cfg=dict(embed_dims=256, num_levels=4,
                               dropout=0.0),  # 0.1 for DeformDETR
            ffn_cfg=dict(
                embed_dims=256,
                feedforward_channels=2048,  # 1024 for DeformDETR
                ffn_drop=0.0))),  # 0.1 for DeformDETR
    decoder=dict(
        num_layers=6,
        return_intermediate=True,
        layer_cfg=dict(
            self_attn_cfg=dict(embed_dims=256, num_heads=8,
                               dropout=0.0),  # 0.1 for DeformDETR
            cross_attn_cfg=dict(embed_dims=256, num_levels=4,
                                dropout=0.0),  # 0.1 for DeformDETR
            ffn_cfg=dict(
                embed_dims=256,
                feedforward_channels=2048,  # 1024 for DeformDETR
                ffn_drop=0.0)),  # 0.1 for DeformDETR
        post_norm_cfg=None,
        
        # late fusion config
        with_late_fusion=True,
        grid_dims = 64,
    ),
    positional_encoding=dict(
        num_feats=128,
        normalize=True,
        offset=0.0,  # -0.5 for DeformDETR
        temperature=20),  # 10000 for DeformDETR
    bbox_head=dict(
        type='DINOHead',
        num_classes=11,
        sync_cls_avg_factor=True,
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),  # 2.0 in DeformDETR
        loss_bbox=dict(type='L1Loss', loss_weight=5.0),
        loss_iou=dict(type='GIoULoss', loss_weight=2.0)),
    dn_cfg=dict(  # TODO: Move to model.train_cfg ?
        label_noise_scale=0.5,
        box_noise_scale=1.0,  # 0.4 for DN-DETR
        group_cfg=dict(dynamic=True, num_groups=None,
                       num_dn_queries=100)),  # TODO: half num_dn_queries
    # training and testing settings
    train_cfg=dict(
        assigner=dict(
            type='HungarianAssigner',
            match_costs=[
                dict(type='FocalLossCost', weight=2.0),
                dict(type='BBoxL1Cost', weight=5.0, box_format='xywh'),
                dict(type='IoUCost', iou_mode='giou', weight=2.0)
            ])),
    test_cfg=dict(max_per_img=300))  # 100 for DeformDETR

find_unused_parameters=True
# train_pipeline, NOTE the img_scale and the Pad's size_divisor is different
# from the default setting in mmdet.
backend_args = None
classes=('Caption', 'Footnote', 'Formula', 'List-item', 'Page-footer', 'Page-header', 'Picture', 'Section-header', 'Table', 'Text', 'Title')

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='LoadAnnotationsMM', with_text=True),
    dict(type='RandomFlipMM', prob=0.5),
    dict(
        type='RandomChoice',
        transforms=[
            [
                dict(
                    type='RandomChoiceResize',
                    resize_type='ResizeMM',
                    scales=[
                            # (480, 1025), (512, 1025), (544, 1025), (576, 1025),
                            (608, 1025), (640, 1025), (672, 1025), (704, 1025),
                            (736, 1025), (768, 1025), (800, 1025), (832, 1025),
                            (864, 1025), (896, 1025), (928, 1025), (960, 1025), 
                            (992, 1025), (1025, 1025)
                            ],
                    keep_ratio=True)
            ],
            [
                dict(
                    type='RandomChoiceResize',
                    resize_type='ResizeMM',
                    # The radio of all image in train dataset < 7
                    # follow the original implement
                    scales=[(400, 4200), (500, 4200), (600, 4200)],
                    keep_ratio=True),
                dict(
                    type='RandomCropMM',
                    crop_type='absolute_range',
                    crop_size=(384, 600),
                    allow_negative_crop=True),
                dict(
                    type='RandomChoiceResize',
                    resize_type='ResizeMM',
                    scales=[
                            # (480, 1025), (512, 1025), (544, 1025), (576, 1025),
                            (608, 1025), (640, 1025), (672, 1025), (704, 1025),
                            (736, 1025), (768, 1025), (800, 1025), (832, 1025),
                            (864, 1025), (896, 1025), (928, 1025), (960, 1025), 
                            (992, 1025), (1025, 1025)
                            ],
                    keep_ratio=True)
            ]
        ]),
    dict(type='PackDetInputsMM')
]

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='LoadAnnotationsMM', with_text=True),
    dict(type='ResizeMM', scale=(1025, 1025), keep_ratio=True),
    dict(
        type='PackDetInputsMM',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]


train_dataloader = dict(
    batch_size=1,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type='DoclayNetDataset',
        metainfo=dict(classes=classes),
        ann_file=
        '/apsarapangu/disk3/chonghuan.zn/data/doc/doclaynet/COCO/train.json',
        img_prefix = '/apsarapangu/disk3/chonghuan.zn/data/doc/doclaynet/core/vsr/',
        ann_prefix = '/apsarapangu/disk3/chonghuan.zn/data/doc/doclaynet/core/vsr/Annos/',
        data_prefix = dict(img='/apsarapangu/disk3/chonghuan.zn/data/doc/doclaynet/core/PNG'),
        filter_cfg=dict(filter_empty_gt=False),
        pipeline=train_pipeline,
        ))
val_dataloader = dict(
    batch_size=1,
    num_workers=8,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='DoclayNetDataset',
        metainfo=dict(classes=classes),
        ann_file=
        '/apsarapangu/disk3/chonghuan.zn/data/doc/doclaynet/COCO/test.json',
        img_prefix = '/apsarapangu/disk3/chonghuan.zn/data/doc/doclaynet/core/vsr/',
        ann_prefix = '/apsarapangu/disk3/chonghuan.zn/data/doc/doclaynet/core/vsr/Annos/',
        data_prefix = dict(img='/apsarapangu/disk3/chonghuan.zn/data/doc/doclaynet/core/PNG'),
        filter_cfg=dict(filter_empty_gt=False),
        pipeline=test_pipeline,
        ))
test_dataloader = dict(
    batch_size=1,
    num_workers=8,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='DoclayNetDataset',
        metainfo=dict(classes=classes),
        ann_file=
        '/apsarapangu/disk3/chonghuan.zn/data/doc/doclaynet/core/COCO/test.json',
        img_prefix = '/apsarapangu/disk3/chonghuan.zn/data/doc/doclaynet/core/vsr/',
        ann_prefix = '/apsarapangu/disk3/chonghuan.zn/data/doc/doclaynet/core/vsr/Annos/',
        data_prefix = dict(img='/apsarapangu/disk3/chonghuan.zn/data/doc/doclaynet/core/PNG'),
        filter_cfg=dict(filter_empty_gt=False),
        pipeline=test_pipeline,
        ))

val_evaluator = dict(
    type='CocoMetric',
    ann_file=
    '/apsarapangu/disk3/chonghuan.zn/data/doc/doclaynet/core/COCO/test.json',
    metric=['bbox'],
    format_only=False,
    classwise=True,
    backend_args=None)
test_evaluator = dict(
    type='CocoMetric',
    ann_file=
    '/apsarapangu/disk3/chonghuan.zn/data/doc/doclaynet/core/COCO/test.json',
    metric=['bbox'],
    format_only=False,
    classwise=True,
    backend_args=None)
# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=0.0001,  # 0.0002 for DeformDETR
        weight_decay=0.0001),
    clip_grad=dict(max_norm=0.1, norm_type=2),
    paramwise_cfg=dict(custom_keys={'backbone': dict(lr_mult=0.1)})
)  # custom_keys contains sampling_offsets and reference_points in DeformDETR  # noqa

# learning policy
max_epochs = 12
train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=1)

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[11,],
        gamma=0.1)
]

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
# base_batch_size = (8 GPUs) x (2 samples per GPU)
auto_scale_lr = dict(base_batch_size=16)
default_scope = 'mmdet'
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=2),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='DetVisualizationHook',
        # draw=True,
        # show=True,
        # wait_time=2,
        # test_out_dir='test_')
    ))
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='DetLocalVisualizer',
    vis_backends=[dict(type='LocalVisBackend')],
    name='visualizer')
log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)
log_level = 'INFO'
load_from = 'https://download.openmmlab.com/mmdetection/v3.0/dino/dino-4scale_r50_8xb2-12e_coco/dino-4scale_r50_8xb2-12e_coco_20221202_182705-55b2bba2.pth'

work_dir = 'work_dirs/dino-4scale_w_m2doc_r50_8xb2-12e_doclaynet'
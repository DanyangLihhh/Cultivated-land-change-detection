# dataset settings
dataset_type = 'JILIN_ONE'
data_root = '/root/siton-gpfs-archive/lidanyang/open-cd-main/data'

crop_size = (256, 256)
train_pipeline = [
    dict(type='MultiImgLoadImageFromFile'),
    dict(type='MultiImgLoadAnnotations'),
    dict(type='MultiImgRandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='MultiImgRandomFlip', prob=0.5),
    dict(type='MultiImgPackSegInputs')
]
test_pipeline = [
    dict(type='MultiImgLoadImageFromFile'),
    dict(type='MultiImgResize', scale=(256, 256), keep_ratio=True),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type='MultiImgLoadAnnotations'),
    dict(type='MultiImgPackSegInputs')
]
img_ratios = [0.75, 1.0, 1.25]
tta_pipeline = [
    dict(type='MultiImgLoadImageFromFile', backend_args=None),
    dict(
        type='TestTimeAug',
        transforms=[
            [
                dict(type='MultiImgResize', scale_factor=r, keep_ratio=True)
                for r in img_ratios
            ],
            [
                dict(type='MultiImgRandomFlip', prob=0., direction='horizontal'),
                dict(type='MultiImgRandomFlip', prob=1., direction='horizontal')
            ],
            [dict(type='MultiImgLoadAnnotations')],
            [dict(type='MultiImgPackSegInputs')]
        ])
]
train_dataloader = dict(
    batch_size=8,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path_from='train/pre',
            img_path_to='train/post',
            seg_map_path='train/label'),
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path_from='val/pre',
            img_path_to='val/post',
            seg_map_path='val/label'),
        pipeline=test_pipeline))
test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path_from='test/pre',
            img_path_to='test/post',
            seg_map_path='test/label_false'),
        pipeline=test_pipeline))
# test_cfg = None
val_evaluator = dict(type='mmseg.IoUMetric', iou_metrics=['mFscore', 'mIoU'])
test_evaluator = val_evaluator
_base_ = 'grounding_dino_swin-t_finetune_16xb2_1x_coco.py'
# _base_ = 'grounding_dino_r50_scratch_8xb2_1x_coco.py'
data_root = 'data/urdformer/'
# class_name = ('drawer', 'left_door', 'right_door', 'handle', 'knob', 'washer_door', 'doorD', 'oven_door', 'doorU')
class_name = ('objects')

num_classes = len(class_name)
print(num_classes)
metainfo = dict(classes=class_name, palette='random')

model = dict(bbox_head=dict(num_classes=num_classes))

train_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='/home/zoeyc/github/mmdetection/data/cabinets/annotations/test.json',
        data_prefix=dict(img='images/')))

val_dataloader = dict(
    dataset=dict(
        metainfo=metainfo,
        data_root=data_root,
        ann_file='/home/zoeyc/github/mmdetection/data/cabinets/annotations/test.json',
        data_prefix=dict(img='images/')))

test_dataloader = val_dataloader

val_evaluator = dict(ann_file='/home/zoeyc/github/mmdetection/data/cabinets/annotations/test.json')
test_evaluator = val_evaluator

max_epoch = 50

default_hooks = dict(
    checkpoint=dict(interval=1, max_keep_ckpts=1, save_best='auto'),
    logger=dict(type='LoggerHook', interval=5))
train_cfg = dict(max_epochs=max_epoch, val_interval=1)

param_scheduler = [
    dict(type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=50),
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epoch,
        by_epoch=True,
        milestones=[40],
        gamma=0.1)
]

optim_wrapper = dict(
    optimizer=dict(lr=0.00005),
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'backbone': dict(lr_mult=0.1),
            'language_model': dict(lr_mult=0),
        }))

auto_scale_lr = dict(base_batch_size=16)

from mmengine.hooks import (CheckpointHook, DistSamplerSeedHook, IterTimerHook,
                            LoggerHook, ParamSchedulerHook)
from mmengine.optim import AmpOptimWrapper, CosineAnnealingLR, LinearLR
from torch.optim import AdamW
from transformers import AutoTokenizer

from xtuner.dataset import ConcatDataset
from xtuner.dataset.samplers import LengthGroupedSampler
from xtuner.engine.hooks import DatasetInfoHook
from xtuner.engine.runner import TrainLoop
from xtuner.utils import PROMPT_TEMPLATE
from xtuner.dataset.map_fns import template_map_fn_factory

from third_parts.mmdet.models.losses import DiceLoss, CrossEntropyLoss, CIoULoss
from peft import LoraConfig

# import sys
# sys.path.append('/home/rqshen/spatial_mllm')

from projects.llava_sam2.models.internvl import InternVL_Slowfast

from projects.llava_sam2.models import VideoLLaVASAMModel, SAM2TrainRunner, VideoLLaVASAMModel_zero3
# from projects.llava_sam2.datasets import VideoReVOSDataset, VideoMeVISDataset, VideoRelationDataset, VideoRefYoutubeVOSDataset, video_lisa_collate_fn, VideoSAM2Dataset
from projects.llava_sam2.datasets import VideoReVOSDataset_box, VideoMeVISDataset_box, VideoRefYoutubeVOSDataset_box, video_lisa_collate_fn
from projects.llava_sam2.datasets import VideoChatUniViDataset
from projects.llava_sam2.datasets import RefCOCOgGCGDataset, OpenPsgGCGDataset, FlickrGCGDataset, GranDfGCGDataset, OspreyDataset, OspreyDescriptionDataset, OspreyShortDescriptionDataset
from projects.llava_sam2.datasets import LLaVADataset
from projects.llava_sam2.datasets import ReferSegmDataset, ReferSegmDataset_box
from projects.llava_sam2.models.preprocess.image_resize import DirectResize
from projects.llava_sam2.datasets import VideoSpatialRelativeDataset_box
from projects.llava_sam2.datasets import VideoSpatialAbsoluteDataset_box
from projects.llava_sam2.datasets import VideoSpatialSwitchDataset_box

# export TORCH_DISTRIBUTED_DEBUG=DETAIL

#######################################################################
#                          PART 1  Settings                           #
#######################################################################
# Model
path='OpenGVLab/InternVL2_5-8B' # InternVL  

# Data
template = "internlm2_chat"
prompt_template = PROMPT_TEMPLATE.internlm2_chat
max_length = 8192

# Scheduler & Optimizer
batch_size = 2  # per_device
accumulative_counts = 4
dataloader_num_workers = 4
max_epochs = 5
optim_type = AdamW
# official 1024 -> 4e-5
# lr = 1e-6
lr = 4e-5
betas = (0.9, 0.999)
weight_decay = 0.05
max_norm = 1  # grad clip
warmup_ratio = 0.05

# Save
save_steps = 1000
save_total_limit = 2  # Maximum checkpoints to keep (-1 means unlimited)

special_tokens = ['<box_sep>', '<no_box>', '<next>', '<single_img>', '<p>', '</p>', '<vp>', '</vp>']

tokenizer = dict(
    type=AutoTokenizer.from_pretrained,
    pretrained_model_name_or_path=path,
    trust_remote_code=True,
    padding_side='right')

extra_image_processor = dict(
    type=DirectResize,
    target_length=1024,
)

#######################################################################
#            PART 2  Model & Tokenizer & Image Processor              #
#######################################################################
model = dict(
    type=VideoLLaVASAMModel_zero3,
    special_tokens=special_tokens,
    # frozen_sam2_decoder=False,
    mllm=dict(
        type=InternVL_Slowfast,
        model_path=path,
        freeze_llm=True,
        freeze_visual_encoder=True,
        llm_lora=dict(
            type=LoraConfig,
            r=128,
            lora_alpha=256,
            lora_dropout=0.05,
            bias='none',
            task_type='CAUSAL_LM'),
        special_tokens=special_tokens,
    ),
    tokenizer=tokenizer, 
    bs=batch_size,
)

#######################################################################
#                      PART 3  Dataset & Dataloader                   #
#######################################################################

############### video res

# ReVOS需要手动下载, 因为下载数据包不全
data_root_revos = '/home/volume_shared/share_datasets/ReVOS/'
video_revos_image_folder = data_root_revos + 'JPEGImages'
video_revos_expression_file = data_root_revos + 'meta_expressions_train_.json'
video_revos_box_file = 'mask_to_box/revos_bbox.json' # home目录下

data_root_mevis = '/home/volume_shared/share_datasets/MeViS/MeViS_release/train/'
video_mevis_image_folder = data_root_mevis + 'JPEGImages'
video_mevis_expression_file = data_root_mevis + 'meta_expressions.json'
video_mevis_box_file = 'mask_to_box/mevis_bbox.json' 

data_root_refytvos = '/home/volume_shared/share_datasets/Ref-Youtube-VOS/'
video_refytvos_image_folder = data_root_refytvos + 'train/JPEGImages/'
video_refytvos_expression_file = 'ref_ytvos_expressions.json' # 注意, ref_ytvos的expression需要用改动后的, 原始的没有anno_id
video_refytvos_box_file = 'mask_to_box/ref_ytvos_bbox.json'

# relative relation
video_relative_image_folder = '/tmp/all_vos2/'
video_relative_expression_file = 'all_vos2_relative_step2.json'
video_relative_box_file = 'mask_to_box/all_vos2_relative_box.json'

# absolute relation
video_absolute_image_folder = '/tmp/all_vos2/'
video_absolute_expression_file = 'all_vos2_absolute_step2.json'
video_absolute_box_file = 'mask_to_box/all_vos2_absolute_box.json'

video_revos_dataset = dict(
    type=VideoReVOSDataset_box,
    image_folder=video_revos_image_folder,
    expression_file=video_revos_expression_file,
    box_file=video_revos_box_file,
    tokenizer=tokenizer,
    template_map_fn=dict(
        type=template_map_fn_factory, template=prompt_template),
    max_length=max_length,
    lazy=True,
    repeats=10,
    special_tokens=special_tokens,
    extra_image_processor=extra_image_processor,
    sampled_frames=5,
)

video_mevis_dataset = dict(
    type=VideoMeVISDataset_box,
    image_folder=video_mevis_image_folder,
    expression_file=video_mevis_expression_file,
    box_file=video_mevis_box_file,
    tokenizer=tokenizer,
    template_map_fn=dict(
        type=template_map_fn_factory, template=prompt_template),
    max_length=max_length,
    lazy=True,
    repeats=4,
    special_tokens=special_tokens,
    extra_image_processor=extra_image_processor,
    sampled_frames=5,
)

video_refytvos_dataset = dict(
    type=VideoRefYoutubeVOSDataset_box,
    image_folder=video_refytvos_image_folder,
    expression_file=video_refytvos_expression_file,
    box_file=video_refytvos_box_file,
    tokenizer=tokenizer,
    template_map_fn=dict(
        type=template_map_fn_factory, template=prompt_template),
    max_length=max_length,
    lazy=True,
    repeats=4,
    special_tokens=special_tokens,
    extra_image_processor=extra_image_processor,
    sampled_frames=5,
)

video_spatial_relative_dataset = dict(
    type=VideoSpatialRelativeDataset_box,
    image_folder=video_relative_image_folder,
    expression_file=video_relative_expression_file,
    box_file=video_relative_box_file,
    tokenizer=tokenizer,
    template_map_fn=dict(
        type=template_map_fn_factory, template=prompt_template),
    max_length=max_length,
    lazy=True,
    repeats=4,
    special_tokens=special_tokens,
    extra_image_processor=extra_image_processor, # Resize (1024,1024)
    sampled_frames=5,
)

video_spatial_absolute_dataset = dict(
    type=VideoSpatialAbsoluteDataset_box,
    image_folder=video_absolute_image_folder,
    expression_file=video_absolute_expression_file,
    box_file=video_absolute_box_file,
    tokenizer=tokenizer,
    template_map_fn=dict(
        type=template_map_fn_factory, template=prompt_template),
    max_length=max_length,
    lazy=True,
    repeats=4,
    special_tokens=special_tokens,
    extra_image_processor=extra_image_processor, # Resize (1024,1024)
    sampled_frames=5,
)

################## image res
refcoco_segm_dataset=dict(
    type=ReferSegmDataset_box,
    tokenizer=tokenizer,
    special_tokens=special_tokens,
    extra_image_processor=extra_image_processor,
    data_root='/home/rqshen/refer_seg/refcoco',
    data_prefix=dict(img_path='coco2014/train2014/'),
    ann_file='instances.json',
    split_file='refs(unc).p',
    prompt_template=prompt_template,
    num_classes_per_sample=5,
    max_length=max_length,
)

refcoco_plus_segm_dataset=dict(
    type=ReferSegmDataset_box,
    tokenizer=tokenizer,
    special_tokens=special_tokens,
    extra_image_processor=extra_image_processor,
    data_root='/home/rqshen/refer_seg/refcoco+',
    data_prefix=dict(img_path='coco2014/train2014/'),
    ann_file='instances.json',
    split_file='refs(unc).p',
    prompt_template=prompt_template,
    num_classes_per_sample=5,
    max_length=max_length,
)

refcocog_segm_dataset=dict(
    type=ReferSegmDataset_box,
    tokenizer=tokenizer,
    special_tokens=special_tokens,
    extra_image_processor=extra_image_processor,
    data_root='/home/rqshen/refer_seg/refcocog',
    data_prefix=dict(img_path='coco2014/train2014/'),
    ann_file='instances.json',
    split_file='refs(umd).p',
    prompt_template=prompt_template,
    num_classes_per_sample=5,
    max_length=max_length,
)

train_dataset = dict(
    type=ConcatDataset, datasets=[
        # image
        refcoco_segm_dataset, refcoco_plus_segm_dataset, refcocog_segm_dataset,
        # video
        video_revos_dataset,
        video_mevis_dataset,
        video_refytvos_dataset, 
        # spatial
        video_spatial_relative_dataset,
        video_spatial_absolute_dataset,
    ]
)

train_dataloader = dict(
    batch_size=batch_size,
    num_workers=dataloader_num_workers,
    dataset=train_dataset,
    sampler=dict(
        type=LengthGroupedSampler,
        length_property='modality_length',
        per_device_batch_size=batch_size * accumulative_counts),
    collate_fn=dict(type=video_lisa_collate_fn)
)


#######################################################################
#                    PART 4  Scheduler & Optimizer                    #
#######################################################################
# optimizer
optim_wrapper = dict(
    type=AmpOptimWrapper,
    optimizer=dict(
        type=optim_type, lr=lr, betas=betas, weight_decay=weight_decay),
    clip_grad=dict(max_norm=max_norm, error_if_nonfinite=False),
    accumulative_counts=accumulative_counts,
    loss_scale='dynamic',
    dtype='bfloat16'
)

# learning policy
# More information: https://github.com/open-mmlab/mmengine/blob/main/docs/en/tutorials/param_scheduler.md  # noqa: E501
param_scheduler = [
    dict(
        type=LinearLR,
        start_factor=1e-5,
        by_epoch=True,
        begin=0,
        end=warmup_ratio * max_epochs,
        convert_to_iter_based=True),
    dict(
        type=CosineAnnealingLR,
        eta_min=0.0,
        by_epoch=True,
        begin=warmup_ratio * max_epochs,
        end=max_epochs,
        convert_to_iter_based=True)
]

train_cfg = dict(type=TrainLoop, max_epochs=max_epochs)

#######################################################################
#                           PART 5  Runtime                           #
#######################################################################
# Log the dialogue periodically during the training process, optional
custom_hooks = [
    # dict(type=DatasetInfoHook, tokenizer=tokenizer),
]

# configure default hooks
default_hooks = dict(
    # record the time of every iteration.
    timer=dict(type=IterTimerHook),
    # print log every 10 iterations.
    logger=dict(type=LoggerHook, log_metric_by_epoch=False, interval=10),
    # enable the parameter scheduler.
    param_scheduler=dict(type=ParamSchedulerHook),
    # save checkpoint per `save_steps`.
    checkpoint=dict(
        type=CheckpointHook,
        save_optimizer=False,
        by_epoch=False,
        interval=save_steps,
        max_keep_ckpts=save_total_limit),
    # set sampler seed in distributed evrionment.
    sampler_seed=dict(type=DistSamplerSeedHook),
)


# configure environment
env_cfg = dict(
    # whether to enable cudnn benchmark
    cudnn_benchmark=False,
    # set multi process parameters
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    # set distributed parameters
    dist_cfg=dict(backend='nccl'),
)

# set visualizer
visualizer = None

# set log level
log_level = 'INFO'

# load from which checkpoint
load_from = None

# whether to resume training from the loaded checkpoint
resume = False

# Defaults to use random seed and disable `deterministic`
randomness = dict(seed=None, deterministic=False)

# set log processor
log_processor = dict(by_epoch=False)
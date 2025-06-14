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


# export TORCH_DISTRIBUTED_DEBUG=DETAIL

#######################################################################
#                          PART 1  Settings                           #
#######################################################################
# Model
path='OpenGVLab/InternVL2_5-8B' # InternVL  

# pretrained_pth='checkpoints/sam2.1_hiera_large.pt'

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

# special_tokens = ['[SEG]', '<p>', '</p>', '<vp>', '</vp>', '[RELATION]']
# special_tokens = ['[SEG]', '<p>', '</p>', '<vp>', '</vp>']
# special_tokens = ['[BOX]', '<p>', '</p>', '<vp>', '</vp>']
special_tokens = ['<box_sep>', '<no_box>', '<next>', '<p>', '</p>', '<vp>', '</vp>']
# special_tokens = ['<p>', '</p>', '<vp>', '</vp>']  

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
    # grounding_encoder=dict(
    #     type=SAM2TrainRunner,
    # ),
    # loss_box_iou=dict(
    #     type='third_parts.mmdet.models.losses.iou_loss.CIoULoss',
    #     reduction='mean',
    #     loss_weight=2.0),
    # loss_box_l1=dict(
    #     type='third_parts.mmdet.models.losses.l1_loss.SmoothL1Loss',
    #     reduction='mean',
    #     loss_weight=20.0),
    # loss_mask=dict(
    #     type=CrossEntropyLoss,
    #     use_sigmoid=True,
    #     reduction='mean',
    #     loss_weight=2.0),
    # loss_dice=dict(
    #     type=DiceLoss,
    #     use_sigmoid=True,
    #     activate=True,
    #     reduction='mean',
    #     naive_dice=True,
    #     eps=1.0,
    #     loss_weight=0.5), 
    # pretrained_pth=pretrained_pth,
    # loss_sample_points=True,
    # loss_sample_points=False,  
    bs=batch_size,
)

#######################################################################
#                      PART 3  Dataset & Dataloader                   #
#######################################################################

VIDEO_DATAS = './data/video_datas/'
IMG_DATAS = './data/image_datas/'

############### video res
# data_root_revos = '/mnt/data_hdd/rqshen/lisa_data/ReVOS/'
# video_revos_image_folder = data_root_revos + 'JPEGImages'
# video_revos_expression_file = data_root_revos + 'meta_expressions_train_.json'
# video_revos_mask_file = data_root_revos + 'mask_dict.json'

data_root_mevis = '/home/volume_shared/share_datasets/MeViS/MeViS_release/train/'
video_mevis_image_folder = data_root_mevis + 'JPEGImages'
video_mevis_expression_file = data_root_mevis + 'meta_expressions.json'
video_mevis_box_file = data_root_mevis + 'bbox.json'
# video_mevis_mask_file = data_root_mevis + 'mask_dict.json'

# Relation dataset
# all_vos
# video_relation_image_folder = '/mnt/data_hdd/rqshen/relation_dataset/JPEGImages'
# video_relation_expression_file = '/mnt/data_hdd/rqshen/Sa2VA/archive/all_vos_expressions.json'
# video_relation_mask_file = '/mnt/data_hdd/rqshen/Sa2VA/archive/all_vos_mask.json'

# all_vos2
# video_relation_image_folder = '/mnt/data_hdd/rqshen/all_vos2_relation/JPEGImages'
# video_relation_expression_file = '/mnt/data_hdd/rqshen/sam2_original_copy/all_vos2_step2.json'
# video_relation_mask_file = '/mnt/data_hdd/rqshen/sam2_original_copy/all_vos2_mask.json'

# all_vos_depth
# video_relation_image_folder = '/mnt/data_hdd/rqshen/all_vos/JPEGImages'
# video_relation_image_folder = '/mnt/data_hdd/rqshen/all_vos_depth/JPEGImages'
# video_relation_expression_file = '/mnt/data_hdd/rqshen/sam2_original_copy/all_vos_step2_depth.json'
# video_relation_mask_file = '/mnt/data_hdd/rqshen/sam2_original_copy/all_vos_mask_depth.json'

data_root_refytvos = '/home/volume_shared/share_datasets/Ref-Youtube-VOS/'
video_refytvos_image_folder = data_root_refytvos + 'train/JPEGImages/'
video_refytvos_expression_file = data_root_refytvos + 'meta_expressions/train/meta_expressions.json'
# video_refytvos_mask_file = data_root_refytvos + 'train/mask_dict.pkl'
video_refytvos_box_file = '/home/rqshen/ref_ytvos_train_box.json'

# video_revos_dataset = dict(
#     type=VideoReVOSDataset_box,
#     image_folder=video_revos_image_folder,
#     expression_file=video_revos_expression_file,
#     mask_file=video_revos_mask_file,
#     tokenizer=tokenizer,
#     template_map_fn=dict(
#         type=template_map_fn_factory, template=prompt_template),
#     max_length=max_length,
#     lazy=True,
#     repeats=10,
#     special_tokens=special_tokens,
#     extra_image_processor=extra_image_processor, # Resize (1024,1024)
#     sampled_frames=5,
# )

video_mevis_dataset = dict(
    type=VideoMeVISDataset_box,
    image_folder=video_mevis_image_folder, # rgbs
    expression_file=video_mevis_expression_file, # expressions
    box_file=video_mevis_box_file, # boxes
    # mask_file=video_mevis_mask_file, # masks
    tokenizer=tokenizer, # 151673 (151643+40)
    template_map_fn=dict(
        type=template_map_fn_factory, template=prompt_template),
    max_length=max_length,
    lazy=True,
    repeats=4,
    special_tokens=special_tokens,
    extra_image_processor=extra_image_processor,
    sampled_frames=4,
)

video_refytvos_dataset = dict(
    type=VideoRefYoutubeVOSDataset_box,
    image_folder=video_refytvos_image_folder,
    expression_file=video_refytvos_expression_file,
    box_file=video_refytvos_box_file,
    # mask_file=video_refytvos_mask_file,
    tokenizer=tokenizer,
    template_map_fn=dict(
        type=template_map_fn_factory, template=prompt_template),
    max_length=max_length,
    lazy=True,
    repeats=4,
    special_tokens=special_tokens,
    extra_image_processor=extra_image_processor,
    sampled_frames=4,
)

# ################### Video chat

# data_root_video_chatunivi = VIDEO_DATAS + 'video_vlm/video_chat/'
# video_chatunivi_image_folder = data_root_video_chatunivi + 'Activity_Videos/'
# video_chatunivi_json_file = data_root_video_chatunivi+ 'video_chat.json'

# video_qa_dataset = dict(
#     type=VideoChatUniViDataset,
#     image_folder=video_chatunivi_image_folder,
#     json_file=video_chatunivi_json_file,
#     tokenizer=tokenizer,
#     template_map_fn=dict(
#         type=template_map_fn_factory, template=prompt_template),
#     max_length=max_length,
#     lazy=True,
#     repeats=1,
#     special_tokens=special_tokens,
#     extra_image_processor=extra_image_processor,
#     sampled_frames=5,
# )

# ################## image chat

# llava_vqa_dataset = dict(
#     type=LLaVADataset,
#     tokenizer=tokenizer,
#     data_path='data/llava_data/LLaVA-Instruct-150K/llava_v1_5_mix665k.json',
#     prompt_template=prompt_template,
#     special_tokens=special_tokens,
#     image_folder='data/llava_data/llava_images/',
# )

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

# # image gcg datas
# glamm_data_root = './data/glamm_data/'

# refcocog_image_path = glamm_data_root + 'images/coco2014/train2014/'
# refcocog_ann_file = glamm_data_root + 'annotations/RefCOCOg_GCG_train.json'

# grandf_image_path = glamm_data_root + 'images/grandf/train/'
# grandf_ann_file = glamm_data_root + 'annotations/GranDf_HA_GCG_train.json'

# flickr_image_path = glamm_data_root + 'images/flickr30k/Flickr30K/'
# flickr_ann_file = glamm_data_root + 'annotations/flickr_mergedGT_GCG_train.json'

# psg_image_path = glamm_data_root + 'images/coco2017/'
# psg_ann_file = glamm_data_root + 'annotations/OpenPsgGCG_train.json'

# glamm_refcocog_dataset = dict(
#     type=RefCOCOgGCGDataset,
#     image_folder=refcocog_image_path,
#     data_path=refcocog_ann_file,
#     tokenizer=tokenizer,
#     max_length=max_length,
#     special_tokens=special_tokens,
#     template_map_fn=dict(type=template_map_fn_factory, template=prompt_template),
#     extra_image_processor=extra_image_processor,
#     lazy=True,
#     repeats=1,
# )

# glamm_grandf_dataset = dict(
#     type=GranDfGCGDataset,
#     data_path=grandf_ann_file,
#     image_folder=grandf_image_path,
#     tokenizer=tokenizer,
#     max_length=max_length,
#     special_tokens=special_tokens,
#     template_map_fn=dict(type=template_map_fn_factory, template=prompt_template),
#     extra_image_processor=extra_image_processor,
#     lazy=True,
#     repeats=10,
# )

# glamm_psg_dataset = dict(
#     type=OpenPsgGCGDataset,
#     data_path=psg_ann_file,
#     image_folder=psg_image_path,
#     tokenizer=tokenizer,
#     max_length=max_length,
#     special_tokens=special_tokens,
#     template_map_fn=dict(type=template_map_fn_factory, template=prompt_template),
#     extra_image_processor=extra_image_processor,
#     lazy=True,
#     repeats=1,
# )

# glamm_flickr_dataset = dict(
#     type=FlickrGCGDataset,
#     data_path=flickr_ann_file,
#     image_folder=flickr_image_path,
#     tokenizer=tokenizer,
#     max_length=max_length,
#     special_tokens=special_tokens,
#     template_map_fn=dict(type=template_map_fn_factory, template=prompt_template),
#     extra_image_processor=extra_image_processor,
#     lazy=True,
#     repeats=1,
# )

# sam2 data
# data_sam2_folder = VIDEO_DATAS + 'segmentation_datasets/sam_v_full/'
# data_sam2_folder = '/mnt/data_hdd/share_data/sa-v'
# data_sam2_expression_file = './whole_pesudo_cap_v3/sam_v_final_v3.json'
# data_sam2_expression_file = '/mnt/data_hdd/rqshen/ref_sav/Ref-SAV.json'

# video_sam2_dataset = dict(
#     type=VideoSAM2Dataset_box,
#     sam2_folder=data_sam2_folder,
#     expression_file=data_sam2_expression_file,
#     tokenizer=tokenizer,
#     template_map_fn=dict(
#         type=template_map_fn_factory, template=prompt_template),
#     max_length=max_length,
#     lazy=True,
#     repeats=4,
#     special_tokens=special_tokens,
#     extra_image_processor=extra_image_processor,
#     sampled_frames=5,
#     select_number=5,
# )

# # osprey
# data_osprey_file = VIDEO_DATAS + 'osprey-724k/Osprey-724K/osprey_conversation.json'
# data_osprey_image_folders = [
#     IMG_DATAS+ 'coco/train2014/',
#     IMG_DATAS + 'coco/val2014/',
#     IMG_DATAS + 'coco/train2017/',
#     IMG_DATAS + 'coco/val2017/',
# ]

# image_osprey_dataset = dict(
#     type=OspreyDataset,
#     image_folder=data_osprey_image_folders,
#     data_path=data_osprey_file,
#     tokenizer=tokenizer,
#     template_map_fn=dict(
#         type=template_map_fn_factory, template=prompt_template),
#     max_length=max_length,
#     lazy=True,
#     repeats=1,
#     special_tokens=special_tokens,
# )

# data_osprey_detail_description_file = VIDEO_DATAS + 'osprey-724k/Osprey-724K/osprey_detail_description.json'
# image_osprey_description_dataset = dict(
#     type=OspreyDescriptionDataset,
#     image_folder=data_osprey_image_folders,
#     data_path=data_osprey_detail_description_file,
#     tokenizer=tokenizer,
#     template_map_fn=dict(
#         type=template_map_fn_factory, template=prompt_template),
#     max_length=max_length,
#     lazy=True,
#     repeats=1,
#     special_tokens=special_tokens,
# )

# data_osprey_short_file = VIDEO_DATAS + 'osprey-724k/Osprey-724K/osprey_short_form.json'
# image_osprey_short_dataset = dict(
#     type=OspreyShortDescriptionDataset,
#     image_folder=data_osprey_image_folders,
#     data_path=data_osprey_short_file,
#     tokenizer=tokenizer,
#     template_map_fn=dict(
#         type=template_map_fn_factory, template=prompt_template),
#     max_length=max_length,
#     lazy=True,
#     repeats=1,
#     special_tokens=special_tokens,
# )

# data_osprey_part_file = VIDEO_DATAS + 'osprey-724k/Osprey-724K/osprey_part_level.json'
# image_osprey_part_dataset = dict(
#     type=OspreyDataset,
#     image_folder=data_osprey_image_folders,
#     data_path=data_osprey_part_file,
#     tokenizer=tokenizer,
#     template_map_fn=dict(
#         type=template_map_fn_factory, template=prompt_template),
#     max_length=max_length,
#     lazy=True,
#     repeats=1,
#     special_tokens=special_tokens,
# )

# data_osprey_positive_neg_file = VIDEO_DATAS + 'osprey-724k/Osprey-724K/osprey_lvis_positive_negative.json'
# image_osprey_positive_neg_dataset = dict(
#     type=OspreyDataset,
#     image_folder=data_osprey_image_folders,
#     data_path=data_osprey_positive_neg_file,
#     tokenizer=tokenizer,
#     template_map_fn=dict(
#         type=template_map_fn_factory, template=prompt_template),
#     max_length=max_length,
#     lazy=True,
#     repeats=1,
#     special_tokens=special_tokens,
# )

train_dataset = dict(
    type=ConcatDataset, datasets=[
        # sem seg
        # semantic_seg_ade20k_dataset,
        # ref seg
        refcoco_segm_dataset, refcoco_plus_segm_dataset, refcocog_segm_dataset,
        # refcoco_segm_dataset, refcoco_plus_segm_dataset, refcocog_segm_dataset,
        # refcoco_segm_dataset, refcoco_plus_segm_dataset, refcocog_segm_dataset,
        # refcoco_segm_dataset, refcoco_plus_segm_dataset, refcocog_segm_dataset,
        # image qa
        # llava_vqa_dataset,
        # video res
        video_mevis_dataset, video_mevis_dataset, video_mevis_dataset,
        video_refytvos_dataset, video_refytvos_dataset, video_refytvos_dataset,
        # video chat
        # video_qa_dataset,
        # sam2 pesudo
        # video_sam2_dataset,
        # gcg data
        # glamm_psg_dataset,
        # glamm_grandf_dataset,
        # glamm_flickr_dataset,
        # glamm_refcocog_dataset,
        # # visual prompt
        # image_osprey_dataset, image_osprey_description_dataset,
        # image_osprey_part_dataset, image_osprey_short_dataset,
        # image_osprey_positive_neg_dataset,
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


# test_dataset = dict(
#     type=ConcatDataset, datasets=[
#         # video_mevis_dataset,
#         video_relation_dataset,
#     ]
# )

# test_dataloader = dict(
#     batch_size=batch_size,
#     num_workers=dataloader_num_workers,
#     dataset=test_dataset,
#     sampler=dict(
#         type=LengthGroupedSampler,
#         length_property='modality_length',
#         per_device_batch_size=batch_size * accumulative_counts),
#     collate_fn=dict(type=video_lisa_collate_fn)
# )

# test_evaluator = [
#     dict(type='MaskIoUEvaluator', iou_thresh=0.5),
#     dict(type='MaskAccEvaluator'),
# ]


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

# train, val, test setting
train_cfg = dict(type=TrainLoop, max_epochs=max_epochs)

# test_cfg = dict(type=TrainLoop, max_epochs=max_epochs)


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
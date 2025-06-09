import os
from .ReVOS_Dataset_box import VideoReVOSDataset_box
import pickle
import logging
import os
from typing import Literal

import torch
from datasets import Dataset as HFDataset
from datasets import DatasetDict
from mmengine import print_log
from PIL import Image
from torch.utils.data import Dataset
import numpy as np

from xtuner.registry import BUILDER
from xtuner.dataset.huggingface import build_origin_dataset
import copy

from .encode_fn import video_lisa_encode_fn_spatial_switch
import json
import random
import pycocotools.mask as maskUtils
import cv2
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
import pdb

SWITCH_BOX_QUESTIONS = [
    "Can you track the {direction}most object in this video?",
    "Track the object that is currently at the {direction}most position.",
    "Keep track of whatever object is at the {direction}most position.",
    "Track whatever object is at the {direction}most position throughout this video sequence.",
    "Keep track of whatever object that is at the {direction}most position.",
    "Track and locate the {direction}most object in the video.",
    "Track and identify the {direction}most object throughout the video.",
    "Identify the {direction}most object(s) in the video.",
]

SWITCH_BOX_ANSWERS = [
    "The {direction}most positions are occupied by objects at <box>{coords}</box>.",
    "Sure, objects appear at the {direction}most position are <box>{coords}</box>.",
    
    "The objects at the {direction}most position appear at <box>{coords}</box>.",
    "The objects occupying the {direction}most position are at <box>{coords}</box>.",
    
    "Sure, the objects are at <box>{coords}</box> throughout the video.",
    "Sure, the objects are at <box>{coords}</box> in the video.",
]

class VideoSpatialSwitchDataset_box(VideoReVOSDataset_box):
    def json_file_preprocess(self, expression_file, box_file):
        with open(box_file, 'rb') as f:
            box_dict = json.load(f)
            
        with open(expression_file, 'r') as f:
            expression_datas = json.load(f)['videos']
            
        metas = []
        anno_count = 0 
        vid2metaid = {}

        for vid_name in expression_datas:
            vid_express_data = expression_datas[vid_name]

            vid_frames = sorted(vid_express_data['frames'])
            vid_len = len(vid_frames)

            exp_id_list = sorted(list(vid_express_data['expressions'].keys()))
            for exp_id in exp_id_list:
                # 目前 exp_id应该是只有一个
                assert len(exp_id_list) == 1
                exp_dict = vid_express_data['expressions'][exp_id]
                meta = {}
                meta['video'] = vid_name
                
                meta['exp'] = exp_dict['exp']  # str
                meta['exp_id'] = exp_id
        
                assert 'obj_id' and 'anno_id' in exp_dict['switches'][0].keys()
                meta['obj_id'] = [exp_dict['switches'][i]['obj_id'] for i in range(len(exp_dict['switches']))]
                meta['mask_anno_id'] = [exp_dict['switches'][i]['anno_id'] for i in range(len(exp_dict['switches']))]
                
                meta['anno_id'] = [str(anno_count), ]
                anno_count += 1
                
                meta['switch_frames'] = [switch['start_frame'] for switch in exp_dict['switches']]

                # overall 
                meta['frames'] = vid_frames
                meta['length'] = vid_len
                
                meta['start'] = vid_express_data['start']
                meta['end'] = vid_express_data['end']
                
                metas.append(meta)
                
                if vid_name not in vid2metaid.keys():
                    vid2metaid[vid_name] = []
                vid2metaid[vid_name].append(len(metas) - 1)

        return vid2metaid, metas, box_dict
        # mask_dict: key是 anno_id (target_objects)
        # metas: key是 expressions
        # vid2metaid: key是 video_name, value是 meta_id (meta即expression)


    def prepare_text(self, n_frames, expressions, num_image_tokens=256):
        frame_token_str = f'{self.IMG_START_TOKEN}' \
                          f'{self.IMG_CONTEXT_TOKEN * num_image_tokens}' \
                        f'{self.IMG_END_TOKEN}'

        questions = []
        answers = []
        # W, H = org_image_size
        for i, exp in enumerate(expressions):
            assert '?' not in exp
            direction = exp.split('most')[0]
                    
            question_template = random.choice(SWITCH_BOX_QUESTIONS)
            question = question_template.format(direction=direction)
            questions.append(question)
            answer_template = random.choice(SWITCH_BOX_ANSWERS)
            answer = answer_template.format(direction=direction, coords="{coords}")  
            answers.append(answer)
        
        
        qa_list = []
        for i, (question, answer) in enumerate(zip(questions, answers)):
            if i == 0:
                frame_tokens = frame_token_str + '\n'  # 1个patch
                frame_tokens = frame_tokens * n_frames  # n个patches
                frame_tokens = frame_tokens.strip()
                qa_list.append(
                    {'from': 'human', 'value': frame_tokens + question}
                )
            else:
                qa_list.append(
                    {'from': 'human', 'value': question}
                )
            qa_list.append(
                {'from': 'gpt', 'value': answer}
            )

        input = ''
        conversation = []
        for msg in qa_list: # 
            if msg['from'] == 'human':
                input += msg['value']
            elif msg['from'] == 'gpt':
                conversation.append({'input': input, 'output': msg['value']})
                input = '' # 清零input, 为下一个expression, for each video, 一共有self.select_number个expressions
            else:
                raise NotImplementedError

        # add system information
        conversation[0].update({'system': self._system})
        return {'conversation': conversation}#


    def dataset_map_fn(self, data_dict, select_k=5, org_image_size=None):
        # data_dict的长度, 来源于self.select_number, 也就是在一个video中, 选多少expressions
        images = []

        frames = data_dict[0]['frames']
        len_frames = len(frames)
        start_idx = data_dict[0]['start']
        end_idx = data_dict[0]['end']
        assert 0 <= start_idx <= end_idx < len_frames, f"start/end 越界: start={start_idx}, end={end_idx}, len={len_frames}"

        for objet_info in data_dict:
            assert len_frames == len(objet_info['frames'])
        
        valid_frames = []
        invalid_frames = []
        for frame_idx in range(start_idx, end_idx + 1):
            has_object = False
            for object_info in data_dict:
                anno_ids = object_info['mask_anno_id']
                unique_anno_ids = list(set(anno_ids))
                for unique_id in unique_anno_ids:
                    mask = self.mask_dict[str(unique_id)][frame_idx]
                    if mask is not None:
                        has_object = True
                        break
                if has_object:
                    break
            if has_object:
                valid_frames.append(frame_idx)
            else:
                invalid_frames.append(frame_idx)
        
        if len(valid_frames) >= select_k:
            selected_frame_indexes = np.random.choice(valid_frames, select_k, replace=False)
        else:
            print(f"Not enough valid frames (frames with objects) to sample. Found {len(valid_frames)} valid frames, need {select_k}")
            selected_frame_indexes = valid_frames.copy()
            remaining_frames_needed = select_k - len(valid_frames)
            additional_frames = np.random.choice(invalid_frames, remaining_frames_needed, replace=False)
            selected_frame_indexes.extend(additional_frames)
    
        selected_frame_indexes = sorted(selected_frame_indexes)
        self.selected_frame_indexes = selected_frame_indexes

        # valid_range = np.arange(start_idx, end_idx + 1)
        # if len(valid_range) >= select_k:
        #     selected_frame_indexes = np.random.choice(valid_range, select_k, replace=False)
        # else:
        #     selected_frame_indexes = np.random.choice(valid_range, select_k, replace=True)
        # selected_frame_indexes.sort()
        # self.selected_frame_indexes = selected_frame_indexes

        if self.use_fast:
            # sample fast branch
            fast_interval = len_frames / (self.n_fast_images + 1e-4)
            sampled_fast_frame_idxs = [min(int(i * fast_interval), len_frames - 1) for i in range(self.n_fast_images)]
            fast_video_frames = []
            for selected_frame_index in sampled_fast_frame_idxs:
                frame_id = data_dict[0]['frames'][selected_frame_index]
                fast_video_frames.append(os.path.join(data_dict[0]['video'], frame_id + '.jpg'))
        else:
            fast_video_frames = None
            sampled_fast_frame_idxs = None

        for selected_frame_index in selected_frame_indexes:
            frame_id = data_dict[0]['frames'][selected_frame_index]
            images.append(os.path.join(data_dict[0]['video'], frame_id + '.jpg'))

        # prepare text
        expressions = [object_info['exp'] for object_info in data_dict]
        if self.use_fast: # False
            text_dict = self.prepare_text(select_k, expressions, num_image_tokens=self.patch_token, n_fast_images=len(fast_video_frames))
        else:
            text_dict = self.prepare_text(select_k, expressions, num_image_tokens=self.patch_token) 
        
        video_boxes = []
        for object_info in data_dict:
            # 这个video的每个selected expression
            anno_ids = object_info['mask_anno_id']
            unique_anno_ids = list(set(anno_ids))
            # obj_masks = []
            frames_masks_ = []
            
            for frame_idx in selected_frame_indexes:
                current_frame_masks = []
                for unique_id in unique_anno_ids:
                    mask = self.mask_dict[str(unique_id)][frame_idx]
                    if mask is not None: 
                        current_frame_masks.append(mask)

                # assert len(current_frame_masks) == 1, f"Frame {frame_idx} has {len(current_frame_masks)} masks, expected exactly 1"
                assert len(current_frame_masks) <= 1, f"Frame {frame_idx} has {len(current_frame_masks)} masks, expected only 0 or 1"
                if len(current_frame_masks) == 0:
                    current_frame_masks.append(None)
                    print(f"Frame {frame_idx} has no object, so we use None, anno_ids: {unique_anno_ids}")
                    
                frames_masks_.append(copy.deepcopy(current_frame_masks[0]))
            
            # for anno_id in anno_ids:
            #     # 这个expression包含几个objects 
            #     anno_id = str(anno_id)
            #     frames_masks = self.mask_dict[anno_id]
            #     frames_masks_ = []
            #     for frame_idx in selected_frame_indexes:
            #         current_frame_masks = []
            #         for unique_id in unique_anno_ids:
            #             mask = self.mask_dict[str(unique_id)][frame_idx]
            #             if mask is not None:
            #                 current_frame_masks.append(unique_id)
            #         assert len(current_frame_masks) == 1, f"Frame {frame_idx} has {len(current_frame_masks)} non-None masks, expected only 1"
            #         frames_masks_.append(copy.deepcopy(frames_masks[frame_idx]))
            
            # obj_masks.append(frames_masks_)
            video_boxes.append(frames_masks_)
            # pdb.set_trace()
        
        if self.use_fast:
            fast_video_masks = []
            assert sampled_fast_frame_idxs is not None
            for object_info in data_dict:
                anno_ids = object_info['mask_anno_id']
                obj_masks = []
                for anno_id in anno_ids:
                    anno_id = str(anno_id)
                    frames_masks = self.mask_dict[anno_id]
                    frames_masks_ = []
                    for frame_idx in sampled_fast_frame_idxs:
                        frames_masks_.append(copy.deepcopy(frames_masks[frame_idx]))
                    obj_masks.append(frames_masks_)
                fast_video_masks.append(obj_masks)
        else:
            fast_video_masks = None

        ret = {'images': images, 'video_boxes': video_boxes, 'conversation': text_dict['conversation'], 'fast_images': fast_video_frames, 'fast_video_masks': fast_video_masks}
        return ret


    def decode_bbox(self, video_bboxes, image_size):
        ori_height, ori_width = image_size
        all_expressions_areas = []
        all_expressions = []
        
        for exp_id, exp_box in enumerate(video_bboxes):
            len_frames = len(exp_box)
            bbox_per_frame = [0.0] * len_frames
            box_expressions = [] 
            
            for frame_idx in range(len_frames):
                # 每帧
                frame_total_area = 0.0
                valid_coords_per_obj = []
                
                if exp_box[frame_idx] is not None:
                    x1, y1, x2, y2 = exp_box[frame_idx]
                    norm_x1 = x1 / ori_width
                    norm_y1 = y1 / ori_height
                    norm_x2 = x2 / ori_width
                    norm_y2 = y2 / ori_height
                    area = (norm_x2 - norm_x1) * (norm_y2 - norm_y1)
                    frame_total_area += area

                    coord_str = f"{norm_x1:.4f},{norm_y1:.4f},{norm_x2:.4f},{norm_y2:.4f}"
                    valid_coords_per_obj.append(coord_str)
        
                bbox_per_frame[frame_idx] = frame_total_area
                
                if valid_coords_per_obj:
                    box_expr = f"<box>{('<box_sep>'.join(valid_coords_per_obj))}</box>"
                else:
                    box_expr = "<box><no_box></box>"
                box_expressions.append(box_expr)

            all_expressions_areas.append(bbox_per_frame)
            all_expressions.append(box_expressions)
            # pdb.set_trace()
            
        return all_expressions_areas, all_expressions
    
    def __getitem__(self, index):
        # 每个video
        index = index % self.real_len()
        # print('index: ', index)
        selected_video_objects = self.vid2metaid[self.videos[index]]
        # len(self.vid2metaid): 1660 # altogether 1660 videos
        # self.vid2metaid['7fc4e406d39e'] = [23046, 23047, 23048, 23049, 23050] # altogether 23051 expressions
        video_objects_infos = [copy.deepcopy(self.text_data[idx]) for idx in selected_video_objects]

        self.select_number = 3 # 感觉对于switch, 不需要那么多，因为都是重复的
        
        if len(video_objects_infos) > self.select_number:
            # select_number: 最多选择几个anno_ids(几个expressions)
            selected_indexes = np.random.choice(len(video_objects_infos), self.select_number) # 允许重复取样
            # selected_indexes = np.random.choice(len(video_objects_infos), self.select_number, replace=False) # 不允许重复取样
            video_objects_infos = [video_objects_infos[_idx] for _idx in selected_indexes]
        else:
            selected_indexes = np.random.choice(len(video_objects_infos), self.select_number, replace=True) # 允许重复取样
            video_objects_infos = [video_objects_infos[_idx] for _idx in selected_indexes]

        org_image_size = Image.open(os.path.join(self.image_folder, self.videos[index], video_objects_infos[0]['frames'][0] + ".jpg")).size
        
        data_dict = self.dataset_map_fn(video_objects_infos, select_k=self.sampled_frames, org_image_size=org_image_size)
        # video_objects_infos 是按照 expressions 来筛选出的

        assert 'images' in data_dict.keys()
        pixel_values = []
        extra_pixel_values = []
        num_video_tokens = None
        num_frame_tokens = None
        
        # pdb.set_trace()
        
        if data_dict.get('images', None) is not None:
            frames_files = data_dict['images']
            frames_files = [os.path.join(self.image_folder, frame_file) for frame_file in frames_files]
            for frame_path in frames_files:
                frame_image = Image.open(frame_path).convert('RGB')
                ori_width, ori_height = frame_image.size
                if self.extra_image_processor is not None:
                    g_image = np.array(frame_image)  # for grounding
                    g_image = self.extra_image_processor.apply_image(g_image)
                    g_pixel_values = torch.from_numpy(g_image).permute(2, 0, 1).contiguous()
                    extra_pixel_values.append(g_pixel_values)

                if self.preprocessor is not None:
                    pass
                else:
                    frame_image = self.transformer(frame_image)
                pixel_values.append(frame_image)
            # pdb.set_trace()
            
            if self.preprocessor is not None:
                if self.arch_type == 'qwen':
                    _data_dict = self.preprocessor(pixel_values, do_resize=True, size=(self.image_size, self.image_size))
                    _data_dict['pixel_values'] = torch.tensor(_data_dict['pixel_values'], dtype=torch.float)
                    _data_dict['image_grid_thw'] = torch.tensor(_data_dict['image_grid_thw'], dtype=torch.int)
                    num_frame_tokens = int(_data_dict['image_grid_thw'][0].prod() * (self.downsample_ratio ** 2))
                    num_frames = _data_dict['image_grid_thw'].shape[0]
                    num_video_tokens = num_frame_tokens * num_frames
                elif self.arch_type == 'llava':
                    _data_dict = self.preprocessor(pixel_values, do_resize=True, size=(self.image_size, self.image_size))
                    _data_dict['pixel_values'] = np.stack(_data_dict['pixel_values'], axis=0)
                    _data_dict['pixel_values'] = torch.tensor(_data_dict['pixel_values'], dtype=torch.float)
                else:
                    raise NotImplementedError
                data_dict.update(_data_dict)
            else:
                pixel_values = torch.stack(pixel_values, dim=0) # (n_f, 3, h, w)
                data_dict['pixel_values'] = pixel_values
            if self.extra_image_processor is not None:
                data_dict['g_pixel_values'] = extra_pixel_values
            
            # process and get masks
            bboxes_areas, bboxes_expression = self.decode_bbox(data_dict['video_boxes'], image_size=(ori_height, ori_width))
            
            data_dict['bboxes_areas'] = bboxes_areas  # 这个area是本来想搞IoU loss, 现在已经没用了
            data_dict['bboxes_expression'] = bboxes_expression
            
        else:
            data_dict['pixel_values'] = torch.zeros(0, 3, self.image_size, self.image_size)
            data_dict['bboxes_areas'] = None
            data_dict['bboxes_expression'] = None

        if num_video_tokens is not None: # False
            assert self.patch_token == 1
            input_str = data_dict['conversation'][0]['input']
            input_str = input_str.replace(self.IMG_CONTEXT_TOKEN, self.IMG_CONTEXT_TOKEN * num_frame_tokens)
            assert input_str.count(self.IMG_CONTEXT_TOKEN) == num_video_tokens
            data_dict['conversation'][0]['input'] = input_str

        switch_frames = video_objects_infos[0]['switch_frames']
        frames = video_objects_infos[0]['frames']
        switch_frames_idx = [frames.index(frame.split('.')[0]) for frame in switch_frames]
        switch_obj_ids = video_objects_infos[0]['obj_id']
        
        # self.selected_frame_indexes 和 switch_frames_idx 的比较
        real_switch_frames_idx = []
        for idx, selected_frame_index in enumerate(self.selected_frame_indexes):
            if idx > 0:
                for j in switch_frames_idx:
                    assert self.selected_frame_indexes[idx] == selected_frame_index
                    if self.selected_frame_indexes[idx] >= j and self.selected_frame_indexes[idx-1] < j: # switch了
                        real_switch_frames_idx.append(idx)
        real_switch_frames_idx = list(set(real_switch_frames_idx))
        # self.selected_frame_indexes中的index
        
        assert len(switch_obj_ids) == len(switch_frames_idx), f"switch_obj_ids: {switch_obj_ids}, switch_frames_idx: {switch_frames_idx}"
        for k in real_switch_frames_idx:
            closest, index_of_closest = max(((x, idx) for idx, x in enumerate(switch_frames_idx) if x <= self.selected_frame_indexes[k]), default=(None, None))
            closest_pre, index_of_closest_pre = max(((x, idx) for idx, x in enumerate(switch_frames_idx) if x <= self.selected_frame_indexes[k-1]), default=(None, None))
            # switch_frames_idx中的index
            if switch_obj_ids[index_of_closest_pre] == switch_obj_ids[index_of_closest]:
                real_switch_frames_idx.remove(k)
    
        data_dict['real_switch_frames_idx'] = real_switch_frames_idx
        result = self.template_map_fn(data_dict) # 丰富conversation的结构, 如增加SEP, BEG_WORDS, END_WORDS等
        data_dict.update(result)
        result = video_lisa_encode_fn_spatial_switch(data_dict, tokenizer=self.tokenizer, max_length=self.max_length)
        data_dict.update(result)

        # for fast branch
        if self.use_fast: # False
            fast_pixel_values = []
            frames_files = data_dict['fast_images']
            frames_files = [os.path.join(self.image_folder, frame_file) for frame_file in frames_files]
            for frame_path in frames_files:
                frame_image = Image.open(frame_path).convert('RGB')
                ori_width, ori_height = frame_image.size

                frame_image = self.transformer(frame_image)
                fast_pixel_values.append(frame_image)

            fast_pixel_values = torch.stack(fast_pixel_values, dim=0)  # (n_f, 3, h, w)
            data_dict['fast_pixel_values'] = fast_pixel_values

            # process and get masks
            masks = self.decode_mask(data_dict['fast_video_masks'], image_size=(ori_height, ori_width))

            if masks is None:
                return self.__getitem__(random.randint(0, self.real_len()))

            data_dict['fast_exists'] = masks.to(dtype=torch.int).sum(dim=(-2, -1)).ge(self.exist_thr).unsqueeze(-1)

            del data_dict['fast_video_masks']
        
        data_dict['type'] = 'video'
        return data_dict
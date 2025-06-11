import os
import sys
sys.path.append('/home/rqshen/sam2_original')
from sam2.build_sam import build_sam2_video_predictor
import json
import pdb
import re
import numpy as np
import shutil
import cv2
from PIL import Image
import torch
import argparse
from collections import defaultdict

def extract_box_content(text):
    pattern = r'<box>(.*?)</box>'
    match = re.search(pattern, text)
    return match.group(1) if match else None

def further_process_box_content(box_content, W, H):
    return [
        [] if frame_box.strip() == "<no_box>" else [
            [x * W, y * H, x2 * W, y2 * H] for x, y, x2, y2 in 
            [list(map(float, box.strip().split(','))) for box in frame_box.split('<box_sep>')]
        ]
        for frame_box in box_content.split('<next>')
    ]

def process_videos(videos, data, video_dir):
    device = "cuda:0"  # Use single GPU
    sam2_checkpoint = "checkpoints/sam2.1_hiera_large.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
    predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)
    
    error_cases = []
    temp_dir = "temp_frames"

    for video_name in videos:
        per_video_dir = os.path.join(video_dir, video_name)
        frame_names = [ 
            p for p in os.listdir(per_video_dir)
            if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
        ]
        frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
        W, H = Image.open(os.path.join(per_video_dir, frame_names[0])).size
        
        expressions = list(data[video_name].keys())
        for exp_text in expressions:
            print(f"Processing: {video_name}, {exp_text}")
            exp_data = data[video_name][exp_text]
            box_content = extract_box_content(exp_data['prediction'])
            try:
                boxes = further_process_box_content(box_content, W, H)
            except:
                print("error: ", video_name, exp_text)
                error_cases.append({
                    "video_name": video_name,
                    "expression": exp_text,
                    "boxes_length": len(boxes),
                    "frames_length": len(sampled_frames)
                })
                continue
            
            sampled_frames = exp_data['video_sampled_frames']
            try:
                assert len(boxes) == len(sampled_frames)
            except:
                if len(boxes) < len(sampled_frames):
                    sampled_frames = sampled_frames[:len(boxes)]
                else:
                    boxes = boxes[:len(sampled_frames)]
            
            output_scores_per_object = defaultdict(dict)
            
            for idx, frame_idx in enumerate(sampled_frames):
                os.makedirs(temp_dir, exist_ok=False)
                start_idx = sampled_frames[idx]
                end_idx = sampled_frames[idx+1] if idx < len(sampled_frames)-1 else len(frame_names)-1
                [shutil.copy2(os.path.join(per_video_dir, f"{i:05d}.jpg"), os.path.join(temp_dir, f"{i:05d}.jpg")) for i in range(start_idx, end_idx + 1)]
                
                if len(boxes[idx]) == 0:
                    shutil.rmtree(temp_dir)
                    for frame_offset in range(end_idx - start_idx + 1):
                        output_scores_per_object[start_idx + frame_offset][1] = None
                else:
                    for box_idx, box in enumerate(boxes[idx]):
                        inference_state = predictor.init_state(video_path=temp_dir)
                        ann_frame_idx = 0
                        ann_obj_id = box_idx + 1
                        
                        box_orig_array = np.array([[box[0], box[1]], 
                                                [box[2], box[3]]], dtype=np.float32)
                        _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
                            inference_state=inference_state,
                            frame_idx=ann_frame_idx,
                            obj_id=ann_obj_id,
                            box=box_orig_array,
                        )
                        
                        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
                            assert len(out_obj_ids) == 1
                            mask = (out_mask_logits > 0.0).cpu().numpy().squeeze()
                            mask = (mask * 255).astype(np.uint8)
                            output_scores_per_object[out_frame_idx+start_idx][ann_obj_id] = mask

                    shutil.rmtree(temp_dir)
                    
                    if idx == 0 and start_idx > 0:
                        os.makedirs(temp_dir, exist_ok=False)
                        [shutil.copy2(os.path.join(per_video_dir, f"{i:05d}.jpg"), os.path.join(temp_dir, f"{i:05d}.jpg")) for i in range(0, start_idx + 1)]
                        
                        for box_idx, box in enumerate(boxes[idx]): #
                            inference_state = predictor.init_state(video_path=temp_dir)
                            ann_frame_idx = start_idx
                            ann_obj_id = box_idx + 1
                            
                            box_orig_array = np.array([[box[0], box[1]], 
                                                    [box[2], box[3]]], dtype=np.float32)
                            _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
                                inference_state=inference_state,
                                frame_idx=ann_frame_idx,
                                obj_id=ann_obj_id,
                                box=box_orig_array,
                            )
                            
                            for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state,
                                                                                                            start_frame_idx=start_idx,
                                                                                                            max_frame_num_to_track=start_idx-0,
                                                                                                            reverse=True):
                                assert len(out_obj_ids) == 1
                                mask = (out_mask_logits > 0.0).cpu().numpy().squeeze()
                                mask = (mask * 255).astype(np.uint8)
                                output_scores_per_object[out_frame_idx][ann_obj_id] = mask

                        shutil.rmtree(temp_dir)
            
            if len(boxes[0]) == 0 and start_idx > 0:
                for i in range(0, start_idx, 1):
                    output_scores_per_object[i][1] = None
                    
            # Save masks
            for frame_idx in sorted(output_scores_per_object.keys()):
                combined_mask = np.zeros((H, W), dtype=np.uint8)
                
                for obj_id in output_scores_per_object[frame_idx].keys():
                    current_mask = output_scores_per_object[frame_idx][obj_id]
                    if current_mask is not None:
                        combined_mask[current_mask > 0] = 255
                
                os.makedirs(f"final_masks/{video_name}/{exp_text}", exist_ok=True)
                Image.fromarray(combined_mask).save(f"final_masks/{video_name}/{exp_text}/{frame_idx:05d}.png")

    return error_cases

if __name__ == "__main__":
    shutil.rmtree("temp_frames", ignore_errors=True)
    # shutil.rmtree("final_masks2", ignore_errors=True)
    # shutil.rmtree("error_cases.json", ignore_errors=True)

    mevis_json_file = "mevis_predictions.json"
    data = json.load(open(mevis_json_file,'r'))
    video_dir = "/home/volume_shared/share_datasets/MeViS/MeViS_release/valid/JPEGImages/"
    
    processed_videos = set(os.listdir("final_masks")) if os.path.exists("final_masks") else set()
    videos_to_process = [v for v in data.keys() if v not in processed_videos]
    
    # all_videos = list(data.keys())
    error_cases = process_videos(videos_to_process, data, video_dir)

    # Save error cases
    # with open("error_cases.json", 'w') as f:
    #     json.dump(error_cases, f, indent=4)
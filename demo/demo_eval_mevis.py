import argparse
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer
import cv2
from mmengine.visualization import Visualizer
import json
from tqdm import tqdm
import torch.distributed as dist
import pdb

def parse_args():
    parser = argparse.ArgumentParser(description='Video Reasoning Segmentation')
    parser.add_argument('--model_path', default="work_dirs/hf_model_spatial_refcoco_mevis_refytvos_relative_absolute_31000")
    parser.add_argument('--work-dir', default='OUTPUT_MEVIS_refcoco_mevis_refytvos_relative_absolute_31000')
    parser.add_argument('--text', type=str, default="<image>Please describe the video content.")
    parser.add_argument('--select', type=int, default=-1)
    args = parser.parse_args()
    return args

def visualize(pred_mask, video_name, exp_id, image_path, work_dir):
    visualizer = Visualizer()
    img = cv2.imread(image_path)
    visualizer.set_image(img)
    visualizer.draw_binary_masks(pred_mask, colors='g', alphas=0.4)
    visual_result = visualizer.get_image()
    output_path = os.path.join(work_dir, video_name, exp_id, os.path.basename(image_path))
    cv2.imwrite(output_path, visual_result)

if __name__ == "__main__":
    cfg = parse_args()
    model_path = cfg.model_path
    
    # Load model and tokenizer to CUDA device 1
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True
    ).to("cuda:0")

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True
    )
    
    # Load validation data
    mevis_val = json.load(open("/home/volume_shared/share_datasets/MeViS/MeViS_release/valid/meta_expressions.json", "r"))['videos']
    all_videos = list(mevis_val.keys())
    all_predictions = {}

    for video_name in all_videos:
        video = mevis_val[video_name]
        video_path = os.path.join("/home/volume_shared/share_datasets/MeViS/MeViS_release/valid/JPEGImages", video_name)
        
        # Load frames
        frames = sorted(list(os.listdir(video_path)))
        frames = [os.path.join(video_path, frame) for frame in frames]
        frame_rgbs = [Image.open(frame).convert('RGB') for frame in frames]
        assert len(frame_rgbs) == len(video['frames'])
        
        # Load expressions
        for exp_id, exp_data in video['expressions'].items():
            exp_name = exp_data['exp']
            print("video_name: ", video_name, "exp_name: ", exp_name)
            exp_name = f'<image>{exp_name}'
            
            # Move the frame tensor to CUDA device 1
            frame_rgbs_cuda = [frame for frame in frame_rgbs]  # Ensure frames are on CUDA device 1

            # Forward pass
            result = model.predict_forward(
                video=frame_rgbs_cuda,
                text=exp_name,
                tokenizer=tokenizer,
            )

            prediction = result['prediction']
            video_sampled_frames = result['video_sampled_frames']
            
            if video_name not in all_predictions:
                all_predictions[video_name] = {}
            
            all_predictions[video_name][exp_name] = {
                "prediction": prediction,
                "video_sampled_frames": video_sampled_frames
            }

            output_file = "mevis_predictions_5frames.json"
            with open(output_file, 'w') as f:
                json.dump(all_predictions, f, indent=4)

    print("Prediction process completed.")

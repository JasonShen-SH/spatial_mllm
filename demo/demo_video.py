import argparse
import os

from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer

import cv2
try:
    from mmengine.visualization import Visualizer
except ImportError:
    Visualizer = None
    print("Warning: mmengine is not installed, visualization is disabled.")


def parse_args():
    parser = argparse.ArgumentParser(description='Video Reasoning Segmentation')
    parser.add_argument('--image_path')
    # parser.add_argument('--model_path', default="work_dirs/hf_model_spatial_refcoco_mevis_refytvos_relative_absolute_31000")
    parser.add_argument('--model_path', default="work_dirs/hf_model_spatial_refcoco_mevis_refytvos_relative_absolute_switch")
    parser.add_argument('--work-dir', default='OUTPUT_DIR')
    parser.add_argument('--text', default="<image>Please describe the video content.")
    parser.add_argument('--select', type=int, default=-1)
    args = parser.parse_args()
    return args


def visualize(pred_mask, image_path, work_dir):
    visualizer = Visualizer()
    img = cv2.imread(image_path)
    visualizer.set_image(img)
    visualizer.draw_binary_masks(pred_mask, colors='g', alphas=0.4)
    visual_result = visualizer.get_image()

    output_path = os.path.join(work_dir, os.path.basename(image_path))
    cv2.imwrite(output_path, visual_result)

if __name__ == "__main__":
    cfg = parse_args()
    model_path = cfg.model_path
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path,
        trust_remote_code=True
    )

    # image_files = []
    image_paths = []
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff"}
    for filename in sorted(list(os.listdir(cfg.image_path))):
        if os.path.splitext(filename)[1].lower() in image_extensions:
            # image_files.append(filename)
            image_paths.append(os.path.join(cfg.image_path, filename))

    vid_frames = []
    for img_path in image_paths:
        img = Image.open(img_path).convert('RGB')
        vid_frames.append(img)

    # image = Image.open(cfg.image_path).convert('RGB')
    # print(f"The input is:\n{cfg.text}")
    # result = model.predict_forward(
    #     image=image,
    #     text=cfg.text,
    #     tokenizer=tokenizer,
    # )

    if cfg.select > 0:
        img_frame = vid_frames[cfg.select - 1]
        print(f"Selected frame {cfg.select}")
        # print(f"The input is:\n{cfg.text}")
        result = model.predict_forward(
            image=img_frame,
            text=cfg.text,
            tokenizer=tokenizer,
        )
    else:
        # print(f"The input is:\n{cfg.text}")
        result = model.predict_forward(
            video=vid_frames,
            text=cfg.text,
            tokenizer=tokenizer,
        )

    prediction = result['prediction']
    print(f"The output is:\n{prediction}")

    if '[SEG]' in prediction and Visualizer is not None:
        _seg_idx = 0
        pred_masks = result['prediction_masks'][_seg_idx]
        for frame_idx in range(len(vid_frames)):
            pred_mask = pred_masks[frame_idx]
            if cfg.work_dir:
                os.makedirs(cfg.work_dir, exist_ok=True)
                visualize(pred_mask, image_paths[frame_idx], cfg.work_dir)
            else:
                os.makedirs('./temp_visualize_results', exist_ok=True)
                visualize(pred_mask, image_paths[frame_idx], './temp_visualize_results')
    else:
        pass
    
    
    # CUDA_VISIBLE_DEVICES=0 python3.10 demo/demo_video.py --image_path /home/volume_shared/share_datasets/MeViS/MeViS_release/valid/JPEGImages/5a1d2cbd224c --text "<image>Identify the lamb moving forward then turning around."
    
    # relative
    # CUDA_VISIBLE_DEVICES=0 python3.10 demo/demo_video.py --image_path /tmp/all_vos2_relative/all_vos2_relative/sav_008378/ --text "<image>Locate the third item right of <box> 0.0667,0.3135,0.1771,0.6583 </box> in the video."
    
    # absolute
    # CUDA_VISIBLE_DEVICES=0 python3.10 demo/demo_video.py --image_path /tmp/all_vos2/train --text "<image>Locate the third item from left."
    
    # 给定具体名称
    # CUDA_VISIBLE_DEVICES=3 python3.10 demo/demo_video.py --image_path /home/rqshen/spatial_mllm/MOSELong_part/woman --text "<image>Identify and locate the woman in the video."
    
    # switch
    # CUDA_VISIBLE_DEVICES=3 python3.10 demo/demo_video.py --image_path /tmp/all_vos2/schoolgirls --text "<image>Can you track the leftmost object in this video?"
    # [0, 14, 32, 33, 38]
    
    # CUDA_VISIBLE_DEVICES=3 python3.10 demo/demo_video.py --image_path /tmp/all_vos2/sav_010976 --text "<image>Can you track the rightmost object in this video?"
    # [13, 16, 20, 31, 41]
    
    # CUDA_VISIBLE_DEVICES=3 python3.10 demo/demo_video.py --image_path /tmp/all_vos2/0442a954 --text "<image>Can you track the leftmost object in this video?"
    # [103, 105, 111, 116, 120]
    
    # CUDA_VISIBLE_DEVICES=3 python3.10 demo/demo_video.py --image_path /tmp/all_vos2/d7058a0d --text "<image>Can you track the leftmost object in this video?"
    
    # CUDA_VISIBLE_DEVICES=3 python3.10 demo/demo_video.py --image_path /tmp/all_vos2/da5e4bc5 --text "<image>Can you track the leftmost object in this video?"
    
    # CUDA_VISIBLE_DEVICES=3 python3.10 demo/demo_video.py --image_path /tmp/all_vos2/07353b2a89 --text "<image>Can you track the rightmost object in this video?"
    
    # CUDA_VISIBLE_DEVICES=3 python3.10 demo/demo_video.py --image_path /tmp/all_vos2/01ed275c6e --text "<image>Identify the giraffe behind the trees."
    
    
import argparse
import os

from PIL import Image, ImageDraw
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import cv2
try:
    from mmengine.visualization import Visualizer
except ImportError:
    Visualizer = None
    print("Warning: mmengine is not installed, visualization is disabled.")

import pdb
def parse_args():
    parser = argparse.ArgumentParser(description='Video Reasoning Segmentation')
    parser.add_argument('--image_folder', help='Path to image file')
    parser.add_argument('--model_path', default="work_dirs/hf_model")
    parser.add_argument('--work-dir', default='OUTPUT_DIR', help='The dir to save results.')
    parser.add_argument('--text', type=str, default="<image>Please describe the video content.")
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
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True
    )
    pdb.set_trace()

    image_files = []
    image_paths = []
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff"}
    for filename in sorted(list(os.listdir(cfg.image_folder))):
        if os.path.splitext(filename)[1].lower() in image_extensions:
            image_files.append(filename)
            image_paths.append(os.path.join(cfg.image_folder, filename))

    vid_frames = []
    for img_path in image_paths:
        img = Image.open(img_path).convert('RGB')
        vid_frames.append(img)

    if cfg.select > 0:
        img_frame = vid_frames[cfg.select - 1]
        print(f"Selected frame {cfg.select}")
        print(f"The input is:\n{cfg.text}")
        result = model.predict_forward(
            image=img_frame,
            text=cfg.text,
            tokenizer=tokenizer,
        )
    else:
        print(f"The input is:\n{cfg.text}")
        result = model.predict_forward(
            video=vid_frames,
            text=cfg.text,
            tokenizer=tokenizer,
        )

    prediction = result['prediction']
    print(f"The output is:\n{prediction}")
    pdb.set_trace()

    # if '[BOX]' in prediction and Visualizer is not None:
    #     box_coords = result['box_coords']
    #     for frame_idx in range(5):
    #         box_coord = box_coords[frame_idx].tolist()
    #         image = Image.open(image_paths[frame_idx]) 
    #         width, height = image.size
    #         print("width", width, "height", height)
    #         ymin, xmin, ymax, xmax = box_coord
    #         print("xmin", xmin, "ymin", ymin, "xmax", xmax, "ymax", ymax)
    #         draw = ImageDraw.Draw(image)
    #         draw.rectangle([xmin, ymin, xmax, ymax], outline="red", width=3) 
    #         image.save(f"image_with_bbox_{frame_idx}.jpg") 
    # else:
    #     pass

# CUDA_VISIBLE_DEVICES=0 python3.10 demo/demo_box.py --image_folder /home/volume_shared/share_datasets/MeViS/MeViS_release/train/JPEGImages/0535d91fd7b5 --text "<image>a trunk is moving forward in front."
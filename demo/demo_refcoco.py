import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0" 
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import pdb
import cv2
try:
    from mmengine.visualization import Visualizer
except ImportError:
    Visualizer = None
    print("Warning: mmengine is not installed, visualization is disabled.")
import re

def parse_args():
    parser = argparse.ArgumentParser(description='Video Reasoning Segmentation')
    parser.add_argument('--image_path')
    # parser.add_argument('--model_path', default="work_dirs/hf_model_refcoco_11000")
    parser.add_argument('--model_path', default="work_dirs/hf_model_spatial_refcoco_mevis_relative_absolute")
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
    ).to("cuda:0")

    tokenizer = AutoTokenizer.from_pretrained(model_path,
        trust_remote_code=True
    )
    
    bbox_results = {}
    error_log = []
    with open("finetune_refcoco_val.json.locout", "r") as file:
        lines = file.readlines()
    
    for i, line in enumerate(lines):
        image_path, grounding_info = line.split('<tab>')
        index = image_path.split('/')[-1]
        image_path = image_path.replace('[image]/path/to/COCO/', '/home/volume_shared/share_datasets/lisa_data/coco/')
        pure_text = re.search(r'<phrase>(.*?)</phrase>', grounding_info).group(1)
        text = "<image>Identify " + pure_text + " with normalized bounding box."

        image = Image.open(image_path).convert('RGB')
        W, H = image.size
        
        result = model.predict_forward(
            image=image,
            text=text,
            tokenizer=tokenizer,
        )
        prediction = result['prediction']
        
        try:
            box_match = re.search(r'<box>(.*?)</box>', prediction)
            assert box_match
            # coords = [float(x.strip()) for x in box_match.group(1).split(',')]
            # scaled_coords = [
            #     int(coords[0] * W),  # x1
            #     int(coords[1] * H),  # y1
            #     int(coords[2] * W),  # x2
            #     int(coords[3] * H)   # y2
            # ]
            # bbox_results[index] = scaled_coords
        
            box_content = box_match.group(1)
            if '<next>' in box_content:
                first_coords = box_content.split('<next>')[0].strip()
            else:
                first_coords = box_content.strip()
            coords = [float(x.strip()) for x in first_coords.split(',')]
            scaled_coords = [
                int(coords[0] * W),  # x1
                int(coords[1] * H),  # y1
                int(coords[2] * W),  # x2
                int(coords[3] * H)   # y2
            ]
            
            unique_key = f"{index}_{pure_text}"
            bbox_results[unique_key] = {
                'coords': scaled_coords,
                'sentence': pure_text
            }
            
            with open('refcoco_results/bbox_spatial_refcoco_mevis_relative_absolute.json', 'w') as f:
                json.dump(bbox_results, f, indent=2)
        
        except Exception as e:
            error_info = {
                "line_number": i + 1,
                "image_index": index,
            }
            error_log.append(error_info)
            print(f"Error processing line with image {index}: {e}")
            with open('error_log.json', 'w') as f:
                json.dump(error_log, f, indent=2)
        
            continue
        
# CUDA_VISIBLE_DEVICES=0 python3.10 demo/demo_refcoco.py --image_path orange.jpg --text "<image>Identify the food with most Vitamin C."
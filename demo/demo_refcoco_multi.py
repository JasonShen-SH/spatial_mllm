import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0" 
import torch
import json
import re
from PIL import Image
from tqdm import tqdm
import multiprocessing as mp
from functools import partial
import torch.multiprocessing as mp
import filelock
from transformers import AutoModelForCausalLM, AutoTokenizer
import pdb

def process_batch(lines, start_idx, model_path, bbox_file, error_file):
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map={"": 0},
        trust_remote_code=True
    ).to("cuda:0")
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True
    )

    bbox_lock = filelock.FileLock(f"{bbox_file}.lock")
    error_lock = filelock.FileLock(f"{error_file}.lock")

    for i, line in enumerate(lines):
        current_idx = start_idx + i
        
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
            assert box_match, "No box found in prediction"

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

            with bbox_lock:
                with open(bbox_file, 'r') as f:
                    bbox_results = json.load(f)
                
                unique_key = f"{index}_{current_idx}"
                bbox_results[unique_key] = {
                    'coords': scaled_coords,
                    'sentence': pure_text
                }
                
                with open(bbox_file, 'w') as f:
                    json.dump(bbox_results, f, indent=2)

        except Exception as e:
            print(e)
            with error_lock:
                with open(error_file, 'r') as f:
                    error_log = json.load(f)

                error_info = {
                    "line_number": current_idx + 1,
                    "image_index": index,
                }
                error_log.append(error_info)
                print(f"Error processing line with image {index}: {e}")
                
                with open(error_file, 'w') as f:
                    json.dump(error_log, f, indent=2)

def main():
    model_path = "work_dirs/hf_model_spatial_refcoco_mevis_relative_absolute"
    
    name = 'spatial_refcoco_mevis_relative_absolute_refcoco_testB.json'
    bbox_file = f'refcoco_results/{name}'
    error_file = f'refcoco_error/{name}'

    if not os.path.exists(bbox_file):
        with open(bbox_file, 'w') as f:
            json.dump({}, f)

    if not os.path.exists(error_file):
        with open(error_file, 'w') as f:
            json.dump([], f)

    lines = open("refcoco_val_test/finetune_refcoco_testB.json.locout", "r").readlines()

    num_processes = 2
    batch_size = len(lines) // num_processes
    batches = [lines[i:i + batch_size] for i in range(0, len(lines), batch_size)]

    mp.set_start_method('spawn', force=True)
    with mp.Pool(num_processes) as pool:
        process_func = partial(process_batch, 
                             model_path=model_path,
                             bbox_file=bbox_file,
                             error_file=error_file)
        
        list(tqdm(
            pool.starmap(process_func, 
                        [(batch, i * batch_size) for i, batch in enumerate(batches)]),
            total=len(batches)
        ))

if __name__ == "__main__":
    main()
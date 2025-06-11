import json
import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0" 
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer
import re
from tqdm import tqdm

def load_video_frames(frame_paths, base_path="/home/rqshen/ElysiumTrack-val500"):
    frames = []
    for frame_path in frame_paths:
        img = Image.open(os.path.join(base_path, frame_path)).convert('RGB')
        frames.append(img)
    return frames

def process_elysium(model, tokenizer, data):
    bbox_results = {}
    error_log = []
    
    for i, item in tqdm(enumerate(data)):
        try:
            video_id = item['video_id']
            caption = item['caption']
            frame_paths = item['frames']
            gt_boxes = item['box']
            
            # 每一个frame_path within frame_paths, 都要 os.path.join("/home/rqshen/ElysiumTrack-val500",frame_path)
            
            vid_frames = load_video_frames(frame_paths)
            assert len(vid_frames) == len(gt_boxes)
            
            W, H = vid_frames[0].size
            
            text = f"<image>Identify " + caption + " with normalized bounding box in the video."
            
            result = model.predict_forward(
                video=vid_frames,
                text=text,
                tokenizer=tokenizer,
            )
            prediction = result['prediction']
            sampled_frames = result['sampled_frames']
            sampled_gt_boxes = [gt_boxes[i] for i in sampled_frames]
            
            box_match = re.search(r'<box>(.*?)</box>', prediction)
            assert box_match, "No box found in prediction"
            box_content = box_match.group(1)
            all_coords = [coord.strip() for coord in box_content.split('<next>')]
            pred_boxes = []
            for coords_str in all_coords:
                coords = [float(x.strip()) for x in coords_str.split(',')]
                # scaled_coords = [
                #     int(coords[0] * W),  # x1
                #     int(coords[1] * H),  # y1
                #     int(coords[2] * W),  # x2
                #     int(coords[3] * H)   # y2
                # ]
                pred_boxes.append(coords)
    
            # single
            # box_match = re.search(r'<box>(.*?)</box>', prediction)
            # assert box_match
            # box_content = box_match.group(1)
            # if '<next>' in box_content:
            #     first_coords = box_content.split('<next>')[0].strip()
            # else:
            #     first_coords = box_content.strip()
                
            # coords = [float(x.strip()) for x in first_coords.split(',')]
            
            # scaled_coords = [
            #     int(coords[0] * W),  # x1
            #     int(coords[1] * H),  # y1
            #     int(coords[2] * W),  # x2
            #     int(coords[3] * H)   # y2
            # ]
            
            bbox_results[video_id] = {
                # 'coords': scaled_coords,
                'sentence': caption,
                'frame_paths': frame_paths,
                'num_frames': len(vid_frames),
                'sampled_frames': sampled_frames,
                'coords': pred_boxes,
                'sampled_gt_boxes': sampled_gt_boxes
            }
            
        except Exception as e:
            error_info = {
                "video_id": video_id,
                "prediction": prediction,
                "error_type": type(e).__name__,
                "error_message": str(e),
            }
            error_log.append(error_info)
            print(f"Error processing line with image {video_id}: {e}")
            with open('error_log.json', 'w') as f:
                json.dump(error_log, f, indent=2)
        
            continue
    
    return bbox_results, error_log

def main():
    model_path = "work_dirs/hf_model_spatial_refcoco_mevis_relative_absolute"
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
    
    data = []
    with open("/home/rqshen/ElysiumTrack-val500/ElysiumTrack-val500.json", "r") as f:
        for line in f:
            if line.strip():
                line_dict = json.loads(line)
                assert len(line_dict['frames']) == len(line_dict['box'])
                data.append({
                    "caption": line_dict["caption"],
                    "box": line_dict["box"],
                    "frames": line_dict['frames'],
                    "video_id": line_dict['vid']
                })
    
    bbox_results, error_log = process_elysium(model, tokenizer, data)
    
    with open('elysium_bbox_results.json', 'w') as f:
        json.dump(bbox_results, f, indent=2)
        
    with open('elysium_error_log.json', 'w') as f:
        json.dump(error_log, f, indent=2)
    
    print(f"Total videos: {len(data)}")
    print(f"Successfully processed: {len(bbox_results)}")
    print(f"Errors: {len(error_log)}")

if __name__ == "__main__":
    main()
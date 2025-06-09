#!/usr/bin/env python3
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from xtuner.model.utils import guess_load_checkpoint
import pdb
import torch.nn as nn
import argparse
import os
import shutil
from PIL import Image
import cv2
from modeling_sa2va_chat import predict_forward
from mmengine.visualization import Visualizer
import json

# CUDA_VISIBLE_DEVICES=0 python3.10 demo/merge_lora_inference.py

# ffmpeg -framerate 3 -pattern_type glob -i "temp_vis/*.jpg" -vf "setpts=N/(3*TB)" output.gif
# ffmpeg -framerate 3 -pattern_type glob -i "/mnt/data_hdd/rqshen/all_vos2/JPEGImages/22b13084b2/*.jpg" -vframes 41 -c:v libx264 -pix_fmt yuv420p output.mp4
# ffmpeg -framerate 3 -start_number 50 -pattern_type glob -i "/mnt/data_hdd/rqshen/all_vos2/JPEGImages/22b13084b2/*.jpg" -vframes 126 -c:v libx264 -pix_fmt yuv420p output.mp4
def visualize(pred_mask, image_path, work_dir):
    visualizer = Visualizer()
    img = cv2.imread(image_path)
    visualizer.set_image(img)
    visualizer.draw_binary_masks(pred_mask, colors='g', alphas=0.4)
    visual_result = visualizer.get_image()
    output_path = os.path.join(work_dir, os.path.basename(image_path))
    cv2.imwrite(output_path, visual_result)
    
def load_checkpoint_with_safe_globals(path):
    from torch.serialization import add_safe_globals
    from mmengine.logging.history_buffer import HistoryBuffer
    add_safe_globals([HistoryBuffer])
    return torch.load(path, map_location="cpu", weights_only=False)

# add new keys 
def set_param_with_dots(model, key, value):
    parts = key.split(".")
    current = model
    for part in parts[:-1]:
        if not hasattr(current, part):
            setattr(current, part, nn.Module())  # 创建中间模块
        current = getattr(current, part)
    setattr(current, parts[-1], nn.Parameter(value))  # 设置最终参数
        
def merge_weights(base_model, ft_state, sam2_state):
    # 1. LORA
    for i in range(len(base_model.language_model.model.layers)):       
        for proj in ["q_proj", "k_proj", "v_proj", "o_proj"]:
            orig_mod = getattr(base_model.language_model.model.layers[i].self_attn, proj)
            # ft_mod   = getattr(ft_state.language_model.layers[i].self_attn, proj)
            PREFIX = f"mllm.model.language_model.base_model.model.model.layers.{i}.self_attn.{proj}"
            # print(PREFIX)
            A = ft_state[f"{PREFIX}.lora_A.default.weight"].to(orig_mod.weight.device)   # [r, d_in], r=128, d_in=2048
            B = ft_state[f"{PREFIX}.lora_B.default.weight"].to(orig_mod.weight.device)   # [d_out, r], d_out=2048, r=128
            assert A.size(0) == 128
            r = A.size(0)
            alpha = 256
            scaling = alpha / r
            # W = W + scaling * B @ A
            orig_mod.weight.data += (B @ A) * scaling

        for proj in ["gate_proj", "up_proj", "down_proj"]:
            orig_mod = getattr(base_model.language_model.model.layers[i].mlp, proj)
            PREFIX = f"mllm.model.language_model.base_model.model.model.layers.{i}.mlp.{proj}"
            # print(PREFIX)
            A = ft_state[f"{PREFIX}.lora_A.default.weight"].to(orig_mod.weight.device)
            B = ft_state[f"{PREFIX}.lora_B.default.weight"].to(orig_mod.weight.device)
            assert A.size(0) == 128
            r = A.size(0)
            alpha = 256
            scaling = alpha / r
            orig_mod.weight.data += (B @ A) * scaling

    # 2. Full replacement
    # (2.1) mlp projection
    mlp1_sd = {k[len("mllm.model.mlp1."):]: v for k, v in ft_state.items() if k.startswith("mllm.model.mlp1.")}
    base_model.mlp1.load_state_dict(mlp1_sd) # projection
    
    # (2.2) language_model.embed_tokens
    new_embed_tokens_weight = ft_state["mllm.model.language_model.base_model.model.model.embed_tokens.weight"].to(base_model.language_model.model.embed_tokens.weight.device)
    new_embed_tokens = nn.Embedding(
        num_embeddings=new_embed_tokens_weight.shape[0], 
        embedding_dim=new_embed_tokens_weight.shape[1],
        _weight=new_embed_tokens_weight 
    )
    base_model.language_model.model.embed_tokens = new_embed_tokens

    # (2.3) language_model.lm_head
    new_lm_head_weight = ft_state["mllm.model.language_model.base_model.model.lm_head.weight"].to(base_model.language_model.lm_head.weight.device)
    new_lm_head = nn.Embedding(
        num_embeddings=new_lm_head_weight.shape[0], 
        embedding_dim=new_lm_head_weight.shape[1],
        _weight=new_lm_head_weight 
    )
    base_model.language_model.lm_head = new_lm_head
    # base_model.language_model.lm_head.weight = torch.nn.Parameter(new_lm_head)

    # 3. SAM2
    sam2_keys = [k for k in sam2_state.keys() if "sam_mask_decoder" not in k]
    for key in sam2_keys:
        # print(key)
        new_key = f"grounding_encoder.sam2_model.{key}" 
        set_param_with_dots(base_model, new_key, sam2_state.get(key))
    
    # print("########################################")
    # print("")
    
    grounding_encoder_keys = [k for k in ft_state.keys() if k.startswith('grounding_encoder')]
    for key in grounding_encoder_keys:
        # print(key)
        set_param_with_dots(base_model, key, ft_state.get(key))

    # 4. text_hidden_fcs (self-defined) 
    text_hidden_fcs_keys = [k for k in ft_state.keys() if k.startswith('text_hidden_fcs')]
    for key in text_hidden_fcs_keys:
        set_param_with_dots(base_model, key, ft_state.get(key))
    
    # pdb.set_trace()
    
    return base_model

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-model", required=False, default="OpenGVLab/InternVL2_5-8B")
        # "--base-model", required=False, default="ByteDance/Sa2VA-4B") 
    parser.add_argument(
        "--finetuned-dir", required=False, default="debug_8B/iter_6000.pth")
    parser.add_argument(
        "--sam2-model", required=False, default="/home/rqshen/Sa2VA/checkpoints/sam2.1_hiera_large.pt")
    parser.add_argument(
        "--out-dir", required=False, default="merged")
    args = parser.parse_args()

    print(">>> Loading base model:", args.base_model)
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model, trust_remote_code=True, device_map="auto", torch_dtype=torch.bfloat16,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    # special_tokens = ['[SEG]', '<p>', '</p>', '<vp>', '</vp>', '[RELATION]']
    special_tokens = ['[SEG]', '<p>', '</p>', '<vp>', '</vp>']
    tokenizer.add_tokens(special_tokens, special_tokens=True)
    # pdb.set_trace()
    # base_model.resize_token_embeddings(len(tokenizer))
    
    print(">>> Loading fine-tuned model:", args.finetuned_dir)
    ft_ckpt = load_checkpoint_with_safe_globals(args.finetuned_dir)
    ft_state = ft_ckpt["state_dict"]
    
    print(">>> Loading original SAM2 model:", args.sam2_model)
    sam2_ckpt = guess_load_checkpoint(args.sam2_model)
    sam2_state = sam2_ckpt["model"]

    print(">>> Merging weights …")
    mllm = merge_weights(base_model, ft_state, sam2_state)
    
    image_files = []
    image_paths = []
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff"}

    # image_folder = "/mnt/data_hdd/rqshen/MOSELong/JPEGImages/IMG_0450_38_122_obj1_duck"
    # test_video 是all_vos2中，符合left/right的那些视频(4524个)
    
    ###############################################################
    # Video Input
    video_id = "3c848ec5167d"
    image_folder = f"/home/volume_shared/share_datasets/MeViS/MeViS_release/train/JPEGImages/{video_id}"
    for filename in sorted(list(os.listdir(image_folder))):  
        if os.path.splitext(filename)[1].lower() in image_extensions:
            image_files.append(filename)
            image_paths.append(os.path.join(image_folder, filename))
    vid_frames = []
    for img_path in image_paths:
        img = Image.open(img_path).convert('RGB')
        vid_frames.append(img)
    
    data = json.load(open("/home/volume_shared/share_datasets/MeViS/MeViS_release/train/meta_expressions.json","r"))['videos']
    expression = data[video_id]['expressions']['0']['exp']
    # expression = json.load(open("all_vos2_step2.json","r"))['videos'][video_id]['expressions']['1']['exp']
    # bbox = json.load(open("all_vos2_step2.json","r"))['videos'][video_id]['expressions']['1']['reference_obj']['ref_box']
    # start = json.load(open("all_vos2_step2.json","r"))['videos'][video_id]['start']
    # end = json.load(open("all_vos2_step2.json","r"))['videos'][video_id]['end']
    # print("video_id: ", video_id)
    # text = f"<image>Can you segment the object that is to the {expression} of the box {bbox}?"
    text = f"<image>Can you segment the {expression}?"
    
    print(f"The input is:\n{text}")
    result = predict_forward(
        model=mllm,
        video=vid_frames,
        text=text,
        tokenizer=tokenizer,
        # start=start,
        # end=end,
    )
    
    ###############################################################
    # Image Input
    # image_path = "orange.jpg"
    # single_frame = Image.open(image_path).convert('RGB')
    # text = f"<image>Can you segment the fruit with most Vitamin C?"
    
    # print(f"The input is:\n{text}")
    # result = predict_forward(
    #     model=mllm,
    #     image=single_frame,
    #     text=text,
    #     tokenizer=tokenizer,
    # )
    
    ###############################################################
    # Visualization
    prediction = result['prediction']
    print(f"The output is:\n{prediction}")

    save_path = "temp_vis"
    if os.path.exists(save_path):
        shutil.rmtree(save_path)
    
    ###############################################################
    # Video input
    frame_indices = result['frame_indices']
    if '[SEG]' in prediction and Visualizer is not None:
        _seg_idx = 0
        pred_masks = result['prediction_masks'][_seg_idx]
        for idx, frame_idx in enumerate(frame_indices):
            pred_mask = pred_masks[idx]
            os.makedirs(save_path, exist_ok=True)
            visualize(pred_mask, image_paths[frame_idx], save_path)  
    else:
        pass
    
    ###############################################################
    # Image input
    # if '[SEG]' in prediction and Visualizer is not None:
    #     _seg_idx = 0
    #     pred_mask = result['prediction_masks'][_seg_idx][0] 
    #     os.makedirs(save_path, exist_ok=True)
    #     visualize(pred_mask, image_path, save_path)
    # else:
    #     pass
    

if __name__ == "__main__":
    main()
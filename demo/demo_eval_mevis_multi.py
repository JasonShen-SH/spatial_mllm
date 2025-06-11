import argparse
import os
import torch
import torch.multiprocessing as mp
from PIL import Image
# 确保你安装了 transformers, mmengine, opencv-python, Pillow, tqdm:
# pip install transformers mmengine opencv-python Pillow tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import cv2
# from mmengine.visualization import Visualizer # 如果不需要 mmengine 的 Visualizer，可以注释掉这行
import json
from tqdm import tqdm
import torch.distributed as dist
import numpy as np
# import pdb # 用于调试，最终版本可移除

# --- 分布式设置与清理 ---
# 修改 setup 函数签名，增加 gpu_ids 参数
def setup(rank, world_size, gpu_ids): # <--- 修改点1: 增加 gpu_ids 参数
    """
    初始化给定进程的分布式环境。
    设置当前进程的默认 CUDA 设备。
    """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355' # 确保这个端口没有被占用

    # 显式设置当前进程的默认 CUDA 设备
    current_gpu_id = gpu_ids[rank] # <--- 修改点2: 根据 rank 从 gpu_ids 中获取实际 GPU ID
    torch.cuda.set_device(current_gpu_id) # <--- 修改点3: 设置当前进程的默认 CUDA 设备

    # 初始化进程组
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    print(f"进程 {rank} 成功初始化，使用 GPU: {torch.cuda.current_device()} (CUDA ID: {current_gpu_id})") # <--- 修改点4: 打印更清晰的 GPU ID
    

def cleanup():
    """
    销毁分布式进程组。
    """
    dist.destroy_process_group()

# --- 参数解析函数 ---
def parse_args():
    """
    解析脚本的命令行参数。
    """
    parser = argparse.ArgumentParser(description='视频推理分割')
    parser.add_argument('--model_path', default="output_8B_continue", help='预训练模型路径')
    parser.add_argument('--work-dir', default='OUTPUT_MEVIS_2', help='输出结果的工作目录')
    parser.add_argument('--text', type=str, default="<image>请描述视频内容。",
                        help='视频描述的默认文本提示')
    parser.add_argument('--select', type=int, default=-1, help='选择一个特定视频进行处理（用于调试）')
    return parser.parse_args()

# --- 可视化函数：同时保存原始 mask 和渲染后的图像 ---
def visualize(pred_mask, video_name, exp_id, image_path, work_dir):
    # --- 保存原始二值 mask ---
    # 将布尔型 mask 转换为 uint8 类型 (0 或 255) 以便保存
    mask_to_save = pred_mask.astype(np.uint8) * 255
    base_mask_name = os.path.basename(image_path).replace('.jpg', '.png') # 通常 mask 保存为 PNG 格式
    output_mask_dir = os.path.join(work_dir, "masks", video_name, exp_id)
    os.makedirs(output_mask_dir, exist_ok=True) # 确保 masks 目录存在
    output_mask_path = os.path.join(output_mask_dir, base_mask_name)
    cv2.imwrite(output_mask_path, mask_to_save)
    # print(f"Mask saved to: {output_mask_path}") # 可以开启这行进行调试

    # --- 渲染并保存叠加图像 ---
    img = cv2.imread(image_path)
    if img is None:
        print(f"警告: 无法读取原始图像文件: {image_path}。跳过渲染保存。")
        return

    alpha = 0.4 # mask 的透明度
    color = (0, 255, 0) # 绿色 (OpenCV 使用 BGR 格式)

    # 检查 mask 形状是否与图像匹配
    if pred_mask.shape[:2] != img.shape[:2]:
        print(f"警告: mask 形状 {pred_mask.shape[:2]} 与图像形状 {img.shape[:2]} 不匹配，文件: {image_path}。将只保存原始图像。")
        visual_result = img # 如果形状不匹配，则只保存原始图像
    else:
        overlay = np.zeros_like(img, dtype=np.uint8)
        overlay[pred_mask] = color # 将 mask 为 True 的位置设置为指定颜色
        visual_result = cv2.addWeighted(img, 1 - alpha, overlay, alpha, 0) # 叠加图像

    if visual_result is None or visual_result.size == 0:
        print(f"警告: 渲染后的图像结果为空，文件: {image_path}。无法保存。")
        return

    # 渲染图像的保存路径
    output_rendered_dir = os.path.join(work_dir, video_name, exp_id)
    os.makedirs(output_rendered_dir, exist_ok=True) # 确保渲染图像的目录存在
    output_rendered_path = os.path.join(output_rendered_dir, os.path.basename(image_path)) # 保持原文件名
    cv2.imwrite(output_rendered_path, visual_result)
    # print(f"Rendered image saved to: {output_rendered_path}") # 可以开启这行进行调试


# --- 每个 GPU 进程的核心推理逻辑 ---
# 修改 process_video 函数签名，增加 gpu_ids 参数
def process_video(rank, world_size, cfg, gpu_ids): # <--- 修改点5: 增加 gpu_ids 参数
    """
    由每个派生的进程 (在特定的 GPU 上) 执行的函数。
    每个进程加载自己的模型并处理视频的子集。
    """
    setup(rank, world_size, gpu_ids) # <--- 修改点6: 将 gpu_ids 传递给 setup 函数
    
    # 确定当前进程的特定 GPU
    device = f"cuda:{gpu_ids[rank]}" # <--- 修改点7: 使用实际的 GPU ID 来构建设备字符串

    print(f"进程 {rank}: 正在加载模型到 {device}...")
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_path,
        torch_dtype="auto",
        trust_remote_code=True
    )
    # 关键：将模型移动到当前进程的正确 GPU
    model.to(device)

    # 可选：验证所有模型参数和缓冲区是否都在正确的设备上
    # 这对于调试可能很有帮助
    # print(f"\n进程 {rank} 模型组件设备信息:")
    # all_on_correct_device = True
    # for name, param in model.named_parameters():
    #     if param.device != torch.device(device):
    #         print(f"警告: 进程 {rank} 的参数 {name} 不在 {device}，而在 {param.device}。")
    #         all_on_correct_device = False
    # for name, buffer in model.named_buffers():
    #     if buffer.device != torch.device(device):
    #         print(f"警告: 进程 {rank} 的缓冲区 {name} 不在 {device}，而在 {buffer.device}。")
    #         all_on_correct_device = False
    # if all_on_correct_device:
    #     print(f"进程 {rank}: 所有模型参数和缓冲区都已成功移动到 {device}。")
    # else:
    #     print(f"进程 {rank}: 警告: 某些模型组件未能完全移动到 {device}。这可能是根本原因。")

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model_path,
        trust_remote_code=True
    )

    # 加载视频元数据 (假设它足够小，每个进程都可以加载)
    mevis_val = json.load(open("/home/volume_shared/share_datasets/MeViS/MeViS_release/valid/meta_expressions.json","r"))['videos']
    all_videos = list(mevis_val.keys())

    # 等待所有进程完成模型加载，然后才继续
    # 这确保了在其他进程准备好之前，没有进程尝试分配任务
    if rank == 0:
        print("所有进程已加载模型，正在同步以分配任务...")
    dist.barrier()

    # --- 任务分配：在进程之间划分视频 ---
    num_videos = len(all_videos)
    # 计算每个 GPU 应该处理多少视频
    # 将剩余的视频 (如果不能完美分配) 分配给靠前的 rank
    videos_per_gpu = num_videos // world_size
    remainder = num_videos % world_size

    # 调整不均匀分配的起始和结束索引
    # 前 'remainder' 个 GPU 各获得一个额外的视频
    if rank < remainder:
        start_idx = rank * (videos_per_gpu + 1)
        end_idx = start_idx + videos_per_gpu + 1
    else:
        start_idx = rank * videos_per_gpu + remainder
        end_idx = start_idx + videos_per_gpu

    my_videos = all_videos[start_idx:end_idx]
    print(f"进程 {rank}: 分配了 {len(my_videos)} 个视频，从索引 {start_idx} 到 {end_idx-1}。")


    # --- 处理分配到的视频 ---
    # 修改 tqdm 的 desc 参数，显示实际的 GPU ID
    for video_name in tqdm(my_videos, desc=f"GPU {gpu_ids[rank]} 正在处理视频"): # <--- 修改点8: 进度条显示实际 GPU ID
        video = mevis_val[video_name]
        video_path = os.path.join("/home/volume_shared/share_datasets/MeViS/MeViS_release/valid/JPEGImages", video_name)

        frames = sorted(list(os.listdir(video_path)))
        frames = [os.path.join(video_path, frame) for frame in frames]
        frame_rgbs = [Image.open(frame).convert('RGB') for frame in frames] # PIL 图像

        for exp_id, exp_data in video['expressions'].items():
            exp_name = f'<image>{exp_data["exp"]}'

            try:
                # 执行预测
                result = model.predict_forward(
                    video=frame_rgbs, # 输入是 PIL 图像列表
                    text=exp_name,
                    tokenizer=tokenizer,
                )

                prediction = result['prediction']

                # 确保 [SEG] 标签存在且有预测 mask
                if '[SEG]' in prediction and 'prediction_masks' in result and len(result['prediction_masks']) > 0:
                    _seg_idx = 0
                    pred_masks_raw = result['prediction_masks'][_seg_idx]

                    # 关键：确保预测 mask 在 CPU 上并转换为 NumPy 数组
                    pred_masks_cpu_np = []
                    for mask_data in pred_masks_raw:
                        if isinstance(mask_data, torch.Tensor):
                            # 移动到 CPU 并转换为 NumPy 数组 (使用 .contiguous() 确保内存连续性)
                            pred_masks_cpu_np.append(mask_data.cpu().contiguous().numpy())
                        elif isinstance(mask_data, np.ndarray): # 如果已经是 NumPy 数组
                            pred_masks_cpu_np.append(mask_data)
                        else:
                            # 处理其他潜在类型或报错
                            print(f"警告: 未预期的 mask 数据类型: {type(mask_data)}。跳过此 mask 的可视化。")
                            continue # 跳过当前 mask

                    for frame_idx in range(len(frames)):
                        # 将单帧的 CPU NumPy mask 传递给 visualize
                        # 确保 pred_masks_cpu_np 有足够的元素，如果预测只针对部分帧
                        if frame_idx < len(pred_masks_cpu_np):
                            pred_mask_for_viz = pred_masks_cpu_np[frame_idx]
                            # 在调用 visualize 前，确保所有必要的输出目录都已创建
                            # 注意：visualize 函数内部现在会创建 work_dir/masks 和 work_dir/video_name/exp_id 目录
                            # 所以这里只需要确保大的 work_dir/video_name/exp_id 存在即可，或者直接由 visualize 内部处理
                            
                            # 原来这里有两行 os.makedirs，现在 visualize 内部会处理，这里可以移除或仅保留顶级目录
                            # os.makedirs(os.path.join(cfg.work_dir, video_name, exp_id), exist_ok=True)
                            # os.makedirs(os.path.join(cfg.work_dir, "masks", video_name, exp_id), exist_ok=True)
                            
                            visualize(pred_mask_for_viz, video_name, exp_id, frames[frame_idx], cfg.work_dir)
                        else:
                            print(f"警告: 视频 {video_name}, 表达式 {exp_id}, 帧 {frame_idx} 没有可用的预测 mask。跳过可视化。")

            except RuntimeError as e:
                print(f"进程 {rank} 在处理视频 {video_name} (表达式: '{exp_data['exp']}') 时发生运行时错误: {e}")
                # 你可以选择记录此错误并继续，或者如果它是关键错误则重新抛出
                raise # 重新抛出错误以终止进程（如果不可恢复）

    cleanup() # 清理当前进程的分布式环境

# --- 主入口点 ---
if __name__ == "__main__":
    cfg = parse_args()

    # <--- 修改点9: 定义你想要使用的具体 GPU ID 列表
    target_gpu_ids = [4, 5, 6, 7]
    # <--- 修改点10: world_size 从列表中获取
    world_size = len(target_gpu_ids) 

    # <--- 修改点11: 检查指定的 GPU 是否实际存在
    for gpu_id in target_gpu_ids:
        if gpu_id >= torch.cuda.device_count():
            print(f"错误: CUDA 设备 {gpu_id} 不可用。系统只检测到 {torch.cuda.device_count()} 个 GPU。请检查您的配置。")
            exit()
    
    if world_size == 0:
        print("未找到 CUDA 设备或未指定目标设备。请确保您有 GPU 且 CUDA 已正确安装，并指定要使用的 GPU ID。")
        exit()

    os.makedirs(cfg.work_dir, exist_ok=True) # 确保主输出目录存在

    print(f"正在启动 {world_size} 个进程，使用 GPU: {target_gpu_ids} 进行分布式推理...")
    mp.spawn(
        process_video,
        args=(world_size, cfg, target_gpu_ids), # <--- 修改点12: 将 target_gpu_ids 传递给 spawn 启动的函数
        nprocs=world_size,
        join=True # `join=True` 等待所有派生的进程完成
    )
    print("所有进程已完成推理和可视化。")
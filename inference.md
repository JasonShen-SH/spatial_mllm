CUDA_VISIBLE_DEVICES=0 python demo/demo.py /mnt/data_hdd/rqshen/MOSELong/JPEGImages/IMG_0450_38_122_obj1_duck --model_path ByteDance/Sa2VA-8B --work-dir OUTPUT_DIR --text "<image>Please segment the second duck from right."

CUDA_VISIBLE_DEVICES=7 bash tools/dist.sh train projects/llava_sam2/configs/sa2va_4b_mevis.py 1 --cfg-options path='OpenGVLab/InternVL2_5-4B' --work-dir debug

# mine
CUDA_VISIBLE_DEVICES=7 bash tools/dist.sh train projects/llava_sam2/configs/sa2va_4b_relation.py 1 --cfg-options path='OpenGVLab/InternVL2_5-4B' --work-dir debug2

# 正常训练
CUDA_VISIBLE_DEVICES=1,2,3 bash tools/dist.sh train projects/llava_sam2/configs/sa2va_8b_box.py 3 --work-dir debug_refcoco

CUDA_VISIBLE_DEVICES=3 bash tools/dist.sh train projects/llava_sam2/configs/sa2va_8b_spatial.py 1 --work-dir debug_spatial_refcoco_mevis_refytvos_relative_absolute_31000_continue_with_switch_again

CUDA_VISIBLE_DEVICES=1,2,3 bash tools/dist.sh train projects/llava_sam2/configs/sa2va_8b_box.py 3 --work-dir debug_refcoco_mevis_refytvos

CUDA_VISIBLE_DEVICES=4,5,6,7 bash tools/dist.sh train projects/llava_sam2/configs/sa2va_8b_spatial.py 4 --work-dir debug_spatial_refcoco_mevis_refytvos_relative_absolute_latest_continue

# 调试 debug
CUDA_VISIBLE_DEVICES=3 bash tools/debug.sh train projects/llava_sam2/configs/sa2va_8b_spatial.py --work-dir debug5

# 推理
CUDA_VISIBLE_DEVICES=1 bash tools/dist.sh test projects/llava_sam2/configs/sa2va_4b_relation.py 1 --checkpoint all_vos_left_right/iter_3360.pth --work-dir inference

CUDA_VISIBLE_DEVICES=1 python demo/demo_self.py /mnt/data_hdd/rqshen/MOSELong/JPEGImages/IMG_0450_38_122_obj1_duck --work-dir OUTPUT_DIR --text "<image>Please segment the second duck from right."

# 调试
CUDA_VISIBLE_DEVICES=4 PYTHON=$(which python3.10) bash tools/debug.sh train projects/llava_sam2/configs/sa2va_8b_spatial.py --work-dir debug_box

# conver to HF
CUDA_VISIBLE_DEVICES=0 python3.10 projects/llava_sam2/hf/convert_to_hf.py projects/llava_sam2/configs/sa2va_4b_mevis.py --pth-model /home/rqshen/Sa2VA/debug_8B/iter_42000.pth --save-path output

# demo/demo
CUDA_VISIBLE_DEVICES=1 python3.10 demo/demo.py /home/volume_shared/share_datasets/MeViS/MeViS_release/valid/JPEGImages/6707bdc41381 --model_path output --work-dir OUTPUT_DIR --text "<image>Bear moving around in a circle"

CUDA_VISIBLE_DEVICES=0 python3.10 demo/demo_single_frame.py orange.jpg --model_path output --work-dir OUTPUT_DIR --text "<image>Are you happy?"

# 测试mevis
CUDA_VISIBLE_DEVICES=1 python3.10 demo/demo_eval_mevis.py

python3.10 demo/demo_eval_mevis_2.py

rsync -av --exclude='debug*' --exclude='archive*' --exclude='work_dir*' --exclude='temp_frames*' --exclude='final_masks*' ./ /home/rqshen/spatial_mllm_run

rsync -av --exclude='output*' --exclude='OUTPUT*' --exclude='8_queries*' --exclude='archive*' --exclude='temp*' --exclude='*.jpg' --exclude='shadow*' ./ /home/rqshen/spatial_mllm/

# !/bin/bash

echo "正在配置环境..."
conda create -n spatial python=3.10 -y
conda activate spatial
pip install -r requirements.txt

echo "正在安装git-lfs并下载数据集..."
sudo apt-get install git-lfs
git lfs install
git clone https://huggingface.co/datasets/Dense-World/Sa2VA-Training
cd Sa2VA-Training
git lfs pull

echo "创建数据存储目录..."
mkdir -p /home/volume_shared/share_datasets

echo "解压数据集并创建软链接..."
SOURCE_DIR=$(pwd)

unzip -n ref_seg_coco.zip
unzip -n ref_seg_coco_g.zip
unzip -n ref_seg_coco_plus.zip
unzip -n video_datas_mevis.zip
unzip -n video_datas_rvos.zip

ln -sf ${SOURCE_DIR}/ref_seg /home/volume_shared/share_datasets/
ln -sf ${SOURCE_DIR}/video_datas /home/volume_shared/share_datasets/

echo "请注意: 由于是需要验证的Onedrive链接, 您需要手动下载ReVOS数据集至/home/volume_shared/share_datasets/video_datas/ReVOS/, 下载路径为:"
echo "https://mailsjlueducn-my.sharepoint.com/personal/yancl9918_mails_jlu_edu_cn/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fyancl9918%5Fmails%5Fjlu%5Fedu%5Fcn%2FDocuments%2Fdataset%2Frevos%5Feccv%5Fdataset%2FReVOS"

# echo "请确保安装README.md的指示, 修改projects/llava_sam2/configs/sa2va_8b_spatial.py中的数据路径配置"

echo "请确保从百度网盘下载空间位置关系数据集, 并存储在/home/volume_shared/share_datasets/video_datas/all_vos2/"

echo "准备工作完成后，使用以下命令开始训练："
echo "CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash tools/dist.sh train projects/llava_sam2/configs/sa2va_8b_spatial.py 8 --work-dir spatial_mllm_save"
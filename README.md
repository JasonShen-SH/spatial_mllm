# 环境准备
```shell
基本要求
- Python 3.10
- PyTorch 2.6.0
- CUDA 12.4

推荐使用conda创建独立环境, 名为spatial
conda create -n spatial python=3.10 -y 
conda activate spatial
pip install -r requirements.txt
```

# 数据准备

下载命令如下

```shell
sudo apt-get install git-lfs
git lfs install
git clone https://huggingface.co/datasets/Dense-World/Sa2VA-Training

# 应该会自动开始下载大文件, 如果没有, 手动执行
cd Sa2VA-Training
git lfs pull

# 下载后的存储路径可更改, 本例为: /home/volume_shared/share_datasets
# 也可以先下载到当前路径，然后软链接到 /home/volume_shared/share_datasets (run.sh按照第二种方式执行)
```



下载完成后, 只需要解压其中的
```shell
ref_seg_coco.zip, 
ref_seg_coco_g.zip，
ref_seg_coco_+.zip，
video_datas_mevis.zip，
video_datas_rvos.zip 
其余Image Video QA数据集暂时不用
```

注意，由于ReVOS在上述链接中不完整，需要手动下载
[ReVOS](https://mailsjlueducn-my.sharepoint.com/personal/yancl9918_mails_jlu_edu_cn/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fyancl9918%5Fmails%5Fjlu%5Fedu%5Fcn%2FDocuments%2Fdataset%2Frevos%5Feccv%5Fdataset%2FReVOS&ga=1)，
并与上述文件放置在同一位置

除此之外，空间位置关系数据集存储路径为all_vos2, 我将其打包上传到了百度网盘

最终期待的数据结构为:
```shell
/home/volume_shared/share_datasets/
├── ref_seg/
│   ├── refcoco/
│   ├── refcocog/
│   └── refcoco+/
└── video_datas/
    ├── mevis/
    └── rvos/
    └── ReVOS/ # 手动下载
    └── all_vos2/ # 手动下载
```

若修改了存放路径，请在配置文件 projects/llava_sam2/configs/sa2va_8b_spatial.py 中同步修改


#  训练
```shell
# 8卡训练

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash tools/dist.sh train projects/llava_sam2/configs/sa2va_8b_spatial.py 8 --work-dir spatial_mllm_save

# checkpoint存储路径在spatial_mllm_save
```
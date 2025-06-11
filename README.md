本说明针对 同时使用 box + point 来做监督的情况

**环境准备** 和 **数据准备** 是相同的，请参见box的README部分

# 一个额外下载

需要手动下载Ref-YoutubeVOS的[mask文件](https://drive.google.com/file/d/1zjfC9EzoZD9E4rk7a5G5VHrrD0zzJTXE/view?usp=sharing), 直接放置在home目录即可

若修改了存放路径，请在配置文件 projects/llava_sam2/configs/sa2va_8b_point.py 中同步修改


#  训练
```shell
# 8卡训练, 注意配置文件有变化

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash tools/dist.sh train projects/llava_sam2/configs/sa2va_8b_point.py 8 --work-dir spatial_mllm_with_point_save

# checkpoint存储路径在spatial_mllm_with_point_save
```
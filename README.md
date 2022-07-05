# Unsupervised Homography Estimation with Coplanarity-Aware GAN [[PDF](https://openaccess.thecvf.com/content/CVPR2022/html/Hong_Unsupervised_Homography_Estimation_With_Coplanarity-Aware_GAN_CVPR_2022_paper.html)]


## Summary
<p align="center">
<img src=https://github.com/megvii-research/HomoGAN/blob/main/images/slide.png width="891px" height=500px">
</p>

## Presentation Video:
[[Bilibili](https://www.bilibili.com/video/BV1Wv4y137Ko?spm_id_from=333.999.0.0&vd_source=0a9f26f2f6a274787d7c263fe3ce7f3d)] [[Youtube](https://www.youtube.com/watch?v=uNFA-yOSz7M)]


## Requirements
```
Pytorch = 1.7.1
Python = 3.7.10
Cuda = 10.2
Numpy = 1.19.4
```
You can also use this command
```
pip install -r requirements.txt
```

## Download the Deep Homography Dataset

Please refer to [Content-Aware Unsupervised Deep Homography Estimation.](https://github.com/JirongZhang/DeepHomography).

- Download raw dataset
```
# GoogleDriver
https://drive.google.com/file/d/19d2ylBUPcMQBb_MNBBGl9rCAS7SU-oGm/view?usp=sharing
# BaiduYun
https://pan.baidu.com/s/1Dkmz4MEzMtBx-T7nG0ORqA (key: gvor)
```
- Unzip the data to directory "./dataset"

- Run "video2img.py"
```
Be sure to scale the image to (640, 360) since the point coordinate system is based on the (640, 360).
e.g. img = cv2.imresize(img, (640, 360))
```

## How to test?
```
python evaluate.py --model_dir ./experiments/HomoGAN/ --restore_file xxx.pth
```
## How to train?
You need to modify ```./dataset/data_loader.py``` slightly for your owner environment, and you can also refer to [Content-Aware Unsupervised Deep Homography Estimation](https://github.com/JirongZhang/DeepHomography).

## Pre-training:
```
1) set "pretrain_phase" in ./experiments/HomoGAN/params.json as True
2) python train.py --model_dir ./experiments/HomoGAN/
```
## Fine-tuning:
```
1) set "pretrain_phase" in ./experiments/HomoGAN/params.json as False
2) python train.py --model_dir ./experiments/HomoGAN/ --restore_file xxx.pth
```
## Code coming soon ...

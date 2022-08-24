# [CVPR2022] Unsupervised Homography Estimation with Coplanarity-Aware GAN 

<h4 align="center">Mingbo Hong$^{1,2}$, Yuhang Lu${1,3}$, Nianjin Ye$^1$, Chunyu Lin$^4$, Qijun Zhao$^2$, Shuaicheng Liu$^5$</center>
<h4 align="center">1. Megvii Technology, 2. Sichuan University, 3. Univesity of South Carolina</center>
<h4 align="center">4. Beijing Jiaotong University, 5. University of Electronic Science and Technology of China</center>


#### This is the official implementation of HomoGAN, CVPR2022, [[PDF](https://openaccess.thecvf.com/content/CVPR2022/html/Hong_Unsupervised_Homography_Estimation_With_Coplanarity-Aware_GAN_CVPR_2022_paper.html)]

## Presentation Video:
[[Bilibili](https://www.bilibili.com/video/BV1Wv4y137Ko?spm_id_from=333.999.0.0&vd_source=0a9f26f2f6a274787d7c263fe3ce7f3d)] [[Youtube](https://www.youtube.com/watch?v=uNFA-yOSz7M)]

## Summary
<p align="center">
<img src=https://github.com/megvii-research/HomoGAN/blob/main/images/slide.png width="780px" height=430px">
</p>

## Pipeline
![pipeline](https://user-images.githubusercontent.com/1344482/180948412-04138e09-b1da-4bf4-bc7c-c96432dff70a.JPG)



## Dependencies
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
## Pre-trained model
```
The models provided below are the retrained version(with minor differences in quantitative results)
```

| model    | RE | LT |LL |SF |LF |Avg |Model |
| --------- | ----------- | ------------ |------------ |------------ |------------ |------------ |------------ |
| Pre-trained    | 0.24       | 0.47       |0.59       |0.62       |0.43      |0.47       |Baidu [Google](https://drive.google.com/file/d/1IwxsB7A9VHofX3l7wbfJ-vn_gLS9Z1H_/view?usp=sharing)       |
| Fine-tuning | 0.22       | 0.38        |0.57       |0.47       |0.30       |0.39       |Baidu [Google](https://drive.google.com/file/d/1U-ZvVNMWO2yfZLwmkS6wva0Xi7R2UKpL/view?usp=sharing)       |

## How to test?
```
python evaluate.py --model_dir ./experiments/HomoGAN/ --restore_file xxx.pth
```
## How to train?
You need to modify ```./dataset/data_loader.py``` slightly for your environment, and you can also refer to [Content-Aware Unsupervised Deep Homography Estimation](https://github.com/JirongZhang/DeepHomography).

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

## Citation
If you use this code or ideas from the paper for your research, please cite our paper:
```
@InProceedings{Hong_2022_CVPR,
    author    = {Hong, Mingbo and Lu, Yuhang and Ye, Nianjin and Lin, Chunyu and Zhao, Qijun and Liu, Shuaicheng},
    title     = {Unsupervised Homography Estimation With Coplanarity-Aware GAN},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2022},
    pages     = {17663-17672}
}
```

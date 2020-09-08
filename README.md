# YOLOv2: Real-time 3D Object Detection from Point Clouds
## 简介

这是YOLOv2的非官方实现，可以直接使用51Sim-One数据集进行训练和测试。

在这个版本的实现中，主要在网络结构上有些不同，在这里使用resnet-22的网络结构。

## 环境配置

```
opencv
path
tensorlfow >= 1.14
easydict
numpy
python
```

## 训练与测试

1. 下载数据集，并解压至以下目录结构

   ```shell
   ├── train
       ├── scene0
           ├── DumpSettings.json
           ├── image
           ├── image_label
       ...
   ├── test
       ├── scene1
           ├── DumpSettings.json
           ├── image
       ...
   ```

2. 运行训练脚本

   ```python
   python train.py
   ```

   

3. 运行测试脚本

   ```python
   python predict.py 
   ```

## Credit

```
YOLO9000: Better, Faster, Stronger
```


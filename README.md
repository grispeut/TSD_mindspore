# 赛题介绍
第三届华为云无人车挑战杯复赛Top1方案分享。本届[无人车挑战杯](https://competition.huaweicloud.com/information/1000041539/introduction)大赛主要考核点有交通信号灯识别、车道线检测、斑马线检测、限速标志识别、施工标志识别、障碍物检测等，其中交通信号灯、斑马线、限速标志检测算法需要基于AI开发平台ModelArts开发。训练数据集包含红灯、绿灯、黄灯、人行横道、限速标志、解除限速标志六种类型图片，需使用ModelArts数据管理模块完成以上六种检测目标的标注。参赛者需基于MindSpore框架（使用其他框架提交的作品无效）建立目标检测模型

# 解决方案及算法介绍
+ 数据集: [初赛数据](https://marketplace.huaweicloud.com/markets/aihub/datasets/detail/?content_id=93d35831-c084-4003-b175-4280ef289379)和[复赛数据](https://marketplace.huaweicloud.com/markets/aihub/notebook/detail/?id=0fbf9486-9e71-41f0-9295-3d75b68b15db)
+ 数据增强：[albu](https://github.com/albumentations-team/albumentations)和[imagecorruptions](https://github.com/bethgelab/imagecorruptions)
+ 后处理： tta+wbf, 使用[wbf](https://github.com/ZFTurbo/Weighted-Boxes-Fusion)进行多尺度集成，wbf应该是目前性能最好后处理方法，优于nms, soft-nms, nmw
+ 检测模型：[YoloV4](https://gitee.com/ascend/modelzoo/tree/master/built-in/MindSpore/Official/cv/detection/YOLOv4_Cspdarknet53_for_MindSpore), 在本次比赛中我们集成了两个yolov4模型，一个使用albu增广，一个使用albu+imagecorruptions增广，可以提升方案的鲁棒性

# 环境配置
## 环境依赖
+ cuda 10.1
+ cudnn 7.6.4
+ gcc 7.3.0
+ python 3.7
+ mindspore 1.3.0
## 训练设备
4张1080ti，batch_size(6x4)

# 模型训练复现流程
## 数据集准备
训练集标签放在`data/annotations_xml`下，训练集图片放在`data/train`下

## 将voc格式的标签转为coco格式
```
mkdir -p data/annotations
python pascal2coco.py
```

## 预训练模型下载
[coco预训练模型](https://mindspore.cn/resources/hub/details?MindSpore/ascend/1.1/yolov4_v1.1)

## 模型训练
```
sh train.sh
```

## 推理
测试图片放在samples文件夹下，推理结果在outputs文件夹下，该脚本可以本地运行，也可以直接用于部署Modelarts在线服务和批量服务
```
python customize_service.py
```

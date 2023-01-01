### 实验环境
---
依赖已生成，见requirements.txt


### 数据集下载
---
数据集采用Kaggle上的Bitmoji数据集：

https://www.kaggle.com/datasets/romaingraux/bitmojis

### 运行方式
---
在VIT + Hybrid VIT文件夹中，训练入口为train.py，也可以通过args设置参数，三个模型在VIT中都已经完成了，直接在train.py的主函数中更改模型即可。

在mobile_vit 文件夹中，先添加数据集到文件夹中，然后按照类别进行分类，最后在train.py中修改一下数据集路径即可，如果要使用预训练的模型的话，放入pretrained，修改路径即可，直接运行train.py即可训练。

### 实验结果
---

详细的实验结果见文件夹“实验结果”


2022.12.31 神经网络课设

参考：
“AN IMAGE IS WORTH 16X16 WORDS:TRANSFORMERS FOR IMAGE RECOGNITION AT SCALE”
“Attention Is All You Need ”
“MobileViT: Light-Weight, General-Purpose, and Mobile-Friendly Vision Transformer”
  paddle paddle：【实践】深入理解图像分类中的Transformer-Vit
  paddle paddle：Mobile-ViT：改进的一种更小更轻精度更高的模型
  CSDN：ECharts模板使用教程
  CSDN：Vision Transformer详解
  bilibili：霹雳啪啦Wz：11.2 使用pytorch搭建Vision Transformer(vit)模型
  Github：chinhsuanwu/mobilevit-pytorch<Public>
  Github：rwightman/pytorch-image-models<Public>
  Github：Polarosjame/Hybrid_ViT_周凌峰
  Github：WZMIAOMIAO/deep-learning-for-image-processing<Public>
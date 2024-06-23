# 神经网络与深度学习课程设计报告

## 实验名称
服饰图像描述生成

## 实验概述
本次课程设计的目标是开发一个能够自动识别服饰图像中的关键特征，并生成对应的自然语言描述的模型。该模型需要理解图像中的视觉内容，例如服饰的种类、颜色、款式和材质，并根据这些特征产生准确、流畅且语义丰富的文字描述。

## 实验环境
- **硬件环境**：Intel(R) UHD Graphics, NVIDIA GeForce MX350, 16GB 内存
- **软件环境**：Python 3.9.16, PyTorch 1.8.2, torchvision 0.3.0, numpy 1.26.2, pandas 1.2.4, matplotlib 3.5.1, seaborn 0.13.0, tqdm 4.66.1, nltk 3.8.1, gensim 4.3.2
- **开发环境**：Windows 10, Jupyter Notebook, Visual Studio Code

## 实验数据集
- **DeepFashion-MultiModal数据集**：一个专门为服饰图像识别和描述而设计的大型数据集，包含丰富的服饰图像和相应的文本描述。

## 实验方法
- **CNN+GRU**：使用卷积神经网络提取图像特征，然后利用门控循环单元生成描述文本。
- **ResNet+Transformer解码器**：采用预训练的ResNet模型提取特征，再使用Transformer模型进行编码解码，生成描述文本。

## 实验步骤
1. 数据预处理：包括分词、构建词汇表、创建键值映射等。
2. 模型构建：定义图像编码器和文本解码器的网络结构。
3. 模型训练：使用训练数据训练模型，记录损失值并调整模型参数。
4. 模型评估：采用METEOR、ROUGE-L和BLEU等指标评估生成文本的质量。

# ARCTIC图像描述模型实现

## 简介
ARCTIC（Attention-based Image Description Model）是一个基于注意力机制的图像描述生成模型。本项目实现了一个图像编码器和文本解码器，编码器使用CNN网格表示提取器，解码器使用RNN结合注意力机制，以生成给定图像的描述。

## 数据集
- 使用的数据集为Flickr8k，包含8000张图片，每张图片对应5个句子描述。
- 数据集下载请访问[Kaggle Flickr8k](https://www.kaggle.com/adityajn105/flickr8k)。
- 数据集划分采用Karpathy提供的方法，分为训练集、验证集和测试集。

## 环境要求
- Python 3.x
- PyTorch 1.x
- torchvision
- numpy
- pillow
- matplotlib
- nltk（用于计算BLEU分数）

## 模型结构
- **图像编码器**：使用ResNet-101作为图像编码器，提取图像特征。
- **文本解码器**：使用GRU结合加性注意力机制生成描述。

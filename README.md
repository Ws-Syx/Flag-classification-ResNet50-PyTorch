# Flag-Classification

The project is based on the PyTorch framework and uses the open source ResNet 50 part of the code to a certain extent. (I did not make too many modifications and changes to the original ResNet 50 part of the code, and the original author's comments have been fully retained.)

This project provides a data set and a detection model that I trained on Titan XP.

(When I recall which dalao code I adopted, I will indicate the source of reference here and express my gratitude)

## 1 Introduction

### 1.1 Supported flag categories
This project provides a data set and a model trained based on this data set.
The project can complete the classification of 64 national flags, covering the major countries in the world, and at the same time select some countries from the beginning based on the lexicographical list of countries.

(The project has a certain degree of scalability, and more developers are welcome to improve the data set)

The following is a list of countries covered in this project:

![avatar](./readme/classes.png)

### 1.2 Data set

The training set has a total of 46 categories, each country has 30 national flag pictures, and the pictures are collected from the Internet. Because individual countries are too small and the source of national flags is scarce, in consideration of a balanced sample, all countries have kept 30.
The pictures of the surplus were included in the test set.

- Training set

Link: https://pan.baidu.com/s/1MFo9bCe_CoZ5WSl6iUu1og

Extraction code: o6u0

- Test set

Link: https://pan.baidu.com/s/1k13flpGrmdhEvPjFxztY-g

Extraction code: 8okd


## 2 How to start

### 2.1 Data set preparation

The structure of the data set is very simple. We only need to put the pictures of each country in the corresponding folder in the form of the above figure. The naming of the pictures is unlimited. The suffix is ​​png, and the name of the folder is the name of the picture. label.

Put the prepared national flag folders in the ".\train_dataset\" directory of the project.

### 2.2 Training

Run "1.train.py" to start the training of the model.

PS: The model I trained has been uploaded to Baidu Netdisk, or you can download it directly and put it in the model folder

Link: https://pan.baidu.com/s/1X52NpARBa2H84yKXli_faA

Extraction code: ssin

### 2.4 Use the trained model

To use the trained model to make predictions, you can refer to a small demo written by me "2.predict.py",
The file structure of the test set in the demo must be the same as the training set, so that the label and the index can have a corresponding relationship.

For further application of this project, I suggest to build a dictionary (dict) to store the correspondence between each label and index during training, so as to facilitate the prediction of labels based on the output results.

# Flag-Classification 国旗分类

该项目基于PyTorch框架，在一定程度上使用了开源的ResNet 50部分代码。(本人并未对原有的ResNet 50部分的代码做过多的修饰与更改，原作者的注释得到了充分的保留。)

本项目提供了数据集和本人在Titan XP上训练完成的检测模型。

(当我回忆起我采用了哪位dalao的代码以后，我会在此注明引用来源，并表示感谢)

## 1 简介

### 1.1 支持的国旗类别
本项目提供了数据集，并提供了基于本数据集训练完成的模型。
该项目可以完成对64种国旗的分类，涵盖了世界上主要国家，同时根据基于字典序的国家列表从头选择了一部分国家。

（该项目具有一定的可扩展性，欢迎更多的开发者完善数据集）

以下是本项目中所涵盖的国家列表：

![avatar](./readme/classes.png)

### 1.2 数据集

训练集共有46个类别，每个国家有30张国旗图片，图片搜集自网络。由于个别国家过于小众，国旗来源稀缺，故处于平衡样本的考虑，所有国家均保留了30张。
盈余的图片被划入测试集。

- 训练集

链接：https://pan.baidu.com/s/1MFo9bCe_CoZ5WSl6iUu1og 

提取码：o6u0
- 测试集

链接：https://pan.baidu.com/s/1k13flpGrmdhEvPjFxztY-g 

提取码：8okd


## 2 如何启动

### 2.1 数据集的准备

数据集的结构十分简单，我们只需按照上图的形式将各个国家的图片放入对应的文件夹中即可，图片的命名无限制，以png为后缀，所在的文件夹名即该图片的label。

将准备好的各国国旗文件夹放到项目的".\train_dataset\"目录下。

### 2.2 训练

运行“1.train.py”，开始模型的训练。

PS：本人训练完成的模型已经上传到了百度网盘，也可以直接下载下来放到model文件夹

链接：https://pan.baidu.com/s/1X52NpARBa2H84yKXli_faA 

提取码：ssin

### 2.4 使用训练完成的模型

要使用训练完成的模型进行预测，可以参考本人写的一个小demo “2.predict.py”，
该demo里测试集的文件家结构要同训练集一致，这样label和索引才能起到对应关系。

若要对本项目进行进一步的应用，本人建议在训练时建立一个字典(dict)来存储各个标签与索引之间的对应关系，以方便根据输出结果得到预测标签。

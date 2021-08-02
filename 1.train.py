import torch
import torch.nn as nn
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as plt2
import os

from utils.config import config

from resnet50 import ResNet50

from utils.FlagData import data_split, FlagData


# Function to show CIFAR images
def show_data(image):
    plt.imshow(np.transpose(image[0], (1, 2, 0)), interpolation='bicubic')
    plt.show()


if __name__ == '__main__':
    # -------------------------------------------------------(0)device-----------------------------------
    # Check GPU, connect to it if it is available
    device = ''
    if torch.cuda.is_available():
        device = 'cuda'
        print('\t+-------------------------+')
        print("\t|     Running on GPU      |")
        print('\t+-------------------------+\n')
    else:
        device = 'cpu'
        print('\t+-------------------------+')
        print("\t|     Running on CPU      |")
        print('\t+-------------------------+\n')

    # -------------------------------------------------------(1)train_dataset-------------------------------------
    print("==> Prepairing train_dataset ...")

    trans = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    # 分割训练集、测试集
    train_dataset_path, test_dataset_path = data_split('train_dataset', config["training_rate"])

    # dataset
    train_dataset = FlagData(train_dataset_path, trans)
    valid_dataset = FlagData(test_dataset_path, trans)

    # dataloader
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True,
                                               num_workers=2,
                                               drop_last=False)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=config["batch_size"], shuffle=True,
                                               num_workers=2, drop_last=False)

    print('load successfully')

    # -------------------------------------------------------(2)model------------------------------------
    # set number of classes
    num_classes = config["num_classes"]
    model = ResNet50(num_classes)
    model = model.to(device)
    model.train()

    # -------------------------------------------------------(3)optimizer & loss---------------------------
    optimizer = optim.SGD(model.parameters(), lr=config["learning_rate"], momentum=0.9, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()

    length_train = len(train_dataset)
    length_test = len(valid_dataset)

    # -------------------------------------------------------(4)train-------------------------------------
    BEST_ACCURACY = 0

    # Start training
    epochs = config["epoch"]
    result_record = {'Train Loss': [], 'Train Acc': [], 'Validation Loss': [], 'Validation Acc': []}

    for epoch in range(epochs):
        print("\nEpoch:", epoch + 1, "/", epochs)

        # ----------------------------训练集-----------------------------
        cost = 0
        correct = 0

        for i, (x, y) in enumerate(train_loader):
            model.train()
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            # 预测
            yhat = model(x)
            yhat = yhat.reshape(-1, num_classes)
            # 计算损失
            loss = criterion(yhat, y)
            # 反向传播
            loss.backward()
            # 梯度下降
            optimizer.step()
            # 累计当前batch的损失至总损失之中
            cost += loss.item()
            # yhat2是每行(每一行，代表了一次预测)最大值的索引
            _, yhat2 = torch.max(yhat.data, 1)
            # 当前batch中成功匹配的个数
            correct += (yhat2 == y).sum().item()

        # 计算每张图的平均损失
        my_loss = cost / len(train_loader)
        # 计算准确度
        my_accuracy = 100 * correct / length_train

        result_record['Train Loss'].append(my_loss)
        result_record['Train Acc'].append(my_accuracy)
        print('Train Accuracy:{}%\tTain Loss:{}'.format(my_accuracy, my_loss))

        # ----------------------------测试集-----------------------------
        cost = 0
        correct = 0

        with torch.no_grad():
            for x, y in valid_loader:
                x, y = x.to(device), y.to(device)
                model.eval()
                # 预测
                yhat = model(x)
                yhat = yhat.reshape(-1, num_classes)
                # 计算loss
                loss = criterion(yhat, y)
                # 累计loss
                cost += loss.item()
                # yhat2是每行(每一行，代表了一次预测)最大值的索引
                _, yhat2 = torch.max(yhat.data, 1)
                # 当前batch中成功匹配的个数
                correct += (yhat2 == y).sum().item()

        my_loss = cost / len(valid_loader)
        my_accuracy = 100 * correct / length_test

        result_record['Validation Loss'].append(my_loss)
        result_record['Validation Acc'].append(my_accuracy)

        print('Valid Accuracy:{}%\tValid Loss:{}'.format(my_accuracy, my_loss))

        # Save the model if you get best accuracy on validation train_dataset
        if my_accuracy > BEST_ACCURACY:
            BEST_ACCURACY = my_accuracy
            print('* Saving the model *')
            model.eval()
            if not os.path.isdir('model'):
                os.mkdir('model')

            model_name = './model/' + config["model_name"] + '.pth'
            torch.save(model.state_dict(), model_name)

    print("TRAINING IS FINISHED !!!")
    results = result_record

    # -------------------------------------------------------(5)plot--------------------------------------
    plt.figure(1)
    plt.plot(results['Train Loss'], 'b', label='training loss')
    plt.plot(results['Validation Loss'], 'r', label='validation loss')
    plt.title("LOSS")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend(['training set', 'validation set'], loc='center right')
    plt.savefig('Loss_ResNet50.png', dpi=300, bbox_inches='tight')

    plt.figure(2)
    plt.plot(results['Train Acc'], 'b', label='training accuracy')
    plt.plot(results['Validation Acc'], 'r', label='validation accuracy')
    plt.title("ACCURACY")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend(['training set', 'validation set'], loc='center right')
    plt.savefig('Accuracy_ResNet50.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

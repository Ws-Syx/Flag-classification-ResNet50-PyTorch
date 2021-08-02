import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as plt2
import os

from utils.config import config

from resnet50 import ResNet50


def get_max_index(yhat):
    prediction = -1
    current_max = -1
    for i in range(num_classes):
        if yhat[0, i] > current_max:
            current_max = yhat[0, i]
            prediction = i
    return prediction


def test(show_error):
    acc = 0
    sum = 0
    model.eval()
    with torch.no_grad():
        for x, y in validation_loader:
            x, y = x.to(device), y.to(device)
            model.eval()
            yhat = model(x)
            yhat = yhat.reshape(-1, num_classes)

            prediction = get_max_index(yhat)

            sum = sum + 1
            if prediction == y[0]:
                acc = acc + 1
            else:
                if show_error:
                    # show images with title
                    res = list(dictionary.keys())[list(dictionary.values()).index(prediction)]
                    real = list(dictionary.keys())[list(dictionary.values()).index(y[0])]
                    show_data(x, 'prediction:' + res + '\nreal:' + real)

    print('accuary = {}%'.format(100 * acc / sum))
    print('sum = {}'.format(sum))
    print('error_num = {}'.format(sum - acc))


# Function to show CIFAR images
def show_data(image_gpu, prediction):
    image = image_gpu
    image = image.cpu()
    plt.imshow(np.transpose(image[0], (1, 2, 0)), interpolation='bicubic')
    plt.title(prediction)
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

    # --------------------------------------------------(1)load test_dataset-------------------------------------
    trans = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    test_data = torchvision.datasets.ImageFolder(root='./test_dataset', transform=trans)
    dictionary = test_data.class_to_idx

    # Put train_dataset into loader, specify batch_size
    validation_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=True, num_workers=2,
                                                    drop_last=False)

    # --------------------------------------------------(2)load the model-------------------------------
    # 建立ResNet50模型
    num_classes = config["num_classes"]
    model = ResNet50(num_classes)
    model = model.to(device)
    # 加载训练完成的模型权重文件
    assert os.path.isdir('model'), 'Error: no model directory found!'
    model.load_state_dict(torch.load('./model/ver1.1.pth'))
    model.eval()
    # 提示
    print("Model is ready.")

    # --------------------------------------------------(3)test--------------------------------------
    print('Test is running.\n\n')

    # 参数的含义在于是否将测试集中预测错误的例子展示出来
    test(show_error=True)

    print('\n\nFinished.')

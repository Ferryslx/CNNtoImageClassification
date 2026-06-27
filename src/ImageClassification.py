import datetime

import torch
import torch.nn as nn
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor  # pip install torchvision -i https://mirrors.aliyun.com/pypi/simple/
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import matplotlib.pyplot as plt
from torchsummary import summary

from utils.log import Logger

# 每批次样本数
BATCH_SIZE = 8


#1.准备数据集
def create_dataset():
    #参1：数据集路径 参2：是否是训练集  参3：数据预处理(张量数据)   参4：是否联网下载
    train_dataset=CIFAR10(root='../data', train=True,transform=ToTensor(),download=True)
    test_dataset=CIFAR10(root='../data', train=False,transform=ToTensor(),download=True)
    return train_dataset,test_dataset

#2.搭建（卷积）神经网络
class ImageClassification(nn.Module):
    def __init__(self):
        super().__init__()
        #参1：输入3通道  参2：卷积核个数（输出6通道）  参3：卷积核大小  参4：步长1  参5：填充0（不填充）
        self.conv1=nn.Conv2d(3,6,3,1,0)
        #参1：池化窗口的大小，步长2，填充0
        self.pool1=nn.MaxPool2d(2,2,0)
        self.conv2=nn.Conv2d(6,16,3,1,0)
        self.pool2=nn.MaxPool2d(2,2,0)
        self.linear1=nn.Linear(576,120)
        self.linear2=nn.Linear(120,84)
        self.output=nn.Linear(84,10)
        #日志
        # 日志
        logfile_name = 'CNNModel' + datetime.datetime.now().strftime('%Y%m%d')
        self.logger = Logger('../', logfile_name).get_logger()

    def forward(self,x):
        #卷积层->激励层(激活函数)->池化层
        x=self.pool1(torch.relu(self.conv1(x)))
        x=self.pool2(torch.relu(self.conv2(x)))
        #全连接层,注意：全连接层只能处理二维数据,所以要将数据拉平,原来的x的维度是:(8,16,6,6)(因为每一批有8个数据)，现在要变成（8，576）
        x=x.reshape(x.size(0),-1)     #参1：样本数（行数）  参2：特征数，-1表示自动计算
        x=torch.relu(self.linear1(x))
        x=torch.relu(self.linear2(x))
        return self.output(x)       #损失函数用多分类交叉熵损失函数，所以不用softmax激活函数


#3.模型训练
def train(model,train_dataset):
    dataloader=DataLoader(train_dataset,batch_size=BATCH_SIZE,shuffle=True)
    criterion=nn.CrossEntropyLoss()
    optimizer=optim.Adam(model.parameters(),lr=0.001,betas=(0.9,0.999))
    logger=model.logger

    epochs=100  #训练总轮数
    logger.info(f'模型训练开始,当前时间为:{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}')
    for epoch in range(epochs):
        start = time.time()
        #定义变量，记录每轮的损失函数和批次数
        total_loss,batch_num=0.0,0
        #定义变量，记录预测正确的样本数
        total_correct=0
        for x,y in dataloader:
            model.train()  # 切换模型状态
            y_pred=model(x)
            loss=criterion(y_pred,y)
            optimizer.zero_grad()
            loss.sum().backward()
            optimizer.step()
            total_loss+=loss.item()*x.size(0)
            batch_num+=x.size(0)
            # 根据加权求和得到类别，用argmax函数获取最大值对应的下标，就是类别
            y_pred = torch.argmax(y_pred, dim=1)  # dim=1表示逐行处理
            # 计算准确率
            total_correct += (y_pred == y).sum().item()
        logger.info(f'当前轮数:{epoch}，当前轮的平均损失:{total_loss/batch_num:.4f},当前轮的正确率(Accuracy):{total_correct/len(train_dataset)*100}%,耗时：{time.time()-start:.4f}s')
        print((f'当前轮数:{epoch}，当前轮的平均损失:{total_loss/batch_num:.4f},当前轮的正确率(Accuracy):{total_correct/len(train_dataset)*100}%,耗时：{time.time()-start:.4f}s'))
    logger.info(f'模型训练结束,当前时间为:{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}')
    torch.save(model.state_dict(),'../model/CNNImageModel.pth')
#4.模型测试



if __name__ == '__main__':
    train_dataset,test_dataset= create_dataset()
    model=ImageClassification()
    logger=model.logger
    logger.info(f'开始创建模型对象:{model}')
    #查看模型参数
    summary(model,(3,32,32),batch_size=BATCH_SIZE)
    train(model,train_dataset)
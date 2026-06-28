import datetime

import torch
import torch.nn as nn
from torchvision.datasets import CIFAR10
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import time
from torchsummary import summary
from torchvision import transforms
from utils.log import Logger
from copy import deepcopy

"""
模型优化（v3）：
增加网络深度：2层卷积 → 6层卷积，3个下采样阶段，通道数逐级增加（32→64→128）
"""

# 每批次样本数
BATCH_SIZE = 32
PATIENCE = 20          # 早停容忍轮数

#数据增强
train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4), #随机裁剪，向四周填充四个像素，然后随机裁剪成32*32
    transforms.RandomHorizontalFlip(),          #随机水平翻转
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2470, 0.2435, 0.2616]
    )
])


val_test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2470, 0.2435, 0.2616]
    )
])


#1.准备数据集
def create_dataset():
    full_train = CIFAR10(root='../data',train=True,transform=train_transform,download=True)
    test_dataset = CIFAR10(root='../data',train=False,transform=val_test_transform,download=True)

    # 从训练集中拆出 10000 条作为验证集
    train_dataset, val_dataset = random_split(full_train,[40000, 10000],generator=torch.Generator().manual_seed(42))

    # 验证集不用增强
    val_dataset.dataset.transform = val_test_transform
    return train_dataset, val_dataset, test_dataset

#2.搭建（卷积）神经网络
class ImageClassification(nn.Module):
    def __init__(self):
        super().__init__()

        # === Stage 1: 32x32 → 16x16 ===
        self.conv1a = nn.Conv2d(3, 32, 3, 1, 1)   # pad=1 保持空间尺寸
        self.bn1a = nn.BatchNorm2d(32)
        self.conv1b = nn.Conv2d(32, 32, 3, 1, 1)
        self.bn1b = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)            # 32→16

        # === Stage 2: 16x16 → 8x8 ===
        self.conv2a = nn.Conv2d(32, 64, 3, 1, 1)
        self.bn2a = nn.BatchNorm2d(64)
        self.conv2b = nn.Conv2d(64, 64, 3, 1, 1)
        self.bn2b = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)            # 16→8

        # === Stage 3: 8x8 → 4x4 ===
        self.conv3a = nn.Conv2d(64, 128, 3, 1, 1)
        self.bn3a = nn.BatchNorm2d(128)
        self.conv3b = nn.Conv2d(128, 128, 3, 1, 1)
        self.bn3b = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)            # 8→4

        # === 全连接层 ===
        self.linear1 = nn.Linear(128 * 4 * 4, 512)
        self.bn_fc1 = nn.BatchNorm1d(512)
        self.linear2 = nn.Linear(512, 256)
        self.bn_fc2 = nn.BatchNorm1d(256)
        self.output = nn.Linear(256, 10)

        # Dropout
        self.dropout = nn.Dropout(p=0.5)
        # 日志
        logfile_name = 'CNNModel_optimization' + datetime.datetime.now().strftime('%Y%m%d')
        self.logger = Logger('../', logfile_name).get_logger()

    def forward(self, x):
        # Stage 1: conv1a → bn1a → relu → conv1b → bn1b → relu → pool
        x = torch.relu(self.bn1a(self.conv1a(x)))
        x = torch.relu(self.bn1b(self.conv1b(x)))
        x = self.pool1(x)

        # Stage 2
        x = torch.relu(self.bn2a(self.conv2a(x)))
        x = torch.relu(self.bn2b(self.conv2b(x)))
        x = self.pool2(x)

        # Stage 3
        x = torch.relu(self.bn3a(self.conv3a(x)))
        x = torch.relu(self.bn3b(self.conv3b(x)))
        x = self.pool3(x)

        # 展开 + 全连接
        x = x.reshape(x.size(0), -1)               # (B, 128*4*4)
        x = torch.relu(self.bn_fc1(self.linear1(x)))
        x = self.dropout(x)
        x = torch.relu(self.bn_fc2(self.linear2(x)))
        x = self.dropout(x)
        return self.output(x)

#创建早停类
class EarlyStopping:
    def __init__(self, patience=10, delta=0.001, mode='max'):
        self.patience = patience
        self.delta = delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_state = None

    def __call__(self, score, model):
        metric = score if self.mode == 'max' else -score

        if self.best_score is None:
            self.best_score = metric
            self.best_state = deepcopy(model.state_dict())
        elif metric < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = metric
            self.best_state = deepcopy(model.state_dict())
            self.counter = 0

#3.模型训练
def train(model,train_dataset,val_dataset):
    dataloader=DataLoader(train_dataset,batch_size=BATCH_SIZE,shuffle=True)
    valloader=DataLoader(val_dataset,batch_size=BATCH_SIZE,shuffle=False)
    criterion=nn.CrossEntropyLoss()
    optimizer=optim.SGD(model.parameters(),lr=0.1,momentum=0.9,weight_decay=1e-4)  #优化器换成SGD+动量法
    scheduler=optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=100)  #学习率调度
    #早停类
    early_stopping = EarlyStopping(patience=PATIENCE)
    logger=model.logger

    epochs=100  #训练总轮数
    logger.info(f'模型训练开始,当前时间为:{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}')
    for epoch in range(epochs):
        start = time.time()
        #定义变量，记录每轮的损失函数和批次数
        total_loss,batch_num=0.0,0
        #定义变量，记录预测正确的样本数
        total_correct=0
        model.train()  # 切换模型状态
        for x,y in dataloader:
            y_pred=model(x)
            loss=criterion(y_pred,y)
            optimizer.zero_grad()
            loss.sum().backward()
            optimizer.step()
            total_loss+=loss.item()*x.size(0)
            batch_num+=x.size(0)
            # 根据加权求和得到类别，用argmax函数获取最大值对应的下标，就是类别
            y_pred = torch.argmax(y_pred, dim=-1)  # dim=-1表示逐行处理
            # 计算准确率
            total_correct += (y_pred == y).sum().item()
        #验证集评估
        model.eval()
        val_correct = 0
        with torch.no_grad():
            for x_val, y_val in valloader:
                y_pred = model(x_val)
                val_correct += (y_pred.argmax(dim=-1) == y_val).sum().item()
        val_acc = val_correct / len(val_dataset)
        train_acc = total_correct / len(train_dataset)
        logger.info(f'当前轮数:{epoch}，当前轮的平均损失:{total_loss/batch_num:.4f},当前轮训练集的正确率(Accuracy):{train_acc*100:.4f}%,当前轮验证集的正确率为:{val_acc*100:.4f}%,耗时：{time.time()-start:.4f}s')
        print(f'当前轮数:{epoch}，当前轮的平均损失:{total_loss/batch_num:.4f},当前轮训练集的正确率(Accuracy):{train_acc*100:.4f}%,当前轮验证集的正确率为:{val_acc*100:.4f}%,耗时：{time.time()-start:.4f}s')

        #早停策略评估
        early_stopping(val_acc, model)
        if early_stopping.early_stop:
            print(f"早停策略触发。当前训练轮数为:{epoch}")
            logger.info(f"早停策略触发。当前训练轮数为:{epoch}")
            break
        scheduler.step()

    logger.info(f'模型训练结束,当前时间为:{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}')
    torch.save(early_stopping.best_state,'../model/CNNImageModel_optimization.pth')


#4.模型测试
def evaluate(test_dataset):
    dataloader=DataLoader(test_dataset,batch_size=BATCH_SIZE,shuffle=False)
    model=ImageClassification()
    model.load_state_dict(torch.load('../model/CNNImageModel_optimization.pth'))
    logger=model.logger
    logger.info(f'模型测试开始，当前时间为:{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}')
    #定义变量,统计预测正确的样本个数
    total_correct=0
    for x,y in dataloader:
        model.eval()   #切换模型状态
        y_pred=model(x)
        y_pred=torch.argmax(y_pred,dim=-1)
        total_correct+=(y_pred == y).sum().item()
    logger.info(f'测试集的正确率(Accuracy):{total_correct / len(test_dataset) * 100:.4f}%，测试集的样本数量是:{len(test_dataset)}')
    print((f'测试集的正确率(Accuracy):{total_correct / len(test_dataset) * 100:.4f}%'))
    logger.info(f'模型测试结束，当前时间为:{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}')


if __name__ == '__main__':
    train_dataset,val_dataset,test_dataset= create_dataset()
    model=ImageClassification()
    logger=model.logger
    logger.info(f'开始创建模型对象:{model}')
    logger.info(f'开始模型优化，当前时间为:{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}')
    #查看模型参数
    summary(model,(3,32,32),batch_size=BATCH_SIZE)
    train(model,train_dataset,val_dataset)
    evaluate(test_dataset)
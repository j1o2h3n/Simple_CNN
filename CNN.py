#encoding=utf-8
import csv
import  torch
import  torch.nn as nn
import  torch.nn.functional as F
import  torch.optim as optim
from    torchvision import datasets, transforms


batch_size=100      # 批量大小
learning_rate=0.001  # 学习率
epochs=3           # 训练次数

# 读取训练集
train_data = datasets.ImageFolder('./train', transform=transforms.Compose([
        # transforms.RandomRotation(10), #在（-10， 10）范围内旋转
        # transforms.RandomRotation(2),
        # transforms.RandomAffine(1),#旋转
        # transforms.RandomAffine(365),
        # transforms.RandomAffine(2),
        # transforms.RandomAffine(364),
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((60, 30)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ]))
train_loader = torch.utils.data.DataLoader(train_data,      # 数据
                                           batch_size=batch_size,    # 批量大小
                                           shuffle=True,    # 每次迭代训练时是否将数据洗牌
                                           )        # 设置为True在每个epoch重新排列数据（默认False,一般打乱比较好）
# 读取测试集
test_data = datasets.ImageFolder('./test', transform=transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((60, 30)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ]))
test_loader = torch.utils.data.DataLoader(test_data,      # 数据
                                           batch_size=batch_size,    # 批量大小
                                           shuffle=True,    # 每次迭代训练时是否将数据洗牌
                                           )        # 设置为True在每个epoch重新排列数据（默认False,一般打乱比较好）

class Net(nn.Module):  # 定义网络，推荐使用Sequential，结构清晰
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Sequential(  # input_size = 60*30*1
            torch.nn.Conv2d(in_channels=1, out_channels=64, kernel_size=6, stride=2, padding=0),     # 卷积层
            torch.nn.ReLU(),     # (60-6)/2+1=28,(30-6)/2+1=13  28*13*8                             # 激活层
            torch.nn.MaxPool2d(kernel_size=3, stride=1),  # output_size = 26*11*8                 # 池化层
            torch.nn.Dropout(0.2)
        )
        self.conv2 = torch.nn.Sequential(  # input_size = 26*11*8
            torch.nn.Conv2d(64, 64, 5, 1, 2), # (26-5+2*2)/1+1=26,(11-5+2*2)/1+1=11  26*11*16
            torch.nn.ReLU()
            # torch.nn.MaxPool2d(2, 1),  # output_size = 25*12*16
            # torch.nn.Dropout(0.2)
        )
        self.conv3 = torch.nn.Sequential(  # input_size = 26*11*8
            torch.nn.Conv2d(64, 16, 2, 1, 2), # (26-5+2*2)/1+1=26,(11-5+2*2)/1+1=11  26*11*16
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 1),  # output_size = 25*12*16
            torch.nn.Dropout(0.2)
        )
        # 网络前向传播过程
        self.dense = torch.nn.Sequential(
            torch.nn.Linear(5824, 2048),# 25*12*16=4800
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(2048, 512)
			torch.nn.Dropout(0.5),
            torch.nn.Linear(512, 68)
        )

    def forward(self, x):  # 正向传播过程
        conv1_out = self.conv1(x)
        conv2_out = self.conv2(conv1_out)
        conv3_out = self.conv3(conv2_out)
        res = conv3_out.view(conv3_out.size(0), -1)
        out = self.dense(res)
        return out

device = torch.device('cuda:0')     # 分配的所有CUDA张量将默认在该设备上创建
net = Net().to(device)              # 链接神经网络Net和GPU
optimizer = optim.SGD(net.parameters(), lr=learning_rate)       # 优化方法,优化器SGD参数设置
criteon = nn.CrossEntropyLoss().to(device)                      # 定义损失函数，交叉熵损失


with open('./datas/test.csv', 'w', newline='') as csvfile:  # 写入文件夹
    writer = csv.writer(csvfile)
    writer.writerow(['loss', 'true'])   # 写入标题

    # 训练网络,迭代epoch
    for epoch in range(epochs): # 循环epoch次数
        for batch_idx, (data, target) in enumerate(train_loader,0):   # 读取训练集
            # data = data.view(-1, 60*30)                             # 取成28*28维数据
            data, target = data.to(device), target.cuda()           # 数据,目标 张量放入GPU
            logits = net(data)                                      # 数据送入神经网络
            loss = criteon(logits, target)                          # 计算标准loss

            optimizer.zero_grad()                                 # 把梯度置零
            loss.backward()                                       # loss求导,反向传播,进行梯度求解
            # print(w1.grad.norm(), w2.grad.norm())
            optimizer.step()                                      # 回传损失过程中计算梯度,更新参数

            # if batch_idx % 100 == 0:
            print('Net Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch+1,                                    # 第几个epoch
                batch_idx * len(data),                      # 已使用数据量
                len(train_loader.dataset),                  # 总共数据量
                100. * batch_idx / len(train_loader),       # 已训练百分比
                loss.item()))                               # 此时loss值

        # 测试网络，喂数据
        test_loss = 0   # 测试的loss值
        correct = 0     # 正确个数
        for data, target in test_loader:            # 读取 数据,目标
            #data = data.view(-1, 60 * 30)
            data, target = data.to(device), target.cuda()   # 数据,目标 张量放入GPU
            logits = net(data)                              # 数据送入神经网络
            test_loss += criteon(logits, target).item()     # 输出值loss累加

            pred = logits.argmax(dim=1)                     # 返回指定维度1最大值的序号,即指神经网络预测
            correct += pred.eq(target).float().sum().item() # 预测正确的总数

        test_loss /= len(test_loader.dataset)               # loss值
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
                    test_loss,                                    # 平均loss值
                    correct,                                      # 预测正确的总数
                    len(test_loader.dataset),                     # 数据总数
                    100. * correct / len(test_loader.dataset)))   # 预测正确百分比

        writer.writerow([test_loss,correct/len(test_loader.dataset)])  # 写入多行

torch.save(Net, './model/model_Net.pth')    # 保存整个模型，体积比较大

csvfile.close()    # 关闭文件

print('All End')


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import time

from Transformer import TransformerModel
from data_loader import get_loader_segment
import matplotlib.pyplot as plt

def adjust_learning_rate(optimizer, epoch, lr_):
    """
     根据epoch调整学习率。

     参数:
     optimizer (torch.optim.Optimizer): 优化器。
     epoch (int): 当前epoch。
     lr_ (float): 初始学习率。
     """
    lr_adjust = {epoch: lr_ * (0.5 ** ((epoch - 1) // 1))}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))


class EarlyStopping:

    def __init__(self, patience=7, verbose=False, dataset_name='', delta=0):
        """
              初始化早期停止类。

              参数:
              patience (int): 在停止训练之前可以接受的没有改进的epoch数量，默认为7。
              verbose (bool): 是否打印信息，默认为False。
              dataset_name (str): 数据集名称，默认为空字符串。
              delta (float): 损失值需要改善的最小阈值，默认为0。
              """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.best_score2 = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.val_loss2_min = np.Inf
        self.delta = delta
        self.dataset = dataset_name

    def __call__(self, val_loss, val_loss2, model, path):
        score = -val_loss# 将验证损失转换为负数，以便最大化
        score2 = -val_loss2# 将第二个验证损失转换为负数，以便最
        if self.best_score is None:
            # 如果是第一次调用，初始化最佳损失值
            self.best_score = score
            self.best_score2 = score2
            self.save_checkpoint(val_loss, val_loss2, model, path)
        elif score < self.best_score + self.delta or score2 < self.best_score2 + self.delta:
            # 如果当前验证损失没有改善超过delta
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            # 如果当前验证损失改善超过delta
            self.best_score = score
            self.best_score2 = score2
            self.save_checkpoint(val_loss, val_loss2, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, val_loss2, model, path):
        import torch

        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), os.path.join(path, str(self.dataset) + '_checkpoint.pth'))
        self.val_loss_min = val_loss
        self.val_loss2_min = val_loss2


class Solver(object):
    DEFAULTS = {}

    def __init__(self, config):

        self.__dict__.update(Solver.DEFAULTS, **config)
        torch.cuda.empty_cache()
 

        self.train_loader = get_loader_segment(self.data_path, batch_size=self.batch_size, win_size=self.win_size,
                                               mode='train', dataset=self.dataset)
        self.vali_loader = get_loader_segment(self.data_path, batch_size=self.batch_size, win_size=self.win_size,
                                              mode='val', dataset=self.dataset)
        self.test_loader = get_loader_segment(self.data_path, batch_size=self.batch_size, win_size=self.win_size,
                                              mode='test', dataset=self.dataset)
        self.thre_loader = get_loader_segment(self.data_path, batch_size=self.batch_size, win_size=self.win_size,
                                              mode='thre', dataset=self.dataset)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.build_model()

    def build_model(self):
        """
         构建Transformer模型并设置优化器和损失函数。
         """
        # 构建Transformer模型实例，并将其移动到指定设备
        self.model = TransformerModel(n_feature=self.input_c, d_model=512, nhead=8, d_hid=512, nlayers=3, dropout=0.2).to(self.device)
        # 设置Adam优化器
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        # 设置均方误差损失函数
        self.criterion = nn.MSELoss()
        if torch.cuda.is_available():
            self.model.cuda()

    def vali(self, vali_loader):
        """
          在验证集上评估模型。

          参数:
          vali_loader (torch.utils.data.DataLoader): 验证集数据加载器。

          返回:
          float: 验证集的平均损失。
          """
        # 定义均方误差损失函数
        self.model.eval()
        # 定义均方误差损失函数
        crit = nn.MSELoss()
        loss_1 = []
        for i, (input_data, _) in enumerate(vali_loader):
            input = input_data.float().to(self.device)#将输入数据转换为浮点数并移动到设备。
            output = self.model(input, src_mask=None)#模型前向传播，得到重构数据。
            rec_loss = crit(output, input)
            loss_1.append(rec_loss.item())  

        return np.average(loss_1)  # 返回平均损失

    def train(self):

        print("======================TRAIN MODE======================")

        time_now = time.time()  # 记录当前时间
        path = self.model_save_path  # 模型保存路径
        if not os.path.exists(path):
            os.makedirs(path)  # 如果路径不存在，创建路径
        early_stopping = EarlyStopping(patience=3, verbose=True, dataset_name=self.dataset)  # 初始化早期停止类
        train_steps = len(self.train_loader)  # 训练步骤数量

        train_losses = []  # 初始化列表用于存储训练损失
        val_losses = []  # 初始化列表用于存储验证损失

        for epoch in range(self.num_epochs):
            iter_count = 0  # 初始化迭代计数器
            loss1_list = []  # 初始化损失列表用于存储每个批次的损失值

            epoch_time = time.time()  # 记录每个epoch的开始时间
            self.model.train()  # 将模型设置为训练模式

            for i, (input_data, labels) in enumerate(self.train_loader):
                self.optimizer.zero_grad()  # 清除梯度
                iter_count += 1

                input = input_data.float().to(self.device)  # 将输入数据转换为浮点数并移动到设备

                output = self.model(input, src_mask=None)  # 模型前向传播

                loss = self.criterion(output, input)  # 计算损失

                loss1_list.append(loss.item())  # 将损失值添加到列表中

                if (i + 1) % 100 == 0:
                    speed = (time.time() - time_now) / iter_count  # 计算当前速度
                    left_time = speed * ((self.num_epochs - epoch) * train_steps - i)  # 计算剩余时间
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))  # 打印速度和剩余时间
                    iter_count = 0  # 重置迭代计数器
                    time_now = time.time()  # 重置时间记录

                loss.backward()  # 反向传播计算梯度
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)  # 梯度裁剪

                self.optimizer.step()  # 更新模型参数

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))  # 打印每个epoch的耗时
            train_loss = np.average(loss1_list)  # 计算训练集的平均损失

            vali_loss = self.vali(self.vali_loader)  # 计算验证集的平均损失

            train_losses.append(train_loss)  # 将训练集损失添加到列表中
            val_losses.append(vali_loss)  # 将验证集损失添加到列表中

            print(
                "Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} ".format(
                    epoch + 1, train_steps, train_loss, vali_loss))  # 打印每个epoch的训练和验证损失
            early_stopping(vali_loss, vali_loss, self.model, path)  # 调用早期停止逻辑
            if early_stopping.early_stop:
                print("Early stopping")  # 打印提前停止信息
                break  # 提前停止训练
            adjust_learning_rate(self.optimizer, epoch + 1, self.lr)  # 调整学习率

        np.save('final_vae_model/tf_train_loss.npy',train_losses)  # 保存训练损失
        np.save('final_vae_model/tf_val_loss.npy', val_losses)  # 保存验证损失
        plt.plot(train_losses)  # 绘制训练损失曲线
        plt.plot(val_losses)  # 绘制验证损失曲线
        plt.title('Training/Validation Loss Curve')  # 设置图表标题
        plt.xlabel('epoch')  # 设置x轴标签
        plt.ylabel('loss')  # 设置y轴标签
        plt.legend(['Training Loss', 'Validation Loss'])  # 设置图例
        plt.grid(True)  # 显示网格
        plt.savefig('tf_loss.png', dpi=500)  # 保存图表为图像文件
        plt.close()  # 关闭图表以释放内存

    def test(self):
        """
        在测试集上评估模型，并保存训练集和测试集的损失以及重构的输出。
        """
        # 加载最佳模型检查点
        self.model.load_state_dict(
            torch.load(
                os.path.join(str(self.model_save_path), str(self.dataset) + '_checkpoint.pth')))

        # 将模型设置为评估模式
        self.model.eval()

        print("======================TEST MODE======================")

        # 定义损失函数，不使用reduce
        criterion = nn.MSELoss(reduce=False)

        # 初始化损失列表用于存储训练集损失
        losses = []

        # 计算训练集的损失
        for i, (input_data, labels) in enumerate(self.train_loader):
            input = input_data.float().to(self.device)  # 将输入数据转换为浮点数并移动到设备

            output = self.model(input, src_mask=None)  # 模型前向传播

            # 计算重构损失
            loss = torch.mean(criterion(input, output), dim=-1)
            loss = loss.detach().cpu().numpy()  # 将损失值转换为NumPy数组
            losses.append(loss)

        # 合并训练集损失
        losses = np.concatenate(losses, axis=0)
        train_loss = np.array(losses)  # 将合并后的损失值转换为NumPy数组
        np.save('final_vae_model/loss_train_data.npy', train_loss)  # 保存训练集损失

        # 初始化标签、损失和输出列表用于存储测试集数据
        test_labels = []
        losses = []
        outputs = []

        # 计算测试集的损失
        for i, (input_data, labels) in enumerate(self.test_loader):
            print(f"Batch index: {i}")

            # 打印输入数据和标签的形状
            print('Input data and labels')
            print(input_data.shape)
            print(labels.shape)

            input = input_data.float().to(self.device)  # 将输入数据转换为浮点数并移动到设备
            output = self.model(input, src_mask=None)  # 模型前向传播
            outputs.append(output.detach().cpu().numpy())  # 将输出数据转换为NumPy数组并添加到列表中

            # 计算重构损失
            loss = torch.mean(criterion(input, output), dim=-1)
            loss = loss.detach().cpu().numpy()  # 将损失值转换为NumPy数组
            losses.append(loss)
            test_labels.append(labels)

        # 合并输出数据
        outputs = np.concatenate(outputs, axis=0)
        np.save('final/tf_output.npy', outputs)  # 保存输出数据

        # 合并测试集损失
        losses = np.concatenate(losses, axis=0)
        test_loss = np.array(losses)  # 将合并后的损失值转换为NumPy数组
        np.save('final/loss_test_data.npy', test_loss)  # 保存测试集损失

        # 合并测试集标签
        test_labels = np.concatenate(test_labels, axis=0).reshape(-1)
        np.save('final_vae_model/test_labels.npy', test_labels)  # 保存测试集标签

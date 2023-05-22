# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import config
import random
from collections import OrderedDict

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# sequential 降维
class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


# 普通CNN
class ConvNet(nn.Module):
    def __init__(self, board_size):
        super(ConvNet, self).__init__()
        self.board_size = board_size
        # 公共层
        self.conv1 = nn.Conv2d(4, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        # self.conv4 = nn.Conv2d(128,256,kernel_size=3, padding=1)
        # 动作概率层
        self.p_conv1 = nn.Conv2d(128, 4, kernel_size=1)  # out:size-(1, 4, size, size)
        self.p_fc1 = nn.Linear(4 * board_size**2, board_size**2)  # in:size-(1,4*size*size)  out-size(1,size*size)
        # 价值层
        self.value_conv1 = nn.Conv2d(128, 2, kernel_size=1)
        self.value_fc1 = nn.Linear(2 * board_size**2, 64)  # 输出 [1,64]
        self.value_fc2 = nn.Linear(64, 1)  # 输出[1] 局面评分
        # sequential
        # self.common_layer = nn.Sequential(
        #     nn.Conv2d(4, 32, kernel_size=3, padding=1),
        #     nn.ReLU(),
        #     nn.Conv2d(32, 64, kernel_size=3, padding=1),
        #     nn.ReLU(),
        #     nn.Conv2d(64, 128, kernel_size=3, padding=1),
        #     nn.ReLU()
        # )
        #
        # self.policy_module = nn.Sequential(
        #     nn.Conv2d(128, 4, kernel_size=1),
        #     nn.ReLU(),
        #     Flatten(),
        #     nn.Linear(4 * self.board_size**2, self.board_size**2),
        #     nn.LogSoftmax(dim=1)
        # )
        #
        # self.value_module = nn.Sequential(
        #     nn.Conv2d(128, 2, kernel_size=1),
        #     nn.ReLU(),
        #     Flatten(),
        #     nn.Linear(2 * self.board_size**2, 64),
        #     nn.ReLU(),
        #     nn.Linear(64, 1),
        #     nn.Tanh()
        # )

    def forward(self, state):
        x = F.relu(self.conv1(state))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        # 动作概率层
        probability = F.relu(self.p_conv1(x))
        probability = probability.view(-1, 4 * self.board_size**2)  # 4d->2d:输出size-(1,size*size)
        probability = F.log_softmax(self.p_fc1(probability), dim=1)  # 先softmax得出概率再log
        # 状态价值层
        value = F.relu(self.value_conv1(x))
        value = value.view(-1, 2 * self.board_size**2)
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))
        # sequential
        # x = self.common_layer(states)
        # probability = self.policy_module(x)
        # value = self.value_module(x)
        return probability, value


# 使用 CNN
class NetFunction:
    def __init__(self, board_size, model_path=None, use_gpu=config.USE_GPU):
        self.use_gpu = use_gpu
        self.board_size = board_size
        # 创建网络
        self.cnn = ConvNet(self.board_size).cuda() if self.use_gpu else ConvNet(board_size)
        # 创建Adam优化器
        self.optimizer = optim.Adam(self.cnn.parameters(), weight_decay=config.L2_NORM, lr=config.LEARNING_RATE)
        # 读取训练好的模型
        if model_path:
            self.cnn.load_state_dict(torch.load(model_path))

    # 输出向量p/标量v 用于指导mcts模拟
    def get_policy_value_for_mcts(self, board_info):
        avail_move_lists = board_info.avail_move_list
        # input:size-(4,8,8)
        # Error:some of the strides of a given numpy array are negative... -> np.ascontiguousarray
        state_planes = torch.from_numpy(np.ascontiguousarray(board_info.get_feature_planes())).unsqueeze(dim=0)  # NCHW 增加一个假维度
        # 获得tensor类型的标量value,可用item()取出
        logp_list, value = self.cnn(state_planes.cuda().float()) if self.use_gpu else self.cnn(state_planes.float())  # 从4维numpy转为tensor格式输入网络
        p_list = np.exp(logp_list.data.cpu().numpy().flatten())
        # tmp = np.zeros(len(p_list))
        # tmp[list(avail_move_lists)] = p_list[avail_move_lists]
        # print(state_planes)
        # print(np.array(tmp).reshape(9, 9))
        # print(value)
        # print(avail_move_lists)
        # print("------")
        p_list = zip(avail_move_lists, p_list[avail_move_lists])
        # print(list(p_list))
        # 输出(move序号, 先验概率) 以及 局面价值
        return p_list, value.item()

    # 训练模型
    def training(self, dataset):
        # 从数据集抽取若干数据
        batch_data = random.sample(dataset, config.BATCH_SIZE)
        state_list = [data[0] for data in batch_data]
        mcts_pi_list = [data[1] for data in batch_data]
        mcts_z_list = [data[2] for data in batch_data]
        # 数据转tensor
        state_list = torch.tensor(state_list).cuda().float() if self.use_gpu else torch.tensor(state_list).float()
        mcts_pi_list = torch.tensor(mcts_pi_list).cuda().float() if self.use_gpu else torch.tensor(mcts_pi_list).float()
        mcts_z_list = torch.tensor(mcts_z_list).cuda().float() if self.use_gpu else torch.tensor(mcts_z_list).float()
        aggregate_loss, value_loss, policy_loss = 0.0, 0.0, 0.0
        # 一组batch训练epochs次 也相当于数据集训练了epochs次
        for _ in range(config.EPOCHS):
            # --优化器-- #
            # 清空梯度
            self.optimizer.zero_grad()
            # 设定学习率
            # for param_group in self.optimizer[0].param_groups:  # 优化器的param_groups（字典）
            #     param_group['lr'] = lr
            # 设定学习率衰减器
            # mode为min->loss停止下降 乘以factor使得lr下降; max->监控量停止上升则乘以factor
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.9, patience=20)
            # 前向传播
            output_log_p_list, output_value_list = self.cnn(state_list)
            # Loss = (z - v)^2 - pi^T * log(p) + c||theta||^2
            # 计算Loss
            value_loss = F.mse_loss(output_value_list.view(-1), mcts_z_list)  # 均方差损失
            policy_loss = -torch.mean(torch.sum(mcts_pi_list * output_log_p_list, 1))  # 交叉熵损失
            aggregate_loss = value_loss + policy_loss  # 总损失
            # 反向传播
            aggregate_loss.backward()
            # 更新参数
            self.optimizer.step()
            scheduler.step(aggregate_loss)
        return aggregate_loss.item(), value_loss.item(), policy_loss.item()

    # 保存模型
    def save_model(self, model_path):
        net_params = self.cnn.state_dict()
        torch.save(net_params, model_path)

    # @staticmethod
    # def transfer_dict_name(model_path):
    #     old_state_dict = torch.load(model_path)
    #     new_state_dict = OrderedDict()
    #     for key in old_state_dict:
    #         if key == 'conv1.weight':
    #             new_state_dict['common_layer.0.weight'] = old_state_dict[key]
    #             new_state_dict['conv1.weight'] = old_state_dict[key]
    #         elif key == 'conv1.bias':
    #             new_state_dict['common_layer.0.bias'] = old_state_dict[key]
    #             new_state_dict['conv1.bias'] = old_state_dict[key]
    #         elif key == 'conv2.weight':
    #             new_state_dict['common_layer.2.weight'] = old_state_dict[key]
    #             new_state_dict['conv2.weight'] = old_state_dict[key]
    #         elif key == 'conv2.bias':
    #             new_state_dict['common_layer.2.bias'] = old_state_dict[key]
    #             new_state_dict['conv2.bias'] = old_state_dict[key]
    #         elif key == 'conv3.weight':
    #             new_state_dict['common_layer.4.weight'] = old_state_dict[key]
    #             new_state_dict['conv3.weight'] = old_state_dict[key]
    #         elif key == 'conv3.bias':
    #             new_state_dict['common_layer.4.bias'] = old_state_dict[key]
    #             new_state_dict['conv3.bias'] = old_state_dict[key]
    #         elif key == 'p_conv1.weight':
    #             new_state_dict['policy_module.0.weight'] = old_state_dict[key]
    #             new_state_dict['p_conv1.weight'] = old_state_dict[key]
    #         elif key == 'p_conv1.bias':
    #             new_state_dict['policy_module.0.bias'] = old_state_dict[key]
    #             new_state_dict['p_conv1.bias'] = old_state_dict[key]
    #         elif key == 'p_fc1.weight':
    #             new_state_dict['policy_module.3.weight'] = old_state_dict[key]
    #             new_state_dict['p_fc1.weight'] = old_state_dict[key]
    #         elif key == 'value_conv1.weight':
    #             new_state_dict['value_module.0.weight'] = old_state_dict[key]
    #             new_state_dict['value_conv1.weight'] = old_state_dict[key]
    #         elif key == 'value_conv1.bias':
    #             new_state_dict['value_module.0.bias'] = old_state_dict[key]
    #             new_state_dict['value_conv1.bias'] = old_state_dict[key]
    #         elif key == 'value_fc1.weight':
    #             new_state_dict['value_module.3.weight'] = old_state_dict[key]
    #             new_state_dict['value_fc1.weight'] = old_state_dict[key]
    #         elif key == 'value_fc1.bias':
    #             new_state_dict['value_module.3.bias'] = old_state_dict[key]
    #             new_state_dict['value_fc1.bias'] = old_state_dict[key]
    #         elif key == 'value_fc2.weight':
    #             new_state_dict['value_module.5.weight'] = old_state_dict[key]
    #             new_state_dict['value_fc2.weight'] = old_state_dict[key]
    #         elif key == 'value_fc2.bias':
    #             new_state_dict['value_module.5.bias'] = old_state_dict[key]
    #             new_state_dict['value_fc2.bias'] = old_state_dict[key]
    #         else:
    #             new_state_dict[key] = old_state_dict[key]
    #     torch.save(new_state_dict, config.TRANSFER_PATH)
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import config
import random
from torch.autograd import Variable


#  展平同一个维度的数据[4,3,2]->[4,6]
class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


class BasicBlock(nn.Module):
    def __init__(self, depth):
        super(BasicBlock, self).__init__()

        # 2层卷积
        self.residual = nn.Sequential(
            nn.Conv2d(depth, depth, 3, 1, 1),
            nn.BatchNorm2d(depth),
            nn.ReLU(),
            nn.Conv2d(depth, depth, 3, 1, 1),
            nn.BatchNorm2d(depth),
        )

    def forward(self, x):
        x = x + self.residual(x)  # 二层输出F(x)+第一层输入x
        x = F.relu(x)
        return x


# ResNet网络
class ResNet(nn.Module):
    def __init__(self, board_size, depth=config.RESNET_FILTER_NUM, blocks_num=config.RESNET_BLOCKS_NUM):
        super(ResNet, self).__init__()
        self.board_size = board_size
        # 公共层为1层卷积,后连n个blocks
        common_module = nn.ModuleList([
            nn.Conv2d(4, depth, 3, 1, 1),
            nn.BatchNorm2d(depth),
            nn.ReLU()
        ])
        # 公共层连blocks
        common_module.extend([BasicBlock(depth) for _ in range(blocks_num)])
        self.main_layer = nn.Sequential(*common_module)

        self.policy_module = nn.Sequential(
            nn.Conv2d(depth, 2, 1, 1),
            nn.BatchNorm2d(2),
            nn.ReLU(),
            Flatten(),  # input:torch.Size([1, 2, 8, 8]) output:torch.Size([1, 128])
            nn.Linear(2 * self.board_size**2, self.board_size**2),
            nn.LogSoftmax(dim=1)
        )

        self.value_module = nn.Sequential(
            nn.Conv2d(depth, 1, 1, 1),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            Flatten(),  # input:torch.Size([1, 1, size, size]) output:torch.Size([1, 1*size**2])
            nn.Linear(self.board_size**2, 1),
            nn.Tanh()  # output -1~1
        )

    def forward(self, states):
        x = self.main_layer(states)
        probability = self.policy_module(x)
        value = self.value_module(x)
        return probability, value


# 神经网络的相关方法
class NetFunction:
    def __init__(self, board_size, model_path=None, use_gpu=config.USE_GPU):
        self.use_gpu = use_gpu
        self.board_size = board_size
        # 创建网络
        self.resnet = ResNet(self.board_size).cuda() if self.use_gpu else ResNet(self.board_size)

        self.optimizer = optim.Adam(self.resnet.parameters(), weight_decay=config.L2_NORM)
        # 读取训练好的模型
        if model_path:
            self.resnet.load_state_dict(torch.load(model_path))

    # 输出p向量/v值用于指导mcts扩展和模拟
    def get_policy_value_for_mcts(self, board_info):
        # 输出(move序号, 先验概率) 以及 局面价值
        avail_move_lists = board_info.avail_move_list
        state_planes = torch.from_numpy(np.ascontiguousarray(board_info.get_feature_planes())).unsqueeze(dim=0)  # NCHW 增加一个假维度

        logp_list, value = self.resnet(state_planes.cuda().float()) if self.use_gpu else self.resnet(state_planes.float())
        p_list = np.exp(logp_list.data.cpu().numpy().flatten())

        p_list = zip(avail_move_lists, p_list[avail_move_lists])
        return p_list, value.item()

    # 训练模型
    def training(self, dataset):
        # 从数据集抽取若干数据
        batch_data = random.sample(dataset, config.BATCH_SIZE)
        state_planes = [data[0] for data in batch_data]  # size->(batch_size,4,s,s)
        mcts_pi_list = [data[1] for data in batch_data]  # size->(batch_size,s*s)
        mcts_z_list = [data[2] for data in batch_data]  # size->(batch_size)
        # 数据转tensor->size(batch_size,4,s,s)
        state_planes = torch.tensor(state_planes).cuda().float() if self.use_gpu else torch.tensor(state_planes).float()
        mcts_pi_list = torch.tensor(mcts_pi_list).cuda().float() if self.use_gpu else torch.tensor(mcts_pi_list).float()
        mcts_z_list = torch.tensor(mcts_z_list).cuda().float() if self.use_gpu else torch.tensor(mcts_z_list).float()
        aggregate_loss, value_loss, policy_loss = 0.0, 0.0, 0.0
        for _ in range(config.EPOCHS):
            # --优化器-- #
            # 清空梯度
            self.optimizer.zero_grad()
            # 设定学习率
            # for param_group in self.optimizer.param_groups:  # 优化器的param_groups（字典）
            #         #     param_group['lr'] = lr
            # 设定学习率衰减器
            # mode为min，则loss不下降学习率乘以factor，max则反之
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.9, patience=20)
            # 前向传播
            net_logp_list, net_value_list = self.resnet(state_planes)
            # Loss = (z - v)^2 - pi^T * log(p) + c||theta||^2
            # 计算Loss
            value_loss = F.mse_loss(net_value_list.view(-1), mcts_z_list)
            policy_loss = -torch.mean(torch.sum(mcts_pi_list * net_logp_list, 1))
            aggregate_loss = value_loss + policy_loss
            # 反向传播
            aggregate_loss.backward()
            # 更新参数
            self.optimizer.step()
            scheduler.step(aggregate_loss)
        return aggregate_loss.item(), value_loss.item(), policy_loss.item()

    # 保存模型
    def save_model(self, model_path):
        net_params = self.resnet.state_dict()
        torch.save(net_params, model_path)

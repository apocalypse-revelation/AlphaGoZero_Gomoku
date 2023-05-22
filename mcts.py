# -*- coding: utf-8 -*-

import numpy as np
import copy
import config
import time


# softmax函数:输入各节点的经过温度处理后的访问值
def softmax_func(x):
    if len(x.shape) > 1:  # 矩阵
        tmp_max = np.max(x, axis=1)  # 得到行中最大值->缩放每行的元素,避免溢出
        x = x - tmp_max.reshape((x.shape[0], 1))  # 缩放
        x = np.exp(x)  # 求指数, 套入softmax公式
        tmp_max = np.sum(x, axis=1)  # 每行元素求和
        x = x / tmp_max.reshape((x.shape[0], 1))
    else:  # 向量
        tmp_max = np.max(x)  # 避免溢出
        x = x - tmp_max  # 缩放数据
        x = np.exp(x)  # 求指数, 套入softmax公式
        tmp_max = np.sum(x)  # 求元素求和
        x = x / tmp_max  # 套入softmax公式
    return x


# mc树结点
class MCTNode(object):
    def __init__(self, parent_node, probability):
        self.parent_node = parent_node
        self.children_node = {}  # {move序号:节点对象, ...}
        self.visit_num = 0
        self.q_value = 0  # 节点价值
        self.w_value = 0
        self.prob = probability  # 节点被选择的先验概率

    # MCTS搜索第一步
    def select(self):
        # 选择子节点，根据选择argmax(Q+U)的节点，返回元组(行动move, 子节点对象obj)
        # 获取节点的Q+U最大值
        q_u = []
        for move_id, node_obj in self.children_node.items():
            q_u.append(node_obj.get_q_u(config.CPUCT))
        max_q_u = max(q_u)
        # 通过比对, 获取Q+U最大值的那个节点
        for item in self.children_node.items():
            if item[1].get_q_u(config.CPUCT) == max_q_u:
                return item

    # MCTS搜索第二、三步（以及神经网络的模拟过程）
    # 加入dirichlet噪声增加探索程度
    # 参数说明:P(s,a) = (1-0.25) * Pa + 0.25*η; dirichlet/η = 0.3; E = 0.25
    def expand(self, move_p_list, add_dirichlet):
        # move_p_list = (move序号,先验概率),(),()...
        # print(move_p_list)
        if add_dirichlet:
            move_p_list = list(move_p_list)
            # print(move_p_list)
            p_len = len(move_p_list)
            dirichlet_noise_list = np.random.dirichlet(config.DIRICHLET_ALPHA * np.ones(p_len))
            for i in range(p_len):
                if move_p_list[i][0] not in self.children_node:
                    self.children_node[move_p_list[i][0]] = MCTNode(self,
                                                                    (1 - config.DIRICHLET_WEIGHT) * move_p_list[i][1] + config.DIRICHLET_WEIGHT *
                                                                    dirichlet_noise_list[i])
        else:
            # move_p_list = list(move_p_list)
            # print(move_p_list)
            for move, p in move_p_list:
                if move not in self.children_node:
                    self.children_node[move] = MCTNode(self, p)

    # MCTS搜索第四步: 扩展后的节点反向更新所有祖先节点
    def back_recur(self, leaf_node_value):
        # 从叶子节点递归回根节点,再从根节点往下更新节点值
        # 更新新扩展的节点
        self.visit_num += 1
        self.w_value += leaf_node_value
        self.q_value = 1.0 * self.w_value / self.visit_num
        # 更新扩展节点的所有祖先节点
        i = 0
        node = self.parent_node
        while node:
            node.visit_num += 1
            node.w_value += -leaf_node_value if i % 2 == 0 else leaf_node_value
            node.q_value = 1.0 * node.w_value / node.visit_num
            node = node.parent_node  # 从下往上
            i = i + 1
            # print(i)

    # select节点的依据
    def get_q_u(self, c_puct):
        ucb_value = (c_puct * self.prob * np.sqrt(self.parent_node.visit_num) / (1 + self.visit_num))
        return self.q_value + ucb_value

    def is_leaf_node(self):
        return self.children_node == {}

    def is_root_node(self):
        return self.parent_node is None

    def is_mid_node(self):
        return self.children_node != {} and self.parent_node is not None


class MCTSPlayer(object):

    def __init__(self, neural_net, playout_num=400, is_selfplay_mode=False):

        self.root_node = MCTNode(None, 1.0)  # (move序号, 先验概率)
        self.is_selfplay_mode = is_selfplay_mode
        self.neural_net = neural_net  # 用于辅助mcts进行模拟过程的神经网络
        self.playout_num = playout_num  # mcts搜索次数

    # 完成一次mcts搜索（共四步）
    def mcts_playout(self, board_info):
        # 初始化节点
        node_obj = self.root_node
        # Select
        while True:
            # 根据Q+U遍历树节点直到叶子节点
            if node_obj.is_leaf_node():
                break
            move_id, node_obj = node_obj.select()  # 返回当前节点的最优子节点
            board_info.game_move(move_id)  # 更新棋盘列表
        # 神经网络给出当前局面(节点)的价值以及即将扩展的子节点的先验概率
        # sub_nodes_p_tuple == (move序号, 对应的先验概率), 局面评分:leaf_value ~ (-1,1)
        # Expand&Simulation
        subnode_p_list, leaf_node_value = self.neural_net(board_info)  # 子节点向量p(zip格式) 当前节点标量v
        result_id = board_info.gobang_rules()
        # print(board_info.is_current_player_black())
        # print(np.array(board_info.get_move_record()))
        # print(result_id)
        # print("----------------")
        # 如果不是平局就扩展节点
        if result_id == -1:
            node_obj.expand(subnode_p_list, add_dirichlet=config.ADD_DIRICHLET_FOR_EXPANSION)  # 扩展叶子节点的子节点(move,先验概率)
        # 如果分出胜负就
        else:
            if result_id == 3:  # 平局
                leaf_node_value = 0.0
            else:
                if config.REWARD_CUSTOM_OPTIONS:
                    # 若是黑棋胜利
                    if board_info.is_current_player_black() is False:
                        leaf_node_value = config.BLACK_WIN_SCORE if result_id == board_info.get_current_player_id() else config.WHITE_LOSE_SCORE
                    # 若是白棋胜利
                    elif board_info.is_current_player_black() is True:
                        leaf_node_value = config.WHITE_WIN_SCORE if result_id == board_info.get_current_player_id() else config.BlACK_LOSE_SCORE
                else:
                    leaf_node_value = 1.0 if result_id == board_info.get_current_player_id() else -1.0
        # Back-Upgrade
        # 扩展完或者select到达终局都需要反向递归更新所有祖先节点
        node_obj.back_recur(-leaf_node_value)

    # 通过num次playout后，获得根节点的子节点的访问次数
    def get_move_visit(self, board_info):
        for _ in range(self.playout_num):
            copy_board_info = copy.deepcopy(board_info)  # 独立操作棋盘需深拷贝
            self.mcts_playout(copy_board_info)  # 传入当前局面 而后mcts搜索1次
        # 得到root_node的children_node->move序号代表的节点和节点访问次数
        # [(act1,visit1),(act2,visit2)...]
        move_visits = [(move, node.visit_num) for move, node in self.root_node.children_node.items()]
        # move_list -> [act1, act2, act3 ...], visit_list -> [visit1, visit2 ,visit3 ...]
        move_list, visit_list = zip(*move_visits)  # 一一对应解压成[()]
        return move_list, visit_list

    # 人机对弈:重新建树;电脑自我对弈:重新建树但保留子树
    def rebuild_mct(self, last_move=None):
        # 自我对弈
        if last_move:
            self.root_node = self.root_node.children_node[last_move]
            self.root_node.parent_node = None
        # 人机对弈
        else:
            self.root_node = MCTNode(None, 1.0)

    # 若干playout完毕后在棋盘上进行真实的落子
    def choose_move(self, board_info):
        expand_nodes_pi_list = np.zeros(board_info.board_size * board_info.board_size)  # 创建扩展节点的概率集
        # 获得
        if board_info.aggregate_move_count < board_info.board_size * board_info.board_size:  # 获取当前棋盘上尚为空的位置
            move_list, visit_list = self.get_move_visit(board_info)  # move表示的节点; 节点访问次数
            # (1) 自我对弈模式
            if self.is_selfplay_mode:
                # >>1.采用AlphaGo Zero的温度逐渐变小的选择
                if config.IS_ALTERNATIVE_TEMPERATURE:
                    # 前FIRST_STEP_NUM步设τ = 1,增加探索程度
                    # 真实下棋采用温度tau随步数相应变化选取概率(pi)的方式
                    if board_info.board_size ** 2 - len(board_info.avail_move_list) <= config.FIRST_STEP_NUM:
                        temperature = 1.0
                        # 落子概率
                        pi_list = softmax_func(1.0 / temperature * np.log(np.array(visit_list) + 1e-10))
                        # 按pi_list概率分布 从move_list抽出一个move用以下棋
                        move = np.random.choice(move_list, p=pi_list)
                    else:
                        temperature = 1e-3
                        # 作为落子采用的概率
                        pi_list = softmax_func(1.0 / temperature * np.log(np.array(visit_list) + 1e-10))
                        move = np.random.choice(move_list, p=pi_list)
                    self.rebuild_mct(move)  # 自我对弈模式要在保留本节点的树分支的基础上抛弃父节点的树分支
                    # 作为训练用的pi标签 tau始终为1
                    pi_label = softmax_func(1.0 / 1.0 * np.log(np.array(visit_list) + 1e-10))
                    # move作为index 与pi一一对应
                    # 已被落子占用的位置则概率为0
                    expand_nodes_pi_list[list(move_list)] = pi_label
                    # print(expand_nodes_pi_list.reshape(8, 8))
                    return move, expand_nodes_pi_list
                # >>2.采用温度始终不变的选择
                else:
                    temperature = 1.0
                    pi_list = softmax_func(1.0 / temperature * np.log(np.array(visit_list) + 1e-10))
                    # 该选择将在落子抽取上增加dirichlet噪声
                    move = np.random.choice(move_list, p=(
                                                                     1 - config.DIRICHLET_WEIGHT) * pi_list + config.DIRICHLET_WEIGHT * np.random.dirichlet(
                        config.DIRICHLET_ALPHA * np.ones(len(pi_list))))
                    expand_nodes_pi_list[list(move_list)] = pi_list
                    # 以move为根节点建树
                    self.rebuild_mct(move)
                    # 还需返回所有的扩展节点的概率 expand_nodes_pi_list
                    return move, expand_nodes_pi_list
            # (2) 人机对弈/模型评估模式
            else:
                # 设置tau始终为1e-3
                temperature = 1e-3
                pi_list = softmax_func(1.0 / temperature * np.log(np.array(visit_list) + 1e-10))
                # print(move_list)
                # print(visit_list)
                move = np.random.choice(move_list, p=pi_list)
                expand_nodes_pi_list[list(move_list)] = pi_list
                # print(move)
                # print("-----")
                # 人机对弈则每落一子就重新建树
                self.rebuild_mct()
                return move
        else:
            print("棋盘已满!")

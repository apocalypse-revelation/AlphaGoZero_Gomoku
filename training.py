# -*- coding: utf-8 -*-
from __future__ import print_function  # support py2.x
import copy
import numpy as np
import config
from collections import defaultdict, deque
from basics import Foundation, Game
from mcts import MCTSPlayer
import resnet
import convnet
import time
import os
import pickle
from torch.utils.tensorboard import SummaryWriter

# TensorBoard可视化
# tensorboard --logdir=D:\graduation_project\GobangZero\visual_data\64f_2b_res_8size --port 1235
writer = SummaryWriter(config.VISUAL_DATA_PATH)


class TrainModel:
    def __init__(self, size, victory_num, model_path=None, net_type=None):
        self.board_size = size
        self.victory_num = victory_num  # x子棋
        self.board = Foundation(board_size=self.board_size, victory_num=self.victory_num)  # 先定义一个棋盘
        self.game = Game(self.board)  # 再定义一个操作棋盘信息的类
        # 存放数据集
        self.data_cache = deque(maxlen=config.DATASET_SIZE)  # 创建FIFO的数据集(队列) 一个state占8个size
        if net_type == 'resnet':
            self.net_func = resnet.NetFunction(self.board_size, model_path=model_path)
        elif net_type == 'cnn':
            self.net_func = convnet.NetFunction(self.board_size, model_path=model_path)
        else:
            print("请指定要训练的网络类型!")
            self.net_func = None
        self.mcts_player = MCTSPlayer(self.net_func.get_policy_value_for_mcts, playout_num=config.TRAIN_MCTS_PLYAOUT_NUM, is_selfplay_mode=True)

    # 自我对弈->收集对弈数据->把对弈数据用来扩充数据集
    def collect_data(self):
        # data -> [([states1],[pi_list1],[z1]), ([states2],[pi_list2],[z2]),...]
        data = self.game.self_play(self.mcts_player)
        one_game_data = copy.deepcopy(data)
        # 数据增强
        one_game_data = self.game.expand_data(one_game_data, self.board_size)  # [(局数0的翻转数据1),(局数0的翻转数据1),(局数0的翻转数据1)...]
        # data_cache[0/1/2/3] index代表一局游戏的所有数据 [(局数0的数据),(局数0的翻转数据),(局数1的数据)]
        self.data_cache.extend(one_game_data)
        return len(data), len(self.data_cache)

    # 评估模型
    def model_evaluate(self, latest_path, good_path):
        if config.TRAIN_WHICH_NET == 'resnet':
            latest_resnet_func = resnet.NetFunction(self.board_size, model_path=latest_path)
            good_resnet_func = resnet.NetFunction(self.board_size, model_path=good_path)
            latest_mcts_player = MCTSPlayer(latest_resnet_func.get_policy_value_for_mcts, playout_num=config.EVAL_MCTS_PLAYOUT_NUM)
            good_mcts_player = MCTSPlayer(good_resnet_func.get_policy_value_for_mcts, playout_num=config.EVAL_MCTS_PLAYOUT_NUM)
        elif config.TRAIN_WHICH_NET == 'cnn':
            latest_convnet_func = convnet.NetFunction(self.board_size, model_path=latest_path)
            good_convnet_func = convnet.NetFunction(self.board_size, model_path=good_path)
            latest_mcts_player = MCTSPlayer(latest_convnet_func.get_policy_value_for_mcts, playout_num=config.EVAL_MCTS_PLAYOUT_NUM)
            good_mcts_player = MCTSPlayer(good_convnet_func.get_policy_value_for_mcts, playout_num=config.EVAL_MCTS_PLAYOUT_NUM)
        else:
            print("无法评估模型!请先指定要训练的网络类型!")
            return
        # 计数
        latest_model_win_count, latest_model_tie_count , latest_model_lose_count = 0, 0, 0
        # 评估eval_num局
        for i in range(config.EVAL_NUM):
            winner_id = self.game.model_play(latest_mcts_player, good_mcts_player, round_num=i)
            if winner_id == 1:
                latest_model_win_count += 1
            elif winner_id == 2:
                latest_model_lose_count += 1
            elif winner_id == 3:
                latest_model_tie_count += 1
        # 胜利+1、平局+0.5
        latest_mcts_win_rate = 1.0 * (latest_model_win_count + 0.5 * latest_model_tie_count) / config.EVAL_NUM
        print("模型MCTS的搜索次数:{}, 最新模型的训练结果-->胜利: {}, 失败: {}, 平局:{}, 分值:{}".format(config.EVAL_MCTS_PLAYOUT_NUM, latest_model_win_count, latest_model_lose_count, latest_model_tie_count, latest_mcts_win_rate))
        return latest_mcts_win_rate

    def models_battle(self):
        if config.RUN_EVAL and os.path.exists(
                config.SAVE_GOOD_MODEL_PATH):
            print(">>>Start evaluating the latest model ...")
            win_ratio = self.model_evaluate(latest_path=config.SAVE_LATEST_MODEL_PATH,
                                            good_path=config.SAVE_GOOD_MODEL_PATH)
            print('Win_ratio: ', win_ratio)

    def start_training(self):
        # 读取本地棋谱数据
        if config.USE_LOCAL_DATASET:
            local_dataset = pickle.load(open(config.LOAD_LOCAL_DATASET_PATH, "rb"))
            self.data_cache = local_dataset
        for game_num in range(1, config.SELFPLAY_NUM):  # 持续几轮
            # 1) 自我对弈收集数据
            start_time = time.time()  # 计时
            episode_len, data_cache_len = self.collect_data()  # 1) 对弈一局->收集该局数据->数据增强->存入数据集
            end_time = time.time()
            cost_time = end_time-start_time
            print("自我对弈第{}局, 该局落子总数:{}, 数据集大小:{}, 该局对弈时间:{}秒".format(game_num, episode_len, data_cache_len, cost_time))
            writer.add_scalar('steps_per_game', episode_len, game_num)  # tensorboard
            if config.SAVE_SELF_PLAY_DATA:
                pickle.dump(self.data_cache, open(config.SAVE_LOCAL_DATASET_PATH, 'wb'))  # 重复覆盖
                print("已完成一局对弈, 当前所有对弈数据已保存至本地")
            # 2) 训练模型
            if len(self.data_cache) > config.DATASET_SIZE_UPPER_LIMIT:   # 数据集装多大才开始训练
                aggregate_loss, mse_loss, cross_entropy_loss = self.net_func.training(self.data_cache)
                print("aggregate_loss:{}, " "mse_loss:{}, " "cross_entropy_loss:{}".format(aggregate_loss, mse_loss, cross_entropy_loss))
                writer.add_scalar('aggregate_loss', aggregate_loss, game_num)  # tensorboard
                writer.add_scalar('cross_entropy_loss', cross_entropy_loss, game_num)  # tensorboard
            # 保存临时模型用于评估（每次训练开始就保存）
            if config.RUN_EVAL and os.path.exists(config.SAVE_GOOD_MODEL_PATH) is False:
                self.net_func.save_model(config.SAVE_GOOD_MODEL_PATH)  # 第一次训练就保存good_model
            # 保存当前模型
            if game_num % config.SAVE_MODEL_FRENQUENCY == 0:
                self.net_func.save_model(config.SAVE_LATEST_MODEL_PATH)
                print(">>>已保存当前模型")
            # 3) 评估当前模型
            if game_num % config.EVAL_MODEL_FRENQUENCY == 0 and config.RUN_EVAL and os.path.exists(config.SAVE_GOOD_MODEL_PATH):
                print(">>>开始对当前模型进行评估...")
                win_ratio = self.model_evaluate(latest_path=config.SAVE_LATEST_MODEL_PATH, good_path=config.SAVE_GOOD_MODEL_PATH)
                if win_ratio >= config.EVAL_WIN_RATE_THRESHOLD:
                    self.net_func.save_model(config.SAVE_GOOD_MODEL_PATH)  # 覆盖之前的good_model
                    print(">>>当前模型强度增强明显 已保存为最好模型")
                else:
                    print(">>>当前模型强度增强不明显")


# 开始训练
if __name__ == '__main__':
    training_process = TrainModel(size=config.TRAIN_BOARD_SIZE, victory_num=config.TRAIN_BOARD_VICTORY_NUM, model_path=config.EXISTING_MODEL_PATH, net_type=config.TRAIN_WHICH_NET)
    training_process.models_battle()

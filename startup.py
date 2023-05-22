# -*- coding: utf-8 -*-
from mcts import MCTSPlayer
import resnet
import convnet
import basics
import config


if __name__ == "__main__":
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>GUI界面<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    resnet_model_path = config.AI_RESNET_MODEL_PATH  # resnet模型的参数路径
    cnn_model_path = config.AI_CNN_MODEL_PATH  # 普通cnn模型的参数路径
    # 初始化棋盘数据(棋盘大小, 输赢判断等)
    board_obj = basics.Foundation(board_size=config.GUI_BOARD_SIZE, victory_num=config.GUI_BOARD_VICTORY_NUM)
    # 初始化游戏数据(gui界面实现,人机对弈,自我训练等)
    game_obj = basics.Game(board_obj)
    # 创建神经网络用于辅助AI的mcts的部分搜索过程（二选一）
    if config.AI_NET_TYPE == 'cnn':
        net = convnet.NetFunction(config.GUI_BOARD_SIZE, model_path=cnn_model_path)  # 普通cnn网络
    elif config.AI_NET_TYPE == 'resnet':
        net = resnet.NetFunction(config.GUI_BOARD_SIZE, model_path=resnet_model_path)  # resnet网络
    else:
        net = None
    # 创建AI
    mcts_player = MCTSPlayer(net.get_policy_value_for_mcts, playout_num=config.AI_MCTS_PLAYOUT_NUM)
    # 开启GUI画面
    game_obj.start_game(mcts_player)


# -*- coding: utf-8 -*-

from __future__ import print_function
import numpy as np
from tkinter import *
from tkinter.messagebox import *
from PIL import Image, ImageTk
import time
import os
import config
import pickle
import copy
from collections import deque


class Foundation(object):
    def __init__(self, board_size, victory_num):
        self.board_size = board_size  # 棋盘尺寸
        self.victory_num = victory_num  # n子棋
        self.aggregate_move_count = 0  # 总棋子数
        self.avail_move_list = [i for i in range(self.board_size ** 2)]  # 一维,可落子位置信息,初始可落子序号
        self.all_move_list = []  # 下过的棋子(序号)
        self.all_player_id_list = []  # 落子玩家序号记录
        self.move_record = [[0] * self.board_size for i in range(self.board_size)]  # 二维,棋盘信息,用于判断胜负
        self.current_player_is_black = None  # 当前将要落子的玩家是否是黑棋

    # 重置棋盘信息
    def build_board(self, who_first='player1'):
        self.current_player_id = 1 if who_first == 'player1' else 2  # 设定第(1/2)个玩家为先手, 用于人机对战设置
        self.current_player_is_black = True
        self.move_record = [[0] * self.board_size for i in range(self.board_size)]  # 用于判断胜负
        self.avail_move_list = [i for i in range(self.board_size ** 2)]  # 初始可落子序号
        self.all_move_list = []  # 下过的棋子(序号)
        self.all_player_id_list = []  # 落子玩家序号记录
        # 初始当前棋盘落子数
        self.aggregate_move_count = 0

    # 落子位置序号 -> 棋局矩阵坐标
    def move2location(self, move_id):
        y = move_id // self.board_size
        x = move_id % self.board_size
        return y, x

    # 获取当前棋局局面的特征平面
    def get_feature_planes(self):
        # 返回当前棋盘状态的?个不同的特征平面（? depth）用于输入神经网络
        feature_plane = np.full((config.FEATURE_PLANE_NUM, self.board_size, self.board_size), 0.0)  # 定义4个平面(三维)
        if len(self.all_move_list) > 0:
            # 所有move+所有交替玩家的id 取得两个玩家各自的move_id(list都必须转为np)
            all_move = np.array(self.get_all_move_list())
            player1_move = all_move[np.array(self.get_all_player_id_list()) != self.current_player_id]
            player2_move = all_move[np.array(self.get_all_player_id_list()) == self.current_player_id]
            # 由落子顺序和玩家顺序 获取特征平面
            # 特征平面1: 当前将要落子的玩家的所有落子位置
            feature_plane[0][
                self.board_size - player2_move // self.board_size - 1, player2_move % self.board_size] = 1.0
            # 特征平面2：当前将要落子的玩家的对手的所有落子位置
            feature_plane[1][
                self.board_size - player1_move // self.board_size - 1, player1_move % self.board_size] = 1.0
            # 特征平面3：当前将要落子的玩家的对手的历史位置（对手的上1步）
            feature_plane[2][
                self.board_size - self.get_last_move_id() // self.board_size - 1, self.get_last_move_id() % self.board_size] = 1.0
            # feature_plane[4][self.board_size - self.latest_move // self.board_size - 1, self.latest_move % self.board_size] = 1.0
            # 特征平面4: 当前将要落子的是否是黑棋？是则棋盘全为1，否则为0；
            feature_plane[3] = 1.0 if len(self.all_move_list) % 2 == 0 else 0.0
        # 返回当前棋局状态所表征的特征平面
        return feature_plane

    # 更新棋盘数组 再返回给gui显示
    def game_move(self, move):
        self.aggregate_move_count += 1  # 记录总步数 用于gui
        # print(len(self.all_move_list))
        # 下一步棋就从可落子数组移除当前下的那个
        self.avail_move_list.remove(move)
        self.all_move_list.append(move)
        self.all_player_id_list.append(self.current_player_id)
        # 获得落子序号的棋盘坐标
        loc_y, loc_x = self.move2location(move)
        # 记录两个玩家的落子坐标 1人类 2电脑
        self.move_record[loc_y][loc_x] = self.current_player_id
        # for a in range(self.width):
        #     print(self.move_record[a])
        # 一方落子后 更新当前将要落子玩家序号
        self.current_player_id = 1 if self.current_player_id == 2 else 2
        self.current_player_is_black = False if self.current_player_is_black is True else True
        # print(self.is_current_player_black(), self.current_player_id, self.gobang_rules())
        # print()
        # print(self.is_current_player_black())

    # 通过最后的落子来判断输赢
    def gobang_rules(self):
        size = self.board_size
        n = self.victory_num
        row, col = self.move2location(self.get_last_move_id())  # 获取最新一步落子的坐标
        player1_stone_count = 0  # 玩家的最大连珠计数
        player2_stone_count = 0  # 电脑的最大连珠计数

        who_win = None
        # 水平检测
        for length in range(-n + 1, n):
            if 0 <= col + length <= size - 1:
                if self.move_record[row][col + length] == 1:
                    player1_stone_count += 1
                    player2_stone_count = 0
                elif self.move_record[row][col + length] == 2:
                    player2_stone_count += 1
                    player1_stone_count = 0
                else:
                    player2_stone_count = 0
                    player1_stone_count = 0
                    continue
                if player1_stone_count == n:
                    who_win = 1
                    return who_win
                if player2_stone_count == n:
                    who_win = 2
                    return who_win

        # 初始化计数,否则出错
        player1_stone_count = 0
        player2_stone_count = 0
        # 垂直检测
        for length in range(-n + 1, n):
            if 0 <= row + length <= size - 1:
                if self.move_record[row + length][col] == 1:
                    player1_stone_count += 1
                    player2_stone_count = 0
                elif self.move_record[row + length][col] == 2:
                    player2_stone_count += 1
                    player1_stone_count = 0
                else:
                    player2_stone_count = 0
                    player1_stone_count = 0
                    continue
                if player1_stone_count == n:
                    who_win = 1
                    return who_win
                if player2_stone_count == n:
                    who_win = 2
                    return who_win

        # 初始化计数,否则出错
        player1_stone_count = 0
        player2_stone_count = 0
        # 右斜检测
        for length in range(-n + 1, n):
            if 0 <= row - length <= size - 1 and 0 <= col + length <= size - 1:
                if self.move_record[row - length][col + length] == 1:
                    player1_stone_count += 1
                    player2_stone_count = 0
                elif self.move_record[row - length][col + length] == 2:
                    player2_stone_count += 1
                    player1_stone_count = 0
                else:
                    player2_stone_count = 0
                    player1_stone_count = 0
                    continue
                if player1_stone_count == n:
                    who_win = 1
                    return who_win
                if player2_stone_count == n:
                    who_win = 2
                    return who_win

        # 初始化计数,否则出错
        player1_stone_count = 0
        player2_stone_count = 0
        # 左斜检测
        for length in range(-n + 1, n):
            if 0 <= row + length <= size - 1 and 0 <= col + length <= size - 1:
                if self.move_record[row + length][col + length] == 1:
                    player1_stone_count += 1
                    player2_stone_count = 0
                elif self.move_record[row + length][col + length] == 2:
                    player2_stone_count += 1
                    player1_stone_count = 0
                else:
                    player2_stone_count = 0
                    player1_stone_count = 0
                    continue
                if player1_stone_count == n:
                    who_win = 1
                    return who_win
                if player2_stone_count == n:
                    who_win = 2
                    return who_win

        # 平局判断
        if self.aggregate_move_count == size ** 2:
            who_win = 3
        # 未分出胜负判断
        else:
            who_win = -1
        return who_win

    # 获取当前将要落子的玩家id
    def get_current_player_id(self):
        return self.current_player_id

    def get_all_move_list(self):
        return self.all_move_list

    def get_all_player_id_list(self):
        return self.all_player_id_list

    # 获得倒数第n步的落子序号
    def get_last_move_id(self, n=1):
        return self.all_move_list[-n] if n <= len(self.all_move_list) else -1

    # 获取当前将要落子的玩家是黑棋还是白棋
    def is_current_player_black(self):
        return self.current_player_is_black

    def get_move_record(self):
        return self.move_record


class Game(object):

    def __init__(self, board_info):
        self.board_info = board_info
        self.who_first = None
        self.allow_human_click = False
        self.states_list_human = []  # 用于人类保存棋谱局面数据
        self.pi_list_human = []  # 用于人类保存棋谱pi数据

    # 定义GUI

    # 画棋盘和网格
    def gui_draw_cross(self, x, y):
        cross_scale = 1.5  # 交叉轴"+"的长度
        # 边界坐标
        screen_x, screen_y = self.standard_size * (x + 1), self.standard_size * (y + 1)
        # 画棋盘(a,b,c,d) -> (a,b)左上角坐标 (c,d)右下角坐标
        self.gui_board.create_rectangle(screen_y - self.cross_size, screen_x - self.cross_size,
                                        screen_y + self.cross_size, screen_x + self.cross_size,
                                        fill=self.board_color, outline=self.board_color)
        # 生成交叉点
        # 棋盘边缘的交叉点的x/y需设为0 其他情况就直接返回具体xyZ坐标
        hor_m, hor_n = [0, cross_scale] if y == 0 else [-cross_scale, 0] if y == self.board_info.board_size - 1 else [
            -cross_scale, cross_scale]
        ver_m, ver_n = [0, cross_scale] if x == 0 else [-cross_scale, 0] if x == self.board_info.board_size - 1 else [
            -cross_scale, cross_scale]
        # 画横线
        self.gui_board.create_line(screen_y + hor_m * self.cross_size, screen_x, screen_y + hor_n * self.cross_size,
                                   screen_x)
        # 画竖线
        self.gui_board.create_line(screen_y, screen_x + ver_m * self.cross_size, screen_y,
                                   screen_x + ver_n * self.cross_size)

    # 画棋子
    def gui_draw_stone(self, x, y, current_stone_color):
        screen_x, screen_y = self.standard_size * (x + 1), self.standard_size * (y + 1)
        # print(screen_x, screen_y)
        if self.board_info.aggregate_move_count == 0:
            self.gui_board.create_oval(screen_y - self.stone_size, screen_x - self.stone_size,
                                       screen_y + self.stone_size, screen_x + self.stone_size,
                                       fill=current_stone_color)
            # 当前落子的编号标红
            self.gui_board.create_text(screen_y, screen_x, text=self.board_info.aggregate_move_count + 1, fill='red')
            self.previous_screen_x, self.previous_screen_y = screen_x, screen_y

        elif self.board_info.aggregate_move_count != 0:
            self.gui_board.create_oval(screen_y - self.stone_size, screen_x - self.stone_size,
                                       screen_y + self.stone_size, screen_x + self.stone_size,
                                       fill=current_stone_color)
            # 当前落子的编号标红
            self.gui_board.create_text(screen_y, screen_x, text=self.board_info.aggregate_move_count + 1, fill='red')
            # 从第二步开始,上一步棋子从红色恢复原本颜色
            self.gui_board.create_oval(self.previous_screen_y - self.stone_size,
                                       self.previous_screen_x - self.stone_size,
                                       self.previous_screen_y + self.stone_size,
                                       self.previous_screen_x + self.stone_size,
                                       fill='black' if current_stone_color == 'white' else 'white')
            self.gui_board.create_text(self.previous_screen_y, self.previous_screen_x,
                                       text=self.board_info.aggregate_move_count,
                                       fill='white' if current_stone_color == 'white' else 'black')
            self.previous_screen_x, self.previous_screen_y = screen_x, screen_y

    # 画棋盘
    def gui_draw_board(self):
        # 根据棋盘尺寸大小循环
        [self.gui_draw_cross(x, y) for y in range(self.board_info.board_size) for x in
         range(self.board_info.board_size)]

    def gui_opt_human_start_btn(self):
        self.two_human_play_mode = True
        self.who_first = 'player1'  # 即设定玩家1为先手
        self.allow_human_click = True
        self.gui_draw_board()  # 重置界面
        self.label_tips.config(text="黑棋回合")
        self.board_info.build_board(self.who_first)  # 即result_id或id=1为黑棋 反之=2为白棋
        self.btn_opt_human_play_save.config(state=DISABLED)
        self.states_list_human = []
        self.pi_list_human = []
        self.states_list_human.append(self.board_info.get_feature_planes())  # 每个ndarr类型的局面的4个特征平面(3d)存入?_list

    def gui_opt_human_save_btn(self):
        # 保存对弈的棋谱并作数据增强
        data = list(zip(self.states_list_human, self.pi_list_human, self.z_list_human))
        one_game_data = copy.deepcopy(data)
        one_game_data = self.expand_data(one_game_data, self.board_size)
        data_cache = deque(maxlen=config.DATASET_SIZE)
        data_cache.extend(one_game_data)
        pickle.dump(data_cache, open('./tmp_dataset/human_data.pkl', 'wb'))
        self.btn_opt_human_play_save.config(state=DISABLED)
        self.states_list_human = []
        self.pi_list_human = []
        return

    # 初始化界面信息（选择黑色）
    def gui_opt_black_btn(self):
        self.two_human_play_mode = False
        self.who_first = 'player1'  # 即设定玩家1为先手
        self.allow_human_click = True
        self.gui_draw_board()  # 重置界面
        self.label_tips.config(text="黑棋回合")
        self.board_info.build_board(who_first=self.who_first)
        self.btn_opt_human_play_save.config(state=DISABLED)
        self.states_list_human = []
        self.pi_list_human = []

    # 初始化界面信息（选择白色）
    def gui_opt_white_btn(self):
        self.two_human_play_mode = False
        self.who_first = 'player2'  # 即设定玩家2为先手
        self.allow_human_click = False
        self.gui_draw_board()  # 重置界面
        self.label_tips.config(text="黑棋回合")
        self.gui_board.update()
        self.board_info.build_board(who_first=self.who_first)  # 人类选白棋， 初始化棋盘
        self.btn_opt_human_play_save.config(state=DISABLED)
        self.states_list_human = []
        self.pi_list_human = []
        current_player_id = self.board_info.get_current_player_id()  # 获取当前将要落子的玩家id
        # 人类选择白棋按钮后,让电脑先走一步后再进入鼠标左键的绑定事件
        if current_player_id == 2:  # 电脑的id都被设置为2
            alternate_player_obj = self.ai_obj
            mcts_move = alternate_player_obj.choose_move(self.board_info)
            gui_pos_row, gui_pos_col = self.board_info.move2location(mcts_move)
            self.gui_draw_stone(gui_pos_row, gui_pos_col, 'black')
            self.board_info.game_move(mcts_move)
            self.gui_board.update()
            self.label_tips.config(text="白棋回合")
            self.allow_human_click = True

    # 棋局结束在canvas中间显示胜负提醒
    def gui_draw_center_text(self, text):
        width, height = int(self.gui_board['width']), int(self.gui_board['height'])
        self.gui_board.create_text(int(width / 2), int(height / 2), text=text, font=("黑体", 30, "bold"), fill="red")

    # 棋盘Canvas的click事件
    def gui_click_board(self, event):
        # 获取鼠标点击的canvas坐标并用int强制转化为网格坐标
        gui_pos_row, gui_pos_col = int((event.y - self.cross_size) / self.standard_size), int(
            (event.x - self.cross_size) / self.standard_size)
        # 防止超出边界
        if gui_pos_row >= self.board_size or gui_pos_col >= self.board_size:
            return
        # 防止其他禁止情况却能落子
        if not self.allow_human_click:
            return
        # 鼠标的点击坐标转化为 具体落子数字的格式
        self.human_move = gui_pos_row * self.board_size + gui_pos_col

        # 已经下过的地方不能再下
        if self.human_move not in self.board_info.avail_move_list:
            return
        current_player_id = self.board_info.get_current_player_id()  # 当前将要落子的玩家id
        # 人机对弈:玩家走子 / 双人对战:黑棋走子
        if current_player_id == 1 and self.two_human_play_mode is False:
            if self.human_move in self.board_info.avail_move_list:
                # 人类选择的先后手判断
                if self.who_first == 'player1':
                    self.gui_draw_stone(gui_pos_row, gui_pos_col, 'black')
                    self.label_tips.config(text="白棋回合")
                elif self.who_first == 'player2':
                    self.gui_draw_stone(gui_pos_row, gui_pos_col, 'white')
                    self.label_tips.config(text="黑棋回合")
                self.allow_human_click = False if self.two_human_play_mode is False else True
                self.board_info.game_move(self.human_move)
                self.gui_board.update()
                winner_id = self.board_info.gobang_rules()  # 1 2 3 -1
                if winner_id in (1, 2, 3):
                    self.allow_human_click = False
                    if winner_id != 3:
                        text = "人类获胜"
                        self.gui_draw_center_text(text)
                        self.label_tips.config(text="等待中")
                        return  # return用来防止进入 current_player_id == 2的代码块
                    else:
                        text = "平局"
                        self.gui_draw_center_text(text)
                        self.label_tips.config(text="等待中")
                        return
        # 双人对战
        if current_player_id == 1 and self.two_human_play_mode is True:
            if self.human_move in self.board_info.avail_move_list:
                # 人类选择的先后手判断
                if self.who_first == 'player1':
                    self.gui_draw_stone(gui_pos_row, gui_pos_col, 'black')
                    self.label_tips.config(text="白棋回合")
                elif self.who_first == 'player2':
                    self.gui_draw_stone(gui_pos_row, gui_pos_col, 'white')
                    self.label_tips.config(text="黑棋回合")
                self.board_info.game_move(self.human_move)
                self.gui_board.update()
                # 存储棋谱数据pi
                y, x = self.board_info.move2location(self.human_move)
                pi_human = np.zeros((self.board_info.board_size, self.board_info.board_size))
                pi_human[y, x] = 1.0
                self.pi_list_human.append(pi_human.flatten())
                winner_id = self.board_info.gobang_rules()  # 1 2 3 -1
                if winner_id in (1, 2, 3):
                    self.allow_human_click = False
                    self.z_list_human = np.zeros(self.board_info.aggregate_move_count)
                    if winner_id == 1:
                        self.allow_human_click = False
                        text = "黑棋获胜"
                        self.gui_draw_center_text(text)
                        self.label_tips.config(text="等待中")
                        self.btn_opt_human_play_save.config(state=NORMAL)
                        self.z_list_human[np.array(self.board_info.get_all_player_id_list()) == winner_id] = 1.0
                        self.z_list_human[np.array(self.board_info.get_all_player_id_list()) != winner_id] = -1.0
                        return
                    elif winner_id == 3:
                        self.allow_human_click = False
                        text = "平局"
                        self.gui_draw_center_text(text)
                        self.label_tips.config(text="等待中")
                        self.btn_opt_human_play_save.config(state=NORMAL)
                        return

        # 人机对弈:AI走子
        current_player_id = self.board_info.get_current_player_id()
        if current_player_id == 2 and self.two_human_play_mode is False:
            alternate_player_obj = self.ai_obj
            mcts_move = alternate_player_obj.choose_move(self.board_info)
            mcts_location = self.board_info.move2location(mcts_move)
            gui_pos_row, gui_pos_col = mcts_location[0], mcts_location[1]
            # 你选择先后手按钮 电脑相应需要改变
            if self.who_first == 'player1':
                self.gui_draw_stone(gui_pos_row, gui_pos_col, 'white')
                self.label_tips.config(text="黑棋回合")
            elif self.who_first == 'player2':
                self.gui_draw_stone(gui_pos_row, gui_pos_col, 'black')
                self.label_tips.config(text="白棋回合")
            self.board_info.game_move(mcts_move)
            self.gui_board.update()
            self.allow_human_click = True
            winner_id = self.board_info.gobang_rules()
            if winner_id in (1, 2, 3):
                self.allow_human_click = False
                if winner_id != 3:
                    text = "电脑获胜"
                    self.gui_draw_center_text(text)
                    self.label_tips.config(text="等待中")
                    return
                else:
                    text = "平局"
                    self.gui_draw_center_text(text)
                    self.label_tips.config(text="等待中")
                    return
        # 双人对战:白棋走子
        elif current_player_id == 2 and self.two_human_play_mode is True:
            if self.human_move in self.board_info.avail_move_list:
                # 人类选择的先后手判断
                if self.who_first == 'player1':
                    self.gui_draw_stone(gui_pos_row, gui_pos_col, 'white')
                    self.label_tips.config(text="黑棋回合")
                elif self.who_first == 'player2':
                    self.gui_draw_stone(gui_pos_row, gui_pos_col, 'black')
                    self.label_tips.config(text="白棋回合")
                self.board_info.game_move(self.human_move)
                self.gui_board.update()
                # 存储棋谱数据pi
                y, x = self.board_info.move2location(self.human_move)
                pi_human = np.zeros((self.board_info.board_size, self.board_info.board_size))
                pi_human[y, x] = 1.0  # 走该落子位置的pi为1.0
                self.pi_list_human.append(pi_human.flatten())
                winner_id = self.board_info.gobang_rules()  # 1 2 3 -1
                if winner_id in (1, 2, 3):
                    self.z_list_human = np.zeros(self.board_info.aggregate_move_count)
                    if winner_id == 2:
                        self.allow_human_click = False
                        text = "白棋获胜"
                        self.gui_draw_center_text(text)
                        self.label_tips.config(text="等待中")
                        self.btn_opt_human_play_save.config(state=NORMAL)
                        self.z_list_human[np.array(self.board_info.get_all_player_id_list()) == winner_id] = 1.0
                        self.z_list_human[np.array(self.board_info.get_all_player_id_list()) != winner_id] = -1.0
                        return
                    elif winner_id == 3:
                        self.allow_human_click = False
                        text = "平局"
                        self.gui_draw_center_text(text)
                        self.label_tips.config(text="等待中")
                        self.btn_opt_human_play_save.config(state=NORMAL)
                        return  # return用来防止进入 current_player_id == 2的代码块
        # print(self.board_info.get_all_player_id_list())
        # print(self.board_info.get_all_move_list())
        # 存储棋谱数据state
        if self.two_human_play_mode is True:
            self.states_list_human.append(self.board_info.get_feature_planes())

    # gui退出
    @staticmethod
    def gui_quit_game():
        os._exit(1)

    # 界面
    def gui(self, board_info, player2_obj):
        self.board_size = board_info.board_size
        # 设置先手是玩家1还是玩家2（player?_obj)
        self.ai_obj = player2_obj
        self.two_human_play_mode = False  # 确定是人机对弈还是双人对战
        # gui参数定义
        sidebar_color = "Moccasin"  # 侧边栏颜色
        btn_font = ("黑体", 12, "bold")  # 按钮文字样式
        self.standard_size = 40  # 设置标准尺寸
        self.board_color = "Tan"  # 棋盘颜色
        self.cross_size = self.standard_size / 2  # 交叉轴大小
        self.stone_size = self.standard_size / 3  # 棋子大小
        self.allow_human_click = False  # 是否允许人类玩家点击棋盘

        # gui初始化（tkinter)
        root = Tk()
        root.title("五子棋")
        root.resizable(width=False, height=False)  # 窗口大小不允许拉动
        # 布局-定义
        gui_sidebar = Frame(root, highlightthickness=0, bg=sidebar_color)
        gui_sidebar.pack(fill=BOTH, ipadx=10, side=RIGHT)  # ipadx 加宽度padding
        btn_opt_black = Button(gui_sidebar, text="选择黑色", command=self.gui_opt_black_btn, font=btn_font)
        btn_opt_white = Button(gui_sidebar, text="选择白色", command=self.gui_opt_white_btn, font=btn_font)
        btn_opt_human_play_start = Button(gui_sidebar, text="开始游戏", command=self.gui_opt_human_start_btn, font=btn_font)
        self.btn_opt_human_play_save = Button(gui_sidebar, text="保存棋局", command=self.gui_opt_human_save_btn, font=btn_font, state=DISABLED)
        btn_opt_quit = Button(gui_sidebar, text="退出游戏", command=self.gui_quit_game, font=btn_font)
        self.label_tips = Label(gui_sidebar, text="等待中", bg=sidebar_color, font=("黑体", 18, "bold"), fg="red4")
        two_human_play_label = Label(gui_sidebar, text="双人对战", bg=sidebar_color, font=("楷体", 12, "bold"))
        machine_man_play_label = Label(gui_sidebar, text="人机对战", bg=sidebar_color, font=("楷体", 12, "bold"))
        # 布局-显示
        two_human_play_label.pack(side=TOP, padx=20, pady=5)
        btn_opt_human_play_start.pack(side=TOP, padx=20, pady=10)
        self.btn_opt_human_play_save.pack(side=TOP, padx=20, pady=10)
        machine_man_play_label.pack(side=TOP, padx=20, pady=10)
        btn_opt_black.pack(side=TOP, padx=20, pady=5)
        btn_opt_white.pack(side=TOP, padx=20, pady=10)
        btn_opt_quit.pack(side=BOTTOM, padx=20, pady=10)
        self.label_tips.pack(side=TOP, expand=YES, fill=BOTH, pady=10)
        self.gui_board = Canvas(root, bg=self.board_color, width=(self.board_size + 1) * self.standard_size,
                                height=(self.board_size + 1) * self.standard_size, highlightthickness=0)
        self.gui_draw_board()  # 初始化棋盘
        self.gui_board.pack()
        self.gui_board.bind("<Button-1>", self.gui_click_board)  # 绑定左键事件
        root.mainloop()  # 事件循环

    # 定义人机对战/电脑自我对弈训练

    # 打开gui界面进行人机对战和双人对战
    def start_game(self, ai_obj):
        self.gui(self.board_info, ai_obj)

    # 模型评估对战
    def model_play(self, latest_obj1, good_obj2, round_num):
        # 轮流设置先后手
        who_black, who_white = ['player1', 'player2'] if round_num % 2 == 0 else ['player2', 'player1']  # 交替先后手
        # 初始化棋盘信息
        self.board_info.build_board(who_black)
        # 两个模型对弈
        while True:
            mcts_player = latest_obj1 if who_black == 'player1' else good_obj2  # 玩家交替
            print(who_black)
            move_id = mcts_player.choose_move(self.board_info)  # mcts搜索
            self.board_info.game_move(move_id)  # 更新棋盘信息
            # 每落子一次就check一次
            result_id = self.board_info.gobang_rules()
            who_black = 'player2' if who_black == 'player1' else 'player1'
            # 1=latest 2=good 3=tie
            if result_id in (1, 2, 3):
                return result_id
            

    # 自我对弈：生成一局棋谱-->状态集S、走子概率集Pi、价值集Z-->数据增强
    def self_play(self, player_obj):
        # 每开一局都要初始化棋盘信息
        self.board_info.build_board()
        # S、Pi、Player_id
        states_list = []  # 一局游戏里的所有棋局状态集
        mcts_pi_list = []  # 一局游戏里的所有局面对应的每个落子位置的概率分布pi(由温度参数给出)
        # 电脑自我对弈
        while True:
            # 当前棋局局面送入mcts+nn 而后输出具体落子位置和落子概率
            move_id, pi = player_obj.choose_move(self.board_info)
            # 下一步棋就存起来 (pi.reshape 与state相反)
            states_list.append(self.board_info.get_feature_planes())  # 每个ndarr类型的局面的4个特征平面(3d)存入?_list
            mcts_pi_list.append(pi)  # 通过多次搜索（playout）后由softmax+tau得出扩展的子节点及其选择概率
            # 更新棋盘数组 记录落子
            self.board_info.game_move(move_id)
            # 每下一步check一次胜负
            result_id = self.board_info.gobang_rules()
            # 分出胜负： 1=player1 2=player2 3=tie
            if result_id in (1, 2, 3):
                # 下完一局后，记录每个状态下的z值
                z_list = np.zeros(self.board_info.aggregate_move_count)  # 一局一共走了多少步, 创建步数记录列表z_list
                # 不是平局:赋对应的胜负reward; 平局:赋全0
                if result_id != 3:
                    # 强化学习的奖惩原理, 主要考虑到无禁手的不平衡性, 可以自定义reward
                    if config.REWARD_CUSTOM_OPTIONS:
                        # 若是黑棋胜利
                        if self.board_info.is_current_player_black is False:
                            # player_id必须用np.array()处理,否则只会赋-1.0,而1.0不会被赋
                            z_list[np.array(self.board_info.get_all_player_id_list()) == result_id] = config.BLACK_WIN_SCORE  # 黑棋赢了
                            z_list[np.array(self.board_info.get_all_player_id_list()) != result_id] = config.WHITE_LOSE_SCORE  # 白棋输了
                        # 若是白棋胜利
                        elif self.board_info.is_current_player_black is True:
                            z_list[np.array(self.board_info.get_all_player_id_list()) == result_id] = config.WHITE_WIN_SCORE  # 白棋赢了
                            z_list[np.array(self.board_info.get_all_player_id_list()) != result_id] = config.BlACK_LOSE_SCORE  # 黑棋输了要扣更多分
                    else:  # 依据Zero围棋的reward
                        z_list[np.array(self.board_info.get_all_player_id_list()) == result_id] = config.NORMAL_SCORE
                        z_list[np.array(self.board_info.get_all_player_id_list()) != result_id] = -config.NORMAL_SCORE
                # 打包为(S、Pi、Z)一一对应的列表
                one_game_data = list(zip(states_list, mcts_pi_list, z_list))
                return one_game_data

    # 数据增强
    def expand_data(self, one_game_data, board_size):
        # play_data: [(state, pi, z), ..., ...]
        data_extension = []
        # mcts_pi -> 长度为size**2的向量
        # s/pi上下相反
        # 一局游戏包含的若干局面的3个数据 np.arr / np.arr / np.float64
        for one_state, mcts_pi, value_z in one_game_data:  # 遍历一局游戏的所有局面的数据(s,pi,v)
            # （1）正常翻转4次
            for i in range(1, 5):
                origin_rot_state_planes = np.array([np.rot90(one_plane, k=i) for one_plane in one_state])
                origin_rot_pi = np.rot90(np.flipud(np.reshape(mcts_pi, (board_size, board_size))), k=i)
                origin_one_state_data = (origin_rot_state_planes, np.flipud(origin_rot_pi).flatten(), value_z)  # pi多转了一次。。
                data_extension.append(origin_one_state_data)
            # （2）镜像
            mirror_state_planes = np.array([np.fliplr(one_plane) for one_plane in one_state])
            mirror_pi = np.fliplr(np.flipud(np.reshape(mcts_pi, (board_size, board_size))))  # 先取反与state一致
            # 镜像翻转4次
            for i in range(1, 5):
                mirror_rot_state_planes = np.array([np.rot90(one_plane, k=i) for one_plane in mirror_state_planes])
                mirror_rot_pi = np.rot90(mirror_pi, k=i)
                mirror_one_state_data = (mirror_rot_state_planes, np.flipud(mirror_rot_pi).flatten(), value_z)  # pi多转了一次。。
                data_extension.append(mirror_one_state_data)
        return data_extension

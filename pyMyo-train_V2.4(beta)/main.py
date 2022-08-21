import pygame
import os, sys
import random
import csv
import time
from myoManager import MyoManager, EventType
from collections import deque
from PyQt5.QtWidgets import QWidget
from pyMyoMain import Win
# from test import ControlStage, Data, coefficient, EstimatePosition, EstimateStiffness, filter
from PyQt5.QtCore import QThread, pyqtSignal, QTimer
from PyQt5.QtWidgets import QApplication
import numpy as np

#读取信号 beta
app = QApplication(sys.argv)
win = Win()
win.show()

# Constants 常量
W, H = 1020, 710
FPS = 30
i = 0



# Setup 设置
pygame.init()
SCREEN = pygame.display.set_mode((1020,710))
pygame.display.set_caption('Flappy Bird --余翊旻')
CLOCK = pygame.time.Clock()

# material 素材
IMAGES = {}
for image in os.listdir('assets/sprites'):
    name, extension = os.path.splitext((image))   # 分割文件名和后缀
    path = os.path.join('assets/sprites', image)  # 拼接
    IMAGES[name] = pygame.image.load(path)

FLOOR_Y = H - IMAGES['floor'].get_height()

AUDIO = {}
for audio in os.listdir('assets/audio'):
    name, extension = os.path.splitext((audio))   # 分割文件名和后缀
    path = os.path.join('assets/audio', audio)  # 拼接
    AUDIO[name] = pygame.mixer.Sound(path)

# def main():
#     while True:
#         temp = Data()
#
#         if(len(temp.emg_data_queue) == 8):
#             ControlStage()
#         AUDIO['start'].play()
#         IMAGES['bgpic'] = IMAGES['day']
#         color = random.choice(['red', 'yellow', 'blue'])
#         IMAGES['birds'] = [IMAGES[color+'-up'], IMAGES[color+'-mid'], IMAGES[color+'-down']]
#         pipe = IMAGES[random.choice(['green-pipe', 'red-pipe'])]
#         IMAGES['pipes'] = [pipe, pygame.transform.flip(pipe, False, True)]
#         menu_window()
#         result = game_window()
#         end_window(result)


def menu_window():
    # 定义全局变量
    global lastEMG, i

    floor_gap = IMAGES['floor'].get_width() - W
    floor_x = 0

    guide_x = (W - IMAGES['guide'].get_width())/2
    guide_y = (FLOOR_Y - IMAGES['guide'].get_height())/2
    bird_x = W * 0.2
    bird_y = (H - IMAGES['birds'][0].get_height())/2
    bird_y_vel = 1
    bird_y_range = [bird_y - 8, bird_y + 8 ]

    # #阻抗控制 风
    # wind_x = W / 2
    # wind_y = H / 2

    idx = 0   # 参数
    repeat = 5    #扇翅膀帧数
    frames = [0] * repeat + [1] * repeat + [2] * repeat + [1] * repeat

    while True:

        # if 0 < i % 400 < 100:
        #     SCREEN.blit(IMAGES['windup'], (W / 2 - 50, 0))
        # if 100 < i % 400 < 200:
        #     SCREEN.blit(IMAGES['winddown'], (W / 2 - 50, 490))
        # if 200 < i % 400 < 300:
        #     SCREEN.blit(IMAGES['windleft'], (0, H / 2 - 100))
        # if 300 < i % 400 < 400:
        #     SCREEN.blit(IMAGES['windright'], (900, H / 2 - 100))
        # i = i + 1
        # pygame.display.update()
        # 测试 读取myo数据
        # if len(win.emg_data_queue) != 0:
        #     if win.begin == 1:
        #         temp = Data()
        #         temp.EMG = win.emg_data_queue.pop()
        #         if temp.EMG != lastEMG:
        #             DATALIST.append(temp)
        #             filter()
        #             EstimatePosition()
        #             EstimateStiffness()
        #             ImpedanceControl()
        #             lastEMG = temp.EMG
        #             # print(temp.EMG)
        #             print('DOF1:', DOF1 * 1000, 'DOF2:', DOF2 * 1000)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                quit()
            if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                return

        floor_x -= 4
        if floor_x <= -floor_gap:
            floor_x = 0

        bird_y += bird_y_vel
        if bird_y < bird_y_range[0] or bird_y > bird_y_range[1]:
            bird_y_vel *= -1

        idx += 1
        idx %= len(frames)

        SCREEN.blit(IMAGES['bgpic'], (0, 0))
        SCREEN.blit(IMAGES['floor'], (floor_x, FLOOR_Y))
        SCREEN.blit(IMAGES['guide'], (guide_x, guide_y))
        SCREEN.blit(IMAGES['birds'][frames[idx]], (bird_x, bird_y))
        # SCREEN.blit(IMAGES['winddown'], (W / 2 - 50, 490))

        pygame.display.update()
        CLOCK.tick(FPS)

def game_window():
    # 定义全局变量
    global lastEMG, i
    score = 0
    AUDIO['flap'].play()

    floor_gap = IMAGES['floor'].get_width() - W
    floor_x = 0

    bird = Bird(W * 0.2, H * 0.4)

    distance = 250
    n_pairs = 6
    pipe_gap = 200
    pipe_group = pygame.sprite.Group()
    for i in range(n_pairs):
        pipe_y = random.randint(int(H * 0.3), int(H * 0.7))
        pipe_up = Pipe(W + i * distance, pipe_y, True)
        pipe_down = Pipe(W + i * distance, pipe_y - pipe_gap, False)
        pipe_group.add(pipe_up)
        pipe_group.add(pipe_down)

    while True:

        # #读取myo数据
        if len(win.emg_data_queue) != 0:
            if win.begin == 1:
                temp = Data()
                temp.EMG = win.emg_data_queue.pop()
                if temp.EMG != lastEMG:
                    DATALIST.append(temp)
                    filter()
                    EstimatePosition()
                    EstimateStiffness()

                    ImpedanceControl()
                    lastEMG = temp.EMG
                    # print(temp.EMG)
                    # print('DOF1:', DOF1 * 1000, 'DOF2:', DOF2 * 1000)

        flapUP = False
        flapDOWN = False
        flapLEFT = False
        flapRIGHT = False
        speedUP = False
        speedDown = False

        #风向随机
        UP = False
        DOWN = False
        LEFT = False
        RIGHT = False

        # 位置映射
        if DOF1 * 1000 < -0.45:
            flapUP = True
            # AUDIO['flap'].play()
        if DOF1 * 1000 > -0.4:
            flapDOWN = True
            # AUDIO['flap'].play()
        if DOF2 * 1000 > 0.6:
            flapLEFT = True
            # AUDIO['flap'].play()
        if DOF2 * 1000 < 0.0:
            flapRIGHT = True
            # AUDIO['flap'].play()

        # #刚度映射
        # if STF1 > 0.2 and STF2 > 0.6:
        #     speedUP = True
        # if STF1 < 0.15 and STF2 < 0.5:
        #     speedDown = True

        #阻抗控制

        if i % 800 < 100:
            UP = True
        if 200 < i % 800 < 300:
            DOWN = True
        if 400 < i % 800 < 500:
            LEFT = True
        if 600 < i % 800 < 700:
            RIGHT = True
        i = i + 1
        bird.wind(UP, DOWN, LEFT, RIGHT)


        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                quit()

        floor_x -= 4
        if floor_x <= -floor_gap:
            floor_x = 0

        bird.update(flapUP, flapDOWN, flapLEFT, flapRIGHT, speedUP, speedDown)

        first_pipe_up = pipe_group.sprites()[0]
        first_pipe_down = pipe_group.sprites()[1]
        if first_pipe_up.rect.right < 0:
            pipe_y = random.randint(int(H * 0.3), int(H * 0.7))
            new_pipe_up = Pipe(first_pipe_up.rect.x + n_pairs * distance, pipe_y, True)
            new_pipe_down = Pipe(first_pipe_down.rect.x + n_pairs * distance, pipe_y - pipe_gap, False)
            pipe_group.add(new_pipe_up)
            pipe_group.add(new_pipe_down)
            first_pipe_up.kill()
            first_pipe_down.kill()


        pipe_group.update()


        if bird.rect.y > FLOOR_Y or bird.rect.y < 0 or pygame.sprite.spritecollideany(bird, pipe_group):
            bird.dying = True
            AUDIO['hit'].play()
            AUDIO['die'].play()
            result = {'bird': bird, 'pipe_group': pipe_group, 'score': score}
            return result

        if bird.rect.left - 4 <= first_pipe_up.rect.centerx < bird.rect.left:
            AUDIO['score'].play()
            score += 1

        '''for pipe in pipe_group.sprites():      # 复杂但是第一想法
            right_to_left = max(bird.rect.right, pipe.rect.right) - min(bird.rect.left, pipe.rect.left)
            bottom_to_top = max(bird.rect.bottom, pipe.rect.bottom) - min(bird.rect.top, pipe.rect.top)
            if right_to_left < bird.rect.width + pipe.rect.width and bottom_to_top < bird.rect.height + pipe.rect.height:
                AUDIO['hit'].play()
                AUDIO['die'].play()
                result = {'bird': bird}
                return result'''

        SCREEN.blit(IMAGES['bgpic'], (0, 0))
        pipe_group.draw(SCREEN)
        SCREEN.blit(IMAGES['floor'], (floor_x, FLOOR_Y))
        show_score(score)
        SCREEN.blit(bird.image, bird.rect)
        pygame.display.update()
        CLOCK.tick(FPS)

def end_window(result):

    gameover_x = (W - IMAGES['gameover'].get_width())/2
    gameover_y = (FLOOR_Y - IMAGES['gameover'].get_height())/2

    bird = result['bird']
    pipe_group = result['pipe_group']

    while True:
        if bird.dying:
            bird.go_die()
        else:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    quit()
                if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                    return

        SCREEN.blit(IMAGES['bgpic'], (0, 0))
        pipe_group.draw(SCREEN)
        SCREEN.blit(IMAGES['floor'], (0, FLOOR_Y))
        SCREEN.blit(IMAGES['gameover'], (gameover_x, gameover_y))
        SCREEN.blit(bird.image, bird.rect)
        show_score(result['score'])
        pygame.display.update()
        CLOCK.tick(FPS)

def show_score(score):
    score_str = str(score)
    n = len(score_str)
    w = IMAGES['0'].get_width() * 1.1
    x = (W - n * w) / 2
    y = H * 0.1
    for number in score_str:
        SCREEN.blit(IMAGES[number], (x, y))
        x += w

class Bird:
    def __init__(self, x, y):
        self.frames = [0] * 5 + [1] * 5 + [2] * 5
        self.idx = 0
        self.images = IMAGES['birds']
        self.image = self.images[self.frames[self.idx]]
        self.rect = self.image.get_rect()
        self.rect.x = x
        self.rect.y = y
        self.x_vel = 0
        # self.y_vel = -10
        self.y_vel = 0
        self.max_y_vel = 10
        self.gravity = 1
        self.rotate = 45
        self.max_rotate = -20
        self.rotate_vel = -3
        self.x_vel_after_flap = -2
        self.y_vel_after_flap = -2
        self.rotate_after_flap = 15
        self.rotate_after_flapX = 30
        self.dying = False
        self.windSpeed = 0.25

    def update(self, flapUP = False, flapDOWN = False, flapLEFT = False, flapRIGHT = False, speedUP = False, speedDOWN = False):

        if flapUP:
            # 初始
            # self.y_vel = self.y_vel_after_flap
            # self.rotate = self.rotate_after_flap
            self.rect.y += 1.5 * self.y_vel_after_flap
            if speedUP:
                self.rect.y += self.y_vel_after_flap
            if speedDOWN:
                self.rect.y -= self.y_vel_after_flap * 0.75

        if flapDOWN:
            # self.y_vel = - self.y_vel_after_flap
            # self.rotate = - self.rotate_after_flap
            self.rect.y -= self.y_vel_after_flap
            if speedUP:
                self.rect.y -= self.y_vel_after_flap
            if speedDOWN:
                self.rect.y += self.y_vel_after_flap * 0.75

        if flapLEFT:
            self.x_vel = self.x_vel_after_flap
            self.rotate = self.rotate_after_flapX
            self.rect.x += self.x_vel
            if speedUP:
                self.rect.x += self.x_vel
            if speedDOWN:
                self.rect.x -= self.x_vel * 0.75

        if flapRIGHT:
            self.x_vel = - self.x_vel_after_flap
            self.rotate = - self.rotate_after_flapX
            self.rect.x += self.x_vel
            if speedUP:
                self.rect.x += self.x_vel
            if speedDOWN:
                self.rect.x -= self.x_vel * 0.75

        # self.y_vel = min(self.y_vel + self.gravity, self.max_y_vel)
        # self.rect.y += self.y_vel
        # self.rotate = max(self.rotate + self.rotate_vel, self.max_rotate)
        self.rotate = max(self.rotate_vel, self.max_rotate)

        self.idx += 1
        self.idx %= len(self.frames)
        self.image = self.images[self.frames[self.idx]]
        self.image = pygame.transform.rotate(self.image, self.rotate)

    def wind(self, UP = False, DOWN = False, LEFT = False, RIGHT = False):
        if UP:
            SCREEN.blit(IMAGES['windup'], (W / 2 - 50, 0))
            pygame.display.update()
            AUDIO['wind'].play()
            # self.rect.y += self.windSpeed
        if DOWN:
            SCREEN.blit(IMAGES['winddown'], (W / 2 - 50, 490))
            pygame.display.update()
            AUDIO['wind'].play()
            # self.rect.y -= self.windSpeed
        if LEFT:
            SCREEN.blit(IMAGES['windleft'], (0, H / 2 - 100))
            pygame.display.update()
            AUDIO['wind'].play()
            # self.rect.x += self.windSpeed
        if RIGHT:
            SCREEN.blit(IMAGES['windright'], (900, H / 2 - 100))
            pygame.display.update()
            AUDIO['wind'].play()
            # self.rect.x -= self.windSpeed

    def go_die(self):
        if self.rect.y < FLOOR_Y:
            self.rect.y += self.max_y_vel
            self.rotate = -90
            self.image = self.images[self.frames[self.idx]]
            self.image = pygame.transform.rotate(self.image, self.rotate)
        else:
            self.dying = False

class Pipe(pygame.sprite.Sprite):
    def __init__(self, x, y, upwards = True):
        pygame.sprite.Sprite.__init__(self)
        if upwards:
            self.image = IMAGES['pipes'][0]
            self.rect = self.image.get_rect()
            self.rect.x = x
            self.rect.top = y
        else:
            self.image = IMAGES['pipes'][1]
            self.rect = self.image.get_rect()
            self.rect.x = x
            self.rect.bottom = y

        self.x_vel = -4

    def update(self):
        self.rect.x += self.x_vel

#####################################

class coefficient:
    def __init__(self):
        birth_data1 = []

        # 60Hz 1Hz
        self.filterA = [1, -1.85214648539594, 0.862348626030081]
        self.filterB = [0.00255053515853629, 0.00510107031707259, 0.00255053515853629]
        csv_reader1 = csv.reader(open("C:\\pyMyo-train_V2.4(beta)\\Data\\S1_mapping_data.csv"))
        for row in csv_reader1:  # 将csv 文件中的数据保存到birth_data中
            birth_data1.append(row)
        birth_data = [float(x) for x in birth_data1[1]]  # 将数据从string形式转换为float形式
        self.stfmax = birth_data[0]
        self.stfmin = birth_data[1]
        self.t11 = birth_data[2]
        self.t12 = birth_data[3]
        self.t13 = birth_data[4]
        self.t14 = birth_data[5]
        self.t21 = birth_data[6]
        self.t22 = birth_data[7]
        self.t23 = birth_data[8]
        self.t24 = birth_data[9]
        self.MVC = birth_data[10]

        birth_data2 = []
        csv_reader2 = csv.reader(open("C:\\pyMyo-train_V2.4(beta)\\Data\\S1_WT_data.csv"))
        for row in csv_reader2:  # 将csv 文件中的数据保存到birth_data中
            birth_data2.append(row)
        birth_data2 = [[float(x) for x in row] for row in birth_data2]  # 将数据从string形式转换为float形式
        self.W1 = birth_data2[0]
        self.W2 = birth_data2[1]
        self.W3 = birth_data2[2]
        self.W4 = birth_data2[3]

        birth_data2 = []
        csv_reader2 = csv.reader(open("C:\\pyMyo-train_V2.4(beta)\\Data\\S1_WN_data.csv"))
        for row in csv_reader2:  # 将csv 文件中的数据保存到birth_data中
            birth_data2.append(row)
        birth_data2 = [[float(x) for x in row] for row in birth_data2]  # 将数据从string形式转换为float形式
        self.WN1 = birth_data2[0]
        self.WN2 = birth_data2[1]
        self.WN3 = birth_data2[2]
        self.WN4 = birth_data2[3]

coe = coefficient()

class Data:
    def __init__(self):
        self.time = 0
        self.EMG = []
        self.filteredEMG = []
        self.estimatedposition = []
        self.estimatedstiffness = []
        self.targetposition = []
        self.targetstiffness = []

DATALIST = []

lastEMG = [0, 0, 0, 0, 0, 0, 0, 0]

# 控制函数
# def ControlStage(lastEMG=None):
#     temp = Data()
#     if temp.emg_data_queue != lastEMG:
#         DATALIST.append(temp)
#         filter()
#         EstimatePosition()
#         EstimateStiffness()
#         lastEMG = temp.emg_data_queue

DOF1 = 0
DOF2 = 0
def EstimatePosition():
    global DOF1, DOF2
    position_feature = []
    estimated_position = []
    normalizedfilteredEMG = []
    # print(coe.MVC)
    for i in range(8):
        normalizedfilteredEMG.append(DATALIST[-1].filteredEMG[i] / coe.MVC)
    # print(DATALIST[-1].filteredEMG)
    position_feature.append(VectorProduct(coe.WN1, normalizedfilteredEMG))
    position_feature.append(VectorProduct(coe.WN2, normalizedfilteredEMG))
    position_feature.append(VectorProduct(coe.WN3, normalizedfilteredEMG))
    position_feature.append(VectorProduct(coe.WN4, normalizedfilteredEMG))
    # print(position_feature)

    #位置估计
    estimated_position.append((coe.t11 * position_feature[0]) + (coe.t12 * position_feature[1]) \
    + (coe.t13 * position_feature[2]) + (coe.t14 * position_feature[3]))
    estimated_position.append((coe.t21 * position_feature[0]) + (coe.t22 * position_feature[1]) \
    + (coe.t23 * position_feature[2]) + (coe.t24 * position_feature[3]))
    # print(estimated_position)

    DOF1 = estimated_position[0]
    DOF2 = estimated_position[1]
    # print('DOF1:', DOF1, 'DOF2：', DOF2)

    DATALIST[-1].estimatedposition = estimated_position.copy()

STF1 = 0
STF2 = 0
def EstimateStiffness():
    global STF1, STF2
    stiffness_feature = []
    estimated_stiffness = []
    normalizedfilteredEMG = []
    for i in range(8):
        normalizedfilteredEMG.append(DATALIST[-1].filteredEMG[i] / coe.MVC)
    stiffness_feature.append(VectorProduct(coe.W1, normalizedfilteredEMG))
    stiffness_feature.append(VectorProduct(coe.W2, normalizedfilteredEMG))
    stiffness_feature.append(VectorProduct(coe.W3, normalizedfilteredEMG))
    stiffness_feature.append(VectorProduct(coe.W4, normalizedfilteredEMG))

    #刚度估计
    estimated_stiffness.append((min(stiffness_feature[0], stiffness_feature[1]) - coe.stfmin) \
    / (coe.stfmax - coe.stfmin))
    estimated_stiffness.append((min(stiffness_feature[2], stiffness_feature[3]) - coe.stfmin) \
    / (coe.stfmax - coe.stfmin))

    STF1 = estimated_stiffness[0]
    STF2 = estimated_stiffness[1]

    DATALIST[-1].estimatedposition = estimated_stiffness.copy()


dxk = xk_pre = dxk_pre = ddxk_pre = 0
dyk = yk_pre = dyk_pre = ddyk_pre = 0
BX = 0

def ImpedanceControl():
    global dxk, xk_pre, dxk_pre, ddxk_pre, dyk, yk_pre, dyk_pre, ddyk_pre, STF1, STF2, DOF1, DOF2, BX
    T = 0.005
    M = 0.1
    if STF1 < 0:
        STF1 = 0;
    if STF2 < 0:
        STF2 = 0;
    BX = 2 * np.sqrt(STF1)
    BY = 2 * np.sqrt(STF2)
    ddxk = 1 / M * (STF1 * xk_pre - STF1 * DOF1 - BX * dxk)
    ddyk = 1 / M * (STF2 * yk_pre - STF2 * DOF2 - BY * dyk)
    dxk = T / 2 * (ddxk + ddxk_pre) + dxk_pre
    dyk = T / 2 * (ddyk + ddyk_pre) + dyk_pre
    DOF1 = T / 2 * (dxk + dxk_pre) - xk_pre
    DOF2 = T / 2 * (dyk + dyk_pre) - yk_pre

    xk_pre = DOF1
    yk_pre = DOF2
    dxk_pre = dxk
    dyk_pre = dyk
    ddxk_pre = ddxk
    ddyk_pre = ddyk


def VectorProduct(W, EMG):
    ret = 0
    for i in range(len(W)):
        ret += W[i] * EMG[i]
    return ret



def filter():

    n = len(DATALIST)
    temp = [0, 0, 0, 0, 0, 0, 0, 0]
    for i in range(8):
        if n == 1:
            temp[i] = (coe.filterB[0] * abs(DATALIST[0].EMG[i])) / coe.filterA[0]
        elif n == 2:
            temp[i] = (coe.filterB[0] * abs(DATALIST[1].EMG[i]) + coe.filterB[1] * abs(DATALIST[0].EMG[i]) - \
                       coe.filterA[1] * abs(DATALIST[0].filteredEMG[i]))/coe.filterA[0]
        else:
            temp[i] = (coe.filterB[0] * abs(DATALIST[n - 1].EMG[i]) + coe.filterB[1] * abs(DATALIST[n - 2].EMG[i]) + \
                       coe.filterB[2] * abs(DATALIST[n - 3].EMG[i])) \
                      - coe.filterA[1] * abs(DATALIST[n - 2].filteredEMG[i]) - \
                      coe.filterA[2] * abs(DATALIST[n - 3].filteredEMG[i]) / coe.filterA[0]
        # print(temp[i])  验证有数据
    DATALIST[-1].filteredEMG = temp.copy()
    # print(DATALIST[n - 1].filteredEMG)

###################################################


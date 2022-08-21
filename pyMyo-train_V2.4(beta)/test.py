# -*-codeing = utf-8 -*-
# @Time : 2022/2/24 0024
# @Author : Yu Yimin
# @File : test.py
# @Software : PyCharm

from pyMyoMain import Win
import myo, sys
from PyQt5.QtWidgets import QApplication
import numpy as np
import csv
import time
from myoManager import MyoManager, EventType
from collections import deque

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
        self.emg_data_queue = deque(maxlen=1000)

    def callback(self, dictMsg):  # 重点函数 加到游戏中
        typeEvt = dictMsg["type"]
        dataEvt = dictMsg["data"]

        if typeEvt == EventType.battery_level:
            self.ui.setBATTERY.setText(repr(dataEvt["battery"]))

        if typeEvt == EventType.connected:
            self.ui.msgBrowser.append("Connected to "
                                      + repr(dataEvt["name"])
                                      + "with mac address: "
                                      + repr(dataEvt["mac_address"])
                                      + '. \n')

            self.ui.connectBtn.setEnabled(False)
            self.ui.disConnectBtn.setEnabled(True)
            self.ui.startBtn.setEnabled(True)

        elif typeEvt == EventType.disconnected:
            if dataEvt["timeout"]:
                self.ui.msgBrowser.append("Connection timed out!" + '\n')
                self.disconnection()

            if dataEvt["unOpenMyo"]:
                self.ui.msgBrowser.append("Unable to connect to Myo Connect. Is Myo Connect running?" + '\n')
                self.disconnection()

        elif typeEvt == EventType.emg:
            self.emg_data_queue.append(dataEvt["emg"])
            emg_data_list = dataEvt["emg"] + [time.time()]
            self.EMG.append(emg_data_list)


DATALIST = []

lastEMG = [0, 0, 0, 0, 0, 0, 0, 0]
def ControlStage(lastEMG):
    app = QApplication(sys.argv)
    temp = Win()
    temp.show()
    if temp.emg_data_queue != lastEMG:
        DATALIST.append(temp)
        filter()
        EstimatePosition()
        EstimateStiffness()
        lastEMG = temp.emg_data_queue

DOF1 = 0
DOF2 = 0
def EstimatePosition():
    position_feature = []
    estimated_position = []
    normalizedfilteredEMG = []
    for i in range(8):
        normalizedfilteredEMG.append(DATALIST[-1].filteredEMG[i] / coe.MVC)
    position_feature.append(VectorProduct(coe.WN1, normalizedfilteredEMG))
    position_feature.append(VectorProduct(coe.WN2, normalizedfilteredEMG))
    position_feature.append(VectorProduct(coe.WN3, normalizedfilteredEMG))
    position_feature.append(VectorProduct(coe.WN4, normalizedfilteredEMG))

    #位置估计
    estimated_position.append((coe.t11 * position_feature[0]) + (coe.t12 * position_feature[1]) \
    + (coe.t13 * position_feature[2]) + (coe.t14 * position_feature[3]))
    estimated_position.append((coe.t21 * position_feature[0]) + (coe.t22 * position_feature[1]) \
    + (coe.t23 * position_feature[2]) + (coe.t24 * position_feature[3]))

    DOF1 = estimated_position[0] * 5
    DOF2 = estimated_position[1] * 5

    DATALIST[-1].estimatedposition = estimated_position.copy()

def EstimateStiffness():
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

    DATALIST[-1].estimatedposition = estimated_stiffness.copy()


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
    DATALIST[-1].filteredEMG = temp.copy()




def main():

    while True:
        app = QApplication(sys.argv)
        temp = Win()
        temp.show()
        print(temp.emg_data_queue)
        if(len(temp.emg_data_queue) == 8):
            ControlStage()

if __name__ == '__main__':
    main()

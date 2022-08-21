# -*-codeing = utf-8 -*-
# @Time : 2022/3/4 0004
# @Author : Yu Yimin
# @File : temp.py
# @Software : PyCharm
import random

from myoManager import MyoManager, EventType
from pyMyoMain import *
from myoManager import MyoManager, EventType
import myo, sys, os
from PyQt5.QtWidgets import QApplication

class Test:
    def __init__(self):
        self.value = None
    def test1(self, value):
        x = 1 + value
        return x
    def test2(self):
        y = self.test1(self.value)
        return y


def main():
    test = Test()
    y = test.test2()
    print(y)


    # app = QApplication(sys.argv)
    # while True:
    #     win = Win()
    #     win.show()
    #
    #     while len(win.emg_data_queue) != 0:
    #         if win.emg_data_queue.pop() != lastEMG:
    #             DATALIST.append(win)
    #             print(DATALIST.pop())
    #
    # sys.exit(app.exec_())
    # print('a:', a,'b:', b)



    # data = []
    # data.append(1)
    # data.append(2)
    # print(data[1])
    # Test = [test() for i in range(9)]
    # Test[0].value = 1
    # print(Test[0].value)
if __name__ == '__main__':
    main()
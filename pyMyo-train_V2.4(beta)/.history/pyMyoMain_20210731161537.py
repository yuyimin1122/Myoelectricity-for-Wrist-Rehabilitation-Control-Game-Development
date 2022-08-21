from collections import deque
import numpy as np
from PyQt5.QtWidgets import QWidget, QMessageBox, QFileDialog
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QIcon, QImage, QPixmap
import pyqtgraph as pg
from myoManager import MyoManager, EventType
from Ui_pyMyo_alpha import Ui_Form
from scipy import signal
from sklearn.decomposition import NMF
from sklearn.linear_model import LinearRegression  #线性回归
import pandas as pd
import csv
import time
import math

pg.setConfigOptions(leftButtonPan=False)
dirname = r'E:\研究生\项目1\MYO\Pymyo\pyMyo-master'
subjectnum = 0
statuscount = 0

model = NMF(n_components=1,  # k value,默认会保留全部特征
          init= None,  # W H 的初始化方法，包括'random' | 'nndsvd'(默认) |  'nndsvda' | 'nndsvdar' | 'custom'.
          solver='mu',  # 'cd' | 'mu'
          beta_loss='frobenius',  # {'frobenius', 'kullback-leibler', 'itakura-saito'}，一般默认就好
          tol=1e-4,  # 停止迭代的极限条件
          max_iter=200,  # 最大迭代次数
          random_state=None,
          alpha=0.,  # 正则化参数
          l1_ratio=0.,  # 正则化参数
          verbose=0,  # 冗长模式
          shuffle=False  # 针对"cd solver"
          )

lrModel = LinearRegression()

def lowpass(data,Fc):
    b, a = signal.butter(2, 2.0*Fc/200, 'lowpass')   
    filtedData = signal.filtfilt(b, a, data)
    return filtedData



class Win(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.ui = Ui_Form()
        self.ui.setupUi(self)
        path = r'logo.png'
        self.setWindowIcon(QIcon(path))

        self.timer = QTimer(self)
        self.emg_curve = []
        self.emg_max = []
        self.emg_min = []
        self.emg_pose1 = []
        self.emg_pose2 = []
        self.emg_pose3 = []
        self.emg_pose4 = []
        self.emg_data = []
        # self.MVC = np.zeros((8,1))
        self.emg_data_queue = deque(maxlen=1000)    

        self.myo = None
        self.initUI()


    def initUI(self):
        self.emg_plot = pg.PlotWidget(name='emg_plot')
        self.emg_plot.setTitle("EMG")
        self.emg_plot.setXRange(0, 1000)
        self.emg_plot.setYRange(0, 3000)
        self.ui.graph_layout.addWidget(self.emg_plot)
        for i in range(8):
            c = self.emg_plot.plot(pen=(i, 10))
            c.setPos(0, i * 400)
            self.emg_curve.append(c)


        self.ui.connectBtn.clicked.connect(self.connection)
        self.ui.startBtn.clicked.connect(self.start)
        self.ui.disConnectBtn.clicked.connect(self.disconnection)
        self.ui.calculateBtn.clicked.connect(self.calculate)
        self.ui.pose1Btn.clicked.connect(self.pose1)
        self.ui.pose2Btn.clicked.connect(self.pose2)
        self.ui.pose3Btn.clicked.connect(self.pose3)
        self.ui.pose4Btn.clicked.connect(self.pose4)
        self.ui.maxBtn.clicked.connect(self.k_max)
        self.ui.minBtn.clicked.connect(self.k_min)
        self.ui.freeBtn.clicked.connect(self.free)


        self.ui.connectBtn.setEnabled(True)
        self.ui.disConnectBtn.setEnabled(False)
        self.ui.startBtn.setEnabled(False)
        self.ui.calculateBtn.setEnabled(False)
        self.ui.pose1Btn.setEnabled(False)
        self.ui.pose2Btn.setEnabled(False)
        self.ui.pose3Btn.setEnabled(False)
        self.ui.pose4Btn.setEnabled(False)
        self.ui.maxBtn.setEnabled(False)
        self.ui.minBtn.setEnabled(False)
        self.ui.freeBtn.setEnabled(False)


    def connection(self):
        self.ui.msgBrowser.append("Trying to connect to Myo (connection will timeout in 5 seconds)." + '\n')
        self.ui.calculateBtn.setEnabled(False)
        self.ui.subjectnum.setEnabled(False)
        self.readConfig()
        if not self.myo:
            self.myo = MyoManager(sender=self)

        if not self.myo.connected:
            self.myo.connect()


    def disconnection(self):
        self.timer.stop()

        self.ui.pose1Btn.setEnabled(False)
        self.ui.pose2Btn.setEnabled(False)
        self.ui.pose3Btn.setEnabled(False)
        self.ui.pose4Btn.setEnabled(False)
        self.ui.maxBtn.setEnabled(False)
        self.ui.minBtn.setEnabled(False)
        self.ui.freeBtn.setEnabled(False)
        self.ui.connectBtn.setEnabled(True)
        self.ui.disConnectBtn.setEnabled(False)
        self.ui.startBtn.setEnabled(False)
        self.ui.calculateBtn.setEnabled(True)
        self.ui.subjectnum.setEnabled(True)

        self.ui.msgBrowser.append("Disconnected from Myo." + '\n')

        if self.myo:
            if self.myo.connected:
                self.myo.disconnect()


    def start(self):
        if self.ui.startBtn.isEnabled():
            self.ui.startBtn.setEnabled(False)
            self.ui.calculateBtn.setEnabled(False)
            self.ui.pose1Btn.setEnabled(True)
            self.ui.pose2Btn.setEnabled(True)
            self.ui.pose3Btn.setEnabled(True)
            self.ui.pose4Btn.setEnabled(True)
            self.ui.maxBtn.setEnabled(True)
            self.ui.minBtn.setEnabled(True)
            self.ui.freeBtn.setEnabled(True)
                  

        self.timer_start()


    def pose1(self):
        global statuscount
        statuscount = 1
        self.emg_data.clear()
        self.ui.msgBrowser.append("Pose1" + '\n')


    def pose2(self):
        global statuscount
        statuscount = 2
        self.emg_data.clear()
        self.ui.msgBrowser.append("Pose2" + '\n')


    def pose3(self):
        global statuscount
        statuscount = 3
        self.emg_data.clear()
        self.ui.msgBrowser.append("Pose3" + '\n')


    def pose4(self):
        global statuscount
        statuscount = 4
        self.emg_data.clear()
        self.ui.msgBrowser.append("Pose4" + '\n')


    def k_max(self):
        global statuscount
        statuscount = 5
        self.emg_data.clear()
        self.ui.msgBrowser.append("Maximum Stiffness" + '\n')


    def k_min(self):
        global statuscount
        statuscount = 6
        self.emg_data.clear()
        self.ui.msgBrowser.append("Minimum Stiffness" + '\n')


    def free(self):
        global statuscount
        self.saveData()
        self.emg_data.clear()
        statuscount = 0
        self.ui.msgBrowser.append("Free movement" + '\n')


    def calculate(self):
        self.ui.msgBrowser.append("Calculating" + '\n')
        W = self.get_W()
        self.get_parameter(W)
        ed = pd.DataFrame(W)
        path = dirname + r'\Data\S' + str(subjectnum) + r'_W_data.csv'
        ed.to_csv(path, mode='w', header=False,index=False)
        ed = pd.DataFrame(np.transpose(W))
        path = dirname + r'\Data\S' + str(subjectnum) + r'_WT_data.csv'
        ed.to_csv(path, mode='w', header=False,index=False)
        self.ui.msgBrowser.append("Calculation complete" + '\n')


    def mintozero(self,data):
        """
        将最小值小于0的数据的最小值变为0

        Parameter
        ---------
        data : filted emg data

        Returns
        -------
        minzerodata : data without negative number
        """
        
        minzerodata = np.zeros((8,len(data[0,:])))
        for i in range(8):
            data_min = np.min(data[i,:])
            if data_min < 0 :
                minzerodata[i,:] = data[i,:] - data_min
            else:
                minzerodata[i,:] = data[i,:]
        return minzerodata


    def equationcalculate(rawdata,samplerate):
        """
        get the filter using difference equation 

        parameter
        --------
        rawdata : original signal  (numpy.ndarray)

        Return
        ------

        filtereddata : signal after filter  (numpy.ndarray)
        """

        filterA = [1, -2, 1]
        filterB = [1, 2, 1 ]
        # 200hz 1hz
        if samplerate == 200 :
            filterA = [1, -1.95557824031504, 0.956543676511203]
            filterB = [0.000241359049041981, 0.000482718098083961, 0.000241359049041981]
        elif samplerate == 60 :
            filterA = [1, -1.85214648539594, 0.862348626030081]
            filterB = [0.00255053515853629, 0.00510107031707259, 0.00255053515853629]


        datachannel = len(rawdata)
        datalenth = len(rawdata[0])
        filtereddata = np.zeros((datachannel,datalenth))

        for i in range(datachannel):
            for j in range(datalenth):
                if j == 0:
                    filtereddata[i,j] = (float(filterB[0]) * np.abs(rawdata[i,j])) / float(filterA[0])
                elif j == 1:
                    filtereddata[i,j] = ((float(filterB[0]) * np.abs(rawdata[i,j]) 
                                        + float(filterB[1]) * np.abs(rawdata[i,j-1])
                                        - float(filterA[1]) * np.abs(filtereddata[i,j-1])) 
                                        / float(filterA[0]))
                else :
                    filtereddata[i,j] = ((float(filterB[0]) * np.abs(rawdata[i,j]) 
                                        + float(filterB[1]) * np.abs(rawdata[i,j-1])
                                        + float(filterB[2]) * np.abs(rawdata[i,j-2])
                                        - float(filterA[1]) * np.abs(filtereddata[i,j-1])
                                        - float(filterA[2]) * np.abs(filtereddata[i,j-2])) 
                                        / float(filterA[0]))

        return filtereddata




    def get_emg_data(self):
        path = dirname + r'\Data\S' + str(subjectnum) + r'_emg_train_pose1_data.csv'
        emg1 = pd.read_csv(path,header=None)
        raw_emg1 = np.transpose(emg1.values)
        path = dirname + r'\Data\S' + str(subjectnum) + r'_emg_train_pose2_data.csv'
        emg2 = pd.read_csv(path,header=None)
        raw_emg2 = np.transpose(emg2.values)
        path = dirname + r'\Data\S' + str(subjectnum) + r'_emg_train_pose3_data.csv'
        emg3 = pd.read_csv(path,header=None)
        raw_emg3 = np.transpose(emg3.values)
        path = dirname + r'\Data\S' + str(subjectnum) + r'_emg_train_pose4_data.csv'
        emg4 = pd.read_csv(path,header=None)
        raw_emg4 = np.transpose(emg4.values)
        path = dirname + r'\Data\S' + str(subjectnum) + r'_emg_max_stiffness_data.csv'
        emg5 = pd.read_csv(path,header=None)
        raw_emg5 = np.transpose(emg5.values)
        path = dirname + r'\Data\S' + str(subjectnum) + r'_emg_min_stiffness_data.csv'
        emg6 = pd.read_csv(path,header=None)
        raw_emg6 = np.transpose(emg6.values)

        b, a = signal.butter(2, 2.0*1/200, 'lowpass')
        self.emg_max = self.mintozero(signal.filtfilt(b, a,np.abs(raw_emg5[0:8,:])))
        # for i in range(8):
        #     self.MVC[i] = np.mean(self.emg_max[i,:])
        self.MVC = np.max(self.emg_max)
        print(self.MVC)
        self.emg_max = self.emg_max / self.MVC
        self.emg_min = self.mintozero(signal.filtfilt(b, a,np.abs(raw_emg6))) / self.MVC
        self.emg_pose1 = self.mintozero(signal.filtfilt(b, a,np.abs(raw_emg1))) / self.MVC
        self.emg_pose2 = self.mintozero(signal.filtfilt(b, a,np.abs(raw_emg2))) / self.MVC
        self.emg_pose3 = self.mintozero(signal.filtfilt(b, a,np.abs(raw_emg3))) / self.MVC
        self.emg_pose4 = self.mintozero(signal.filtfilt(b, a,np.abs(raw_emg4))) / self.MVC

    def normalize_emg(self,femg,MVC):
        """
        Normalized with MVC of each channel of filted emg data
        
        Parameter
        ---------
        femg : filted emg data
        MVC : average of each channel in max stiffness
        
        Returns
        -------
        normalized_data : normalized data
        """
        for i in range(8):
            normalized_data = femg[i] / MVC[i]

        return normalized_data



    def get_W(self):
        self.get_emg_data()
        w1 = model.fit_transform(X=self.emg_pose1)
        h1 = model.components_
        w2 = model.fit_transform(X=self.emg_pose2)
        h2 = model.components_
        w3 = model.fit_transform(X=self.emg_pose3)
        h3 = model.components_
        w4 = model.fit_transform(X=self.emg_pose4)
        h4 = model.components_

        W1 = w1 * np.max(h1)
        W2 = w2 * np.max(h2)
        W3 = w3 * np.max(h3)
        W4 = w4 * np.max(h4)

        W = np.c_[W1,W2,W3,W4]
        return W


    def get_pose_seed(list):
        length = len(list[0])
        total = np.zeros(length)
        for i in range(length):
            for j in range(8):
                total[i] += list[j][i]

        total = lowpass(total,1)        
        totalmin = np.min(total)
        totalmax = np.max(total)
        pose_seed = np.zeros(length)
        for i in range(length):
            pose_seed[i] = (total[i] - totalmin)/(totalmax - totalmin)
        return pose_seed


    def get_parameter(self,W):
        stiffness1 = np.dot(np.transpose(W),self.emg_max)
        stiffness2 = np.dot(np.transpose(W),self.emg_min)
        stiffnessmax1 = np.minimum(stiffness1[0],stiffness1[1])
        stiffnessmax2 = np.minimum(stiffness1[2],stiffness1[3])
        maxstiffness = np.r_[stiffnessmax1,stiffnessmax2]
        k_max = np.max(maxstiffness)
        k_min = np.min(stiffness2)
        WN = np.linalg.pinv(W)

        position1 = np.dot(WN,self.emg_pose1)
        position2 = np.dot(WN,self.emg_pose2)
        position3 = np.dot(WN,self.emg_pose3)
        position4 = np.dot(WN,self.emg_pose4)

        pose1_train = self.get_pose_seed(position1)
        pose2_train = self.get_pose_seed(position2)
        pose3_train = self.get_pose_seed(position3)
        pose4_train = self.get_pose_seed(position4)

        X1_train = np.hstack((position1[0],position2[0],position3[0],position4[0]))
        X2_train = np.hstack((position1[1],position2[1],position3[1],position4[1]))
        X3_train = np.hstack((position1[2],position2[2],position3[2],position4[2]))
        X4_train = np.hstack((position1[3],position2[3],position3[3],position4[3]))

        Y1_train = np.hstack((pose1_train,-pose2_train,np.zeros(len(position3[0])),np.zeros(len(position4[0]))))
        Y2_train = np.hstack((np.zeros(len(position1[0])),np.zeros(len(position2[0])),pose3_train,-pose4_train))

        DOF1_X = np.c_[X1_train,X2_train]
        DOF1_Y = Y1_train.reshape(-1,1)
        lrModel.fit(DOF1_X,DOF1_Y)
        t11 = lrModel.coef_[0][0]
        t12 = lrModel.coef_[0][1]
        #DOF1 线性回归
        DOF2_X = np.c_[X3_train,X4_train]
        DOF2_Y = Y2_train.reshape(-1,1)
        lrModel.fit(DOF2_X,DOF2_Y)
        t21 = lrModel.coef_[0][0]
        t22 = lrModel.coef_[0][1]

        list2 = [{"k_max":k_max,
                  "k_min":k_min,
                  "t11":t11,
                  "t12":t12,
                  "t21":t21,
                  "t22":t22 ,
                  "MVC":self.MVC,
                }]
        ed = pd.DataFrame(list2,columns=["k_max","k_min","t11","t12","t21","t22","MVC"])        
        path = dirname + r'\Data\S' + str(subjectnum) + r'_mapping_data.csv'
        ed.to_csv(path, mode='w', header=True,index=False)
        #WN矩阵4*8
        ed = pd.DataFrame(WN)
        path = dirname + r'\Data\S' + str(subjectnum) + r'_WN_data.csv'
        ed.to_csv(path, mode='w', header=False,index=False)



    def pause_sample(self):
        self.timer.stop()
        self.ui.startBtn.setEnabled(True)


    def callback(self, dictMsg):
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
            self.emg_data.append(emg_data_list)


    def readConfig(self):
        global subjectnum
        subjectnum = int(self.ui.subjectnum.text())


    def timer_start(self):
        self.timer.timeout.connect(self.update_plots_emg)

        self.timer.start()


    def saveData(self):
        if statuscount != 0:
            ed = pd.DataFrame(self.emg_data)
            if statuscount == 1:
                path = dirname + r'\Data\S' + str(subjectnum) + r'_emg_train_pose1_data.csv'
                ed.to_csv(path, mode='w', header=False,index=False)
            elif statuscount == 2:
                path = dirname + r'\Data\S' + str(subjectnum) + r'_emg_train_pose2_data.csv'
                ed.to_csv(path, mode='w', header=False,index=False)
            elif statuscount == 3:
                path = dirname + r'\Data\S' + str(subjectnum) + r'_emg_train_pose3_data.csv'
                ed.to_csv(path, mode='w', header=False,index=False)                            
            elif statuscount == 4:
                path = dirname + r'\Data\S' + str(subjectnum) + r'_emg_train_pose4_data.csv'
                ed.to_csv(path, mode='w', header=False,index=False)
            elif statuscount == 5:
                path = dirname + r'\Data\S' + str(subjectnum) + r'_emg_max_stiffness_data.csv'
                ed.to_csv(path, mode='w', header=False,index=False)
            elif statuscount == 6:
                path = dirname + r'\Data\S' + str(subjectnum) + r'_emg_min_stiffness_data.csv'
                ed.to_csv(path, mode='w', header=False,index=False) 


    def update_plots_emg(self):
        buffer0 = []
        buffer1 = []
        buffer2 = []
        buffer3 = []
        buffer4 = []
        buffer5 = []
        buffer6 = []
        buffer7 = []
        emgSolve = self.emg_data_queue
        for j in emgSolve:
            emg = j
            buffer0.append(emg[0])
            buffer1.append(emg[1])
            buffer2.append(emg[2])
            buffer3.append(emg[3])
            buffer4.append(emg[4])
            buffer5.append(emg[5])
            buffer6.append(emg[6])
            buffer7.append(emg[7])
        all_buffer = [buffer7, buffer6, buffer5, buffer4, buffer3, buffer2, buffer1, buffer0]
        for i in range(8):
            self.emg_curve[i].setData(all_buffer[i])
        
        return emgSolve


    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Alt:
            self.pause_sample()


    def closeEvent(self, event):
        result = QMessageBox.question(
            self, 'Quit', 'Are you sure?',
            QMessageBox.Yes | QMessageBox.No)
        if result == QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()


    






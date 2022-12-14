# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'myoGame_Form.ui'
#
# Created by: PyQt5 UI code generator 5.13.0
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(845, 505)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("../../../../DYX/.designer/backup/logo.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        Form.setWindowIcon(icon)
        self.GroupEMG = QtWidgets.QGroupBox(Form)
        self.GroupEMG.setGeometry(QtCore.QRect(10, 10, 600, 491))
        self.GroupEMG.setTitle("")
        self.GroupEMG.setObjectName("GroupEMG")
        self.verticalLayoutWidget = QtWidgets.QWidget(self.GroupEMG)
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(20, 20, 571, 461))
        self.verticalLayoutWidget.setObjectName("verticalLayoutWidget")
        self.graph_layout = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)
        self.graph_layout.setSizeConstraint(QtWidgets.QLayout.SetDefaultConstraint)
        self.graph_layout.setContentsMargins(0, 0, 0, 0)
        self.graph_layout.setObjectName("graph_layout")
        self.funcArea = QtWidgets.QGroupBox(Form)
        self.funcArea.setGeometry(QtCore.QRect(630, 40, 201, 81))
        self.funcArea.setTitle("")
        self.funcArea.setObjectName("funcArea")
        self.layoutWidget = QtWidgets.QWidget(self.funcArea)
        self.layoutWidget.setGeometry(QtCore.QRect(10, 0, 181, 72))
        self.layoutWidget.setObjectName("layoutWidget")
        self.gridLayout = QtWidgets.QGridLayout(self.layoutWidget)
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.gridLayout.setSpacing(6)
        self.gridLayout.setObjectName("gridLayout")
        self.Battery = QtWidgets.QLabel(self.layoutWidget)
        self.Battery.setEnabled(True)
        self.Battery.setMaximumSize(QtCore.QSize(16777215, 16777215))
        font = QtGui.QFont()
        font.setPointSize(15)
        self.Battery.setFont(font)
        self.Battery.setObjectName("Battery")
        self.gridLayout.addWidget(self.Battery, 0, 0, 1, 1)
        self.setBATTERY = QtWidgets.QLabel(self.layoutWidget)
        font = QtGui.QFont()
        font.setPointSize(15)
        self.setBATTERY.setFont(font)
        self.setBATTERY.setText("")
        self.setBATTERY.setObjectName("setBATTERY")
        self.gridLayout.addWidget(self.setBATTERY, 0, 1, 1, 1)
        self.btnGroup = QtWidgets.QGroupBox(Form)
        self.btnGroup.setGeometry(QtCore.QRect(630, 130, 201, 111))
        self.btnGroup.setTitle("")
        self.btnGroup.setObjectName("btnGroup")
        self.connectBtn = QtWidgets.QPushButton(self.btnGroup)
        self.connectBtn.setGeometry(QtCore.QRect(10, 10, 95, 40))
        self.connectBtn.setObjectName("connectBtn")
        self.disConnectBtn = QtWidgets.QPushButton(self.btnGroup)
        self.disConnectBtn.setGeometry(QtCore.QRect(10, 60, 95, 40))
        self.disConnectBtn.setObjectName("disConnectBtn")
        self.startBtn = QtWidgets.QPushButton(self.btnGroup)
        self.startBtn.setGeometry(QtCore.QRect(130, 10, 60, 40))
        self.startBtn.setObjectName("startBtn")
        self.saveBtn = QtWidgets.QPushButton(self.btnGroup)
        self.saveBtn.setGeometry(QtCore.QRect(130, 60, 60, 40))
        self.saveBtn.setObjectName("saveBtn")
        self.msgGroup = QtWidgets.QGroupBox(Form)
        self.msgGroup.setGeometry(QtCore.QRect(630, 250, 201, 241))
        self.msgGroup.setTitle("")
        self.msgGroup.setObjectName("msgGroup")
        self.msgBrowser = QtWidgets.QTextBrowser(self.msgGroup)
        self.msgBrowser.setGeometry(QtCore.QRect(0, 0, 201, 241))
        self.msgBrowser.setLocale(QtCore.QLocale(QtCore.QLocale.English, QtCore.QLocale.UnitedStates))
        self.msgBrowser.setObjectName("msgBrowser")

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)
        Form.setTabOrder(self.connectBtn, self.disConnectBtn)
        Form.setTabOrder(self.disConnectBtn, self.startBtn)
        Form.setTabOrder(self.startBtn, self.msgBrowser)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "pyMyo-train"))
        self.Battery.setText(_translate("Form", "BATTERY:"))
        self.connectBtn.setText(_translate("Form", "Connect"))
        self.disConnectBtn.setText(_translate("Form", "Disconnect"))
        self.startBtn.setText(_translate("Form", "Start"))
        self.saveBtn.setText(_translate("Form", "Save"))

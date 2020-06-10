# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 12:15:37 2020

@author: marli
"""
import sys
import numpy as np
import os
import glob
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication, QFileSystemModel, QTreeView, QWidget, QFrame
from PyQt5.QtGui import QIcon, QImage, QPalette, QBrush, QColor


class Ui_Dialog(object):
    def setupUi(self, Dialog):
        
        Dialog.setObjectName("Dialog")
        Dialog.resize(900, 890)
    
  
        self.Vis=False
        self.n_pic_all=False
        self.file_path_ausw=''
        self.file_path_mess=''
        
        
        
        self.buttonBox = QtWidgets.QDialogButtonBox(Dialog)
        self.buttonBox.setGeometry(QtCore.QRect(-10, 750, 341, 32))
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.Cancel|QtWidgets.QDialogButtonBox.Ok)
        self.buttonBox.setObjectName("buttonBox")
        
        



#%%      Messdaten

        self.model = QFileSystemModel()
        self.model.setRootPath('')


        
        self.messdaten_treeView= QTreeView(Dialog)
        self.messdaten_treeView.setModel(self.model)
        self.messdaten_treeView.setGeometry(QtCore.QRect(30,30, 400, 250))
        self.messdaten_treeView.setSortingEnabled(True)  #sorting the element
        self.messdaten_treeView.setAnimated(False)
        self.messdaten_treeView.setIndentation(20)
        self.messdaten_treeView.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.messdaten_treeView.customContextMenuRequested.connect(self.context_menu)
        
        
        self.text_messdaten = QtWidgets.QLabel(Dialog)
        self.text_messdaten.setGeometry(QtCore.QRect(30,10, 400, 22))
        self.text_messdaten.setObjectName("text1")
        self.text_messdaten.setText("Zielordner für Auswertung, Auswahl mit Rechtsklick")
        

        
#%%          AUSWERTUNG
        
        self.model2 = QFileSystemModel()
        self.model2.setRootPath('')
        
        self.auswertung_treeView = QTreeView(Dialog)
        self.auswertung_treeView.setModel(self.model2)
        self.auswertung_treeView.setGeometry(QtCore.QRect(460, 30, 400, 250))
        self.auswertung_treeView.setSortingEnabled(True)
        self.auswertung_treeView.setAnimated(False)
        self.auswertung_treeView.setIndentation(20)
        self.auswertung_treeView.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.auswertung_treeView.customContextMenuRequested.connect(self.context_menu_ausw)
        
        
        self.text2 = QtWidgets.QLabel(Dialog)
        self.text2.setGeometry(QtCore.QRect(460, 10, 400, 22))
        self.text2.setObjectName("text2")
        self.text2.setText("Zielordner für Auswertung, Auswahl mit Rechtsklick")


#%% Kalibration
        
        
        
        self.text_calib = QtWidgets.QLabel(Dialog)
        self.text_calib.setGeometry(QtCore.QRect(30, 350, 700, 22))        
        self.text_calib.setText("Wie viele Kalibrationsordner oder andere nicht auszuwerende Ordner?")

        
        self.spinBox_calib = QtWidgets.QSpinBox(Dialog)
        self.spinBox_calib.setGeometry(QtCore.QRect(490, 350, 42, 22))
        self.spinBox_calib.setMaximum(500)

        
#%%     SCALE
        self.text_scale = QtWidgets.QLabel(Dialog)
        self.text_scale.setGeometry(QtCore.QRect(30, 450, 300, 22))
        self.text_scale.setObjectName("text_scale")
        
        self.text_scale.setText("Skalierungsfaktor 250mum = ?? Pixel")

        
        self.spinBox_scale = QtWidgets.QSpinBox(Dialog)
        self.spinBox_scale.setGeometry(QtCore.QRect(390, 450, 42, 22))
        self.spinBox_scale.setMaximum(500)
        self.spinBox_scale.setObjectName("spinBox_scale")
        
        # self.spinBox_scale.valueChanged.connect(self.valuechange_scale)
        
        self.label_scale = QtWidgets.QLabel(Dialog)
        self.label_scale.setGeometry(QtCore.QRect(460, 450, 381, 16))
        self.label_scale.setText("")
        
#%%         VISUALISIERUNG
        
        self.text_vis = QtWidgets.QLabel(Dialog)
        self.text_vis.setGeometry(QtCore.QRect(200, 550, 111, 22))
        self.text_vis.setObjectName("text_vis")
        
        self.text_vis.setText("Anzahl Bilder Visualisierung")
        
        self.checkBox_vis = QtWidgets.QCheckBox(Dialog)
        self.checkBox_vis.setGeometry(QtCore.QRect(30, 550, 131, 20))
        self.checkBox_vis.setObjectName("checkBox_vis")
        self.checkBox_vis.setText( "Visualisierung ?")

        self.checkBox_vis.stateChanged.connect(self.clickBox_vis)
        
        self.spinBox_vis = QtWidgets.QSpinBox(Dialog)
        self.spinBox_vis.setGeometry(QtCore.QRect(390, 550, 42, 22))
        self.spinBox_vis.setObjectName("spinBox_vis")
        # self.spinBox_vis.valueChanged.connect(self.valuechange_vis)
        
        self.label_vis = QtWidgets.QLabel(Dialog)
        self.label_vis.setGeometry(QtCore.QRect(460, 550, 381, 16))
        self.label_vis.setText("")
        self.label_vis.setObjectName("label_vis")

#%%    Anzahl Bilder Auswertung
        self.text_npic = QtWidgets.QLabel(Dialog)
        self.text_npic.setGeometry(QtCore.QRect(200, 650, 111, 22))
        self.text_npic.setObjectName("text_npic")
        self.text_npic.setText("Anzahl Bilder Auswertung")
        
        self.checkBox_npic = QtWidgets.QCheckBox(Dialog)
        self.checkBox_npic.setGeometry(QtCore.QRect(30, 650, 161, 20))
        self.checkBox_npic.setObjectName("checkBox_anzahlpic")
        self.checkBox_npic.setText("Alle Bilder auswerten? ")
        self.checkBox_npic.stateChanged.connect(self.clickBox_npic)
        
        self.spinBox_npic = QtWidgets.QSpinBox(Dialog)
        self.spinBox_npic.setGeometry(QtCore.QRect(390, 650, 42, 22))
        self.spinBox_npic.setMaximum(2000)
        self.spinBox_npic.setObjectName("spinBox_npic")
        
        # self.spinBox_npic.valueChanged.connect(self.valuechange_npic)
        
        self.label_npic = QtWidgets.QLabel(Dialog)
        self.label_npic .setGeometry(QtCore.QRect(460, 650, 381, 16))
        self.label_npic .setText("")
    
                
#%%     
        self.retranslateUi(Dialog)
        self.buttonBox.accepted.connect(Dialog.accept)
        self.buttonBox.rejected.connect(Dialog.reject)
        QtCore.QMetaObject.connectSlotsByName(Dialog)
        
#%% FUNKTIONEN

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Shadowgraphy Auswertung"))    
#%%
        
    def context_menu(self):
        menu=QtWidgets.QMenu()
        open1=menu.addAction("SELECT")
        open1.triggered.connect(self.open_file)
        cursor=QtGui.QCursor()
        menu.exec_(cursor.pos())    
  
    def open_file(self):
        index1=self.messdaten_treeView.currentIndex()
        file_path=self.model.filePath(index1)
        self.file_path_mess=file_path

    def context_menu_ausw(self):
        menu2=QtWidgets.QMenu()
        open2=menu2.addAction("SELECT")
        open2.triggered.connect(self.select_file_ausw)
        cursor=QtGui.QCursor()
        menu2.exec_(cursor.pos())        
    
    def select_file_ausw(self):
        index2=self.auswertung_treeView.currentIndex()
        file_path_ausw=self.model.filePath(index2)
        self.file_path_ausw=file_path_ausw
    

#%%               
    def clickBox_vis(self, state):
        if state == QtCore.Qt.Checked:
            self.label_vis.setText("Visualisierung an:")   
            self.Vis=True
        else:
            self.label_vis.setText("Visualisierung ausgeschaltet !!!:")
            self.Vis=False
    
    def clickBox_npic(self, state):
        
        if state == QtCore.Qt.Checked:
            self.label_npic.setText("Es werden alle Bilder ausgewertet") 
            self.n_pic_all=True
        else:
            self.label_npic.setText(" Nur bestimmte Anzahl")
            self.n_pic_all=False
            
    def valuechange_vis(self):
        
      # if label_vis.Text=="Visualisierung ausgeschaltet !!!:" :
      #     pass
      # else:
      self.label_vis.setText("Es werden "+str(self.spinBox_vis.value())+" Bilder angezeigt")


    def valuechange_scale(self):
      self.label_scale.setText("Der innere Kreis der Kallibrationsbilder entspricht "+str(self.spinBox_scale.value())+" Pixel")  
      # self.scale=self.spinBox_scale.value()

    def valuechange_npic(self):
      self.label_npic.setText("Es werden "+str(self.spinBox_npic.value())+" Bilder ausgewertet")
      # self.n_pic=self.spinBox_npic.value()
    
      
 #%%

def GUI_Shadowgraphy():
    app = QtWidgets.QApplication(sys.argv)
    Dialog = QtWidgets.QDialog()
    ui = Ui_Dialog()
    ui.setupUi(Dialog)
    Dialog.show()
    app.exec_()
    
    n_calib=ui.spinBox_calib.value()
    scale=ui.spinBox_scale.value()
    n_pic=ui.spinBox_npic.value()
    n_vis=ui.spinBox_vis.value()
    Visual=ui.Vis
    n_pic_all=ui.n_pic_all
    
    folder_ausw=ui.file_path_ausw
    folder    =ui.file_path_mess
    
    print(" ")
    
    if scale==0:
        scale=250
    print(scale , 'Pixel entsprechen 250 um -> scale=', 250/scale)
    scale=250/scale
    
    print('Zur Auswertung werden die ersten', n_calib, 'Ordner übersprungen')
    

    print('Ordner mit Messdaten:' ,folder)
    print('Ordner zur Auswertung:' ,folder_ausw )
    
    return folder, folder_ausw, n_calib, n_vis, scale,n_pic, Visual, n_pic_all

    
    



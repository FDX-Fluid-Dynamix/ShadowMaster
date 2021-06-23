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
from PyQt5.QtWidgets import QApplication,  QRadioButton, QFileSystemModel, QTreeView, QWidget, QFrame
from PyQt5.QtWidgets import QRadioButton
from PyQt5.QtWidgets import QHBoxLayout
from PyQt5.QtWidgets import QVBoxLayout
from PyQt5.QtGui import QIcon, QImage, QPalette, QBrush, QColor


class Ui_Dialog(object):
    
    
    
    def setupUi(self, Dialog):
        
        Dialog.setObjectName("Dialog")
        Dialog.resize(900, 890)
    
        self.image_type="tiff"
        self.Vis=False
        self.n_pic_all=False
        self.file_path_ausw='evaluation/'
        self.file_path_mess='data/'
        self.s_bg=False
        self.number_bg=20
        self.detection_min_score=0.9
        
        
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
        self.text_messdaten.setText("Parent folder with data, Selection with right click")
        

        
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
        self.text2.setText("Parent folder for evaluation, Selection with right click")



#%% Kalibration
        
        x_spin=390
        y_calib=350
        
        
        self.text_calib = QtWidgets.QLabel(Dialog)
        self.text_calib.setGeometry(QtCore.QRect(30, y_calib, 700, 22))        
        self.text_calib.setText("How many calibration folders or other folders not to be evaluated?")

        
        self.spinBox_calib = QtWidgets.QSpinBox(Dialog)
        self.spinBox_calib.setGeometry(QtCore.QRect(x_spin, y_calib, 42, 22))
        self.spinBox_calib.setMaximum(500)

        
#%%     SCALE
        self.text_scale = QtWidgets.QLabel(Dialog)
        self.text_scale.setGeometry(QtCore.QRect(30, 400, 300, 22))
        self.text_scale.setObjectName("text_scale")
        
        self.text_scale.setText("Scaling factor 100mum = ?? Pixel")

        
        self.spinBox_scale = QtWidgets.QSpinBox(Dialog)
        self.spinBox_scale.setGeometry(QtCore.QRect(390, 400, 42, 22))
        self.spinBox_scale.setValue(38)
        self.spinBox_scale.setMaximum(800)
        self.spinBox_scale.setObjectName("spinBox_scale")
        
        # self.spinBox_scale.valueChanged.connect(self.valuechange_scale)
        
        self.label_scale = QtWidgets.QLabel(Dialog)
        self.label_scale.setGeometry(QtCore.QRect(460, 400, 381, 16))
        self.label_scale.setText("")
        
#%%         VISUALISIERUNG
        
        self.text_vis = QtWidgets.QLabel(Dialog)
        self.text_vis.setGeometry(QtCore.QRect(200, 450, 111, 22))
        self.text_vis.setObjectName("text_vis")
        
        self.text_vis.setText("Number of images")
        
        self.checkBox_vis = QtWidgets.QCheckBox(Dialog)
        self.checkBox_vis.setGeometry(QtCore.QRect(30, 450, 131, 20))
        self.checkBox_vis.setObjectName("checkBox_vis")
        self.checkBox_vis.setText( "Visualization ?")
        self.checkBox_vis.setChecked(False)
        self.checkBox_vis.stateChanged.connect(self.clickBox_vis)
        
        self.spinBox_vis = QtWidgets.QSpinBox(Dialog)
        self.spinBox_vis.setGeometry(QtCore.QRect(390, 450, 42, 22))
        self.spinBox_vis.setObjectName("spinBox_vis")
        self.spinBox_vis.setValue(3)
        # self.spinBox_vis.valueChanged.connect(self.valuechange_vis)
        
        self.label_vis = QtWidgets.QLabel(Dialog)
        self.label_vis.setGeometry(QtCore.QRect(460, 450, 381, 16))
        self.label_vis.setText("")
        self.label_vis.setObjectName("label_vis")

#%%    Anzahl Bilder Auswertung
        
        y_npic=500
        
        
        self.text_npic = QtWidgets.QLabel(Dialog)
        self.text_npic.setGeometry(QtCore.QRect(200, y_npic, 111, 22))
        self.text_npic.setObjectName("text_npic")
        self.text_npic.setText("Number of images")
        
        self.checkBox_npic = QtWidgets.QCheckBox(Dialog)
        self.checkBox_npic.setGeometry(QtCore.QRect(30, y_npic, 161, 20))
        self.checkBox_npic.setObjectName("checkBox_anzahlpic")
        self.checkBox_npic.setText("Evaluate all images? ")
        self.checkBox_npic.stateChanged.connect(self.clickBox_npic)
        
        self.spinBox_npic = QtWidgets.QSpinBox(Dialog)
        self.spinBox_npic.setGeometry(QtCore.QRect(390, y_npic, 42, 22))
        self.spinBox_npic.setMaximum(8000)
        self.spinBox_npic.setValue(3)
        self.spinBox_npic.setObjectName("spinBox_npic")
        
        # self.spinBox_npic.valueChanged.connect(self.valuechange_npic)
        
        self.label_npic = QtWidgets.QLabel(Dialog)
        self.label_npic .setGeometry(QtCore.QRect(460, y_npic, 381, 16))
        self.label_npic .setText("")
    
    
    
    
    #%%     Image type 
    
        y_it=550
        self.text_it = QtWidgets.QLabel(Dialog)
        self.text_it.setGeometry(QtCore.QRect(30, y_it, 300, 22))
        self.text_it.setObjectName("text_it")
        
        self.text_it.setText("Image type? ")
        
        
        self.checkBox_tpic1 = QtWidgets.QCheckBox(Dialog)
        self.checkBox_tpic1.setText("tiff (default)")
        self.checkBox_tpic1.setGeometry(QtCore.QRect(180, y_it, 161, 20))
        self.checkBox_tpic1.setObjectName("image_type1")
        self.checkBox_tpic1.stateChanged.connect(self.clickBox_tpic1)
        
        self.checkBox_tpic2 = QtWidgets.QCheckBox(Dialog)
        self.checkBox_tpic2.setText("png? ")
        self.checkBox_tpic2.setGeometry(QtCore.QRect(360, y_it, 161, 20))
        self.checkBox_tpic2.setObjectName("image_type2")
        self.checkBox_tpic2.stateChanged.connect(self.clickBox_tpic2)
        
        self.checkBox_tpic3 = QtWidgets.QCheckBox(Dialog)
        self.checkBox_tpic3.setText("b16 ")
        self.checkBox_tpic3.setGeometry(QtCore.QRect(480, y_it, 161, 20))
        self.checkBox_tpic3.setObjectName("image_type3")
        self.checkBox_tpic3.stateChanged.connect(self.clickBox_tpic3)
        

#%%     Subtract Background
        
        y_bg=600
    
        self.checkBox_bg = QtWidgets.QCheckBox(Dialog)
        self.checkBox_bg.setGeometry(QtCore.QRect(30, y_bg, 150, 20))
        self.checkBox_bg.setObjectName("checkBox_bg")
        self.checkBox_bg.setText( "Background Subtraction ?")

        self.checkBox_bg.stateChanged.connect(self.clickBox_bg)
        
        self.spinBox_bg = QtWidgets.QSpinBox(Dialog)
        self.spinBox_bg.setGeometry(QtCore.QRect(390, y_bg, 42, 22))
        self.spinBox_bg.setObjectName("spinBox_vis")
#        self.spinBox_bg.valueChanged.connect(self.valuechange_bg)
        self.spinBox_bg.setValue(20)
        self.spinBox_bg.setMaximum(250)
        self.label_bg = QtWidgets.QLabel(Dialog)
        self.label_bg.setGeometry(QtCore.QRect(460, y_bg, 381, 16))
        self.label_bg.setText("")
        self.label_bg.setObjectName("label_bg")
        
        
        
#%% Minimal Score

        y_score=650        
        
        self.spinBox_sc = QtWidgets.QDoubleSpinBox(Dialog)
        self.spinBox_sc.setGeometry(QtCore.QRect(390, y_score, 42, 22))
        self.spinBox_sc.setObjectName("spinBox_vis")
        
        self.spinBox_sc.setRange( 0.2, 0.95)
        self.spinBox_sc.setSingleStep(0.05)
        self.spinBox_sc.setValue(0.7) 
        
        self.label_sc = QtWidgets.QLabel(Dialog)
        self.label_sc.setGeometry(QtCore.QRect(30, y_score, 381, 16))
        self.label_sc.setText("Minimum Score for the detected drop")
  



#%%    Settings for the Histogramm
        
        y_hist=700
    
        self.label_h = QtWidgets.QLabel(Dialog)
        self.label_h.setGeometry(QtCore.QRect(30, y_hist, 381, 16))
        self.label_h.setText("Settings for the Histogramm Minimum and Maximum mum")
        self.label_h.setObjectName("label_hist")

        
        self.spinBox_h1 = QtWidgets.QSpinBox(Dialog)
        self.spinBox_h1.setGeometry(QtCore.QRect(360, y_hist, 42, 22))
        self.spinBox_h1.setValue(20)
        self.spinBox_h1.setRange(0,500)
        
        self.spinBox_h2 = QtWidgets.QSpinBox(Dialog)
        self.spinBox_h2.setGeometry(QtCore.QRect(480,y_hist, 42, 22))
        self.spinBox_h2.setRange(0,3000)
        self.spinBox_h2.setValue(500)
        


   

                
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
            self.label_vis.setText("Visualization on")   
            self.Vis=True
        else:
            self.label_vis.setText("Visualization is turn off !!!")
            self.Vis=False
    
    def clickBox_bg(self, state):
        if state == QtCore.Qt.Checked:
            self.s_bg=True
            self.label_bg.setText("Background Subtraction on")
        else:
            self.label_bg.setText("Background Subtraction is turn off !!!")
            self.s_bg=False
            
            
    def clickBox_tpic1(self, state):
        if state == QtCore.Qt.Checked:
            self.image_type="tiff"
    def clickBox_tpic2(self, state):
        if state == QtCore.Qt.Checked:
            self.image_type="png"
    def clickBox_tpic3(self, state):
        if state == QtCore.Qt.Checked:
            self.image_type="b16"
    
    def clickBox_npic(self, state):
        
        if state == QtCore.Qt.Checked:
            self.label_npic.setText("All images are evaluated") 
            self.n_pic_all=True
        else:
            self.label_npic.setText("Only certain number ")
            self.n_pic_all=False
            
    def valuechange_vis(self):
        
      # if label_vis.Text=="Visualisierung ausgeschaltet !!!:" :
      #     pass
      # else:
      self.label_vis.setText(str(self.spinBox_vis.value())+" images are displayed")


    def valuechange_scale(self):
      self.label_scale.setText("Der innere Kreis der Kallibrationsbilder entspricht "+str(self.spinBox_scale.value())+" Pixel")  
      # self.scale=self.spinBox_scale.value()

    def valuechange_npic(self):
      self.label_npic.setText(str(self.spinBox_npic.value() )+"images are evaluated"  )
      # self.n_pic=self.spinBox_npic.value()
#
#    def valuechange_bg(self):
#        self.label_bg.setText(str(self.spinBox_bg.value()) +"images are used to create BG image"   )
      
    def onClicked(self):
        radioBtn = self.sender()
        if radioBtn.isChecked():
            self.text_it.setText("You live in " + radioBtn.text()) 
     
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
    
    image_type=ui.image_type
    
    folder_ausw=ui.file_path_ausw
    folder    =ui.file_path_mess
    
    subtract_bg=ui.s_bg
    n_sbg=ui.spinBox_bg.value()
    
    detection_min_score=ui.spinBox_sc.value()
    print(" ")
    
    if scale==0:
        scale=250
    print( '%i Pixel correspond to 100 um -> scale=%.2f um/pix '%(scale, 100/scale))
    scale=100/scale
    
    hist_min=ui.spinBox_h1.value()
    hist_max=ui.spinBox_h2.value()
    range_hist=(hist_min,hist_max)
    
    
    bins_hist=200
    
    return folder, folder_ausw, n_calib, n_vis, scale,n_pic, Visual, n_pic_all , image_type, subtract_bg , n_sbg, detection_min_score , range_hist, bins_hist

    
    



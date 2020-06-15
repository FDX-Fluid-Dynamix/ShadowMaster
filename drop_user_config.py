# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 12:21:00 2020

@author: student
"""
import glob
import os
import configparser



def configure_NN(file, config_art):
    
    config_file = configparser.ConfigParser()
    config_file .read(file)
    
    print('Sections?',config_file .sections())
    pic_art=config_file[config_art]['pic_art']

    # Linux?
    linux=config_file[config_art].getboolean('linux')
    
    #Background Subtraktion?
    bg_sub=config_file[config_art].getboolean('bg_sub')
    files_bg=int(config_file[config_art]['files_bg'])
    
    # Vorheriger Mean Filter?
    filter_mean=config_file[config_art].getboolean('filter_mean')
    
    # Einstellugen Histogramm
    hist_end=config_file[config_art]['range_grenze']
    range_hist=[0,0]
    range_hist[0]=0
    range_hist[1]=int(hist_end)
    bins_hist=int(config_file[config_art]['bins_hist'])
    
    
    # Grenzwert_Tropfen
    D_grenze           =int(config_file[config_art]['D_grenze'])
    detection_min_score=float(config_file[config_art]['detection_min_score'])
    
    

    # Pfad zum Ordner 
    folder=     config_file[config_art]['folder']
    folder_ausw=config_file[config_art]['folder_ausw']
    n_calib=int(config_file[config_art]['n_calib'])
    
   
        
    # Wie viele Bilder?
    n_pic_all=config_file[config_art].getboolean('n_pic_all')
    n_pic=int(config_file[config_art]['n_pic'])
    
    # Visualisierung
    Visual=           config_file[config_art].getboolean('Visual')
    Visual_anzahl=int(config_file[config_art]['Visual_anzahl'])
    
    
    #Kallibration
    calib_pixel=int(config_file[config_art]['calib_pixel'])
    scale=250/calib_pixel
    

    print(" ")
    print('%i Pixel entsprechen 250 mum -> scale=%.4f mum pro Pixel'%(calib_pixel, scale))
        

        
    #%% Ueberprufung der Pfads zu den Ordner
    if os.path.isdir(folder) ==True :
        subfolders=glob.glob( folder+'/m*')
        
##____________________ ############       
        if config_file[config_art].getboolean('Manuelle_subfolder_Anpassung')==True:
            subfolders=[subfolders[0],subfolders[7], subfolders[47] ]
##____________________############  

        name_subfolders=subfolders.copy()
        for i_sfol in range(0, len(subfolders)):  
            name_subfolders[i_sfol]=subfolders[i_sfol].replace(folder+'\\', '')
            if linux:
                name_subfolders[i_sfol]=subfolders[i_sfol].replace(folder+'/', '')
        print(" ")
        print("Folgende Ordner sind in Berabeitung:" )
        print(*subfolders, sep = "\n")
    else:
        raise ValueError("Der Ordner mit den Messdaten : %s 'ist nicht vorhanden !!!" %folder)
      
    if os.path.isdir(folder_ausw) ==True :
        subfolders_ausw=glob.glob(folder_ausw+'/*')
        if len(subfolders_ausw)!=len(subfolders):
            for names in name_subfolders:
                if os.path.isdir(folder_ausw+'//'+names)==False:
                    os.mkdir(folder_ausw+'//'+names)
            subfolders_ausw=glob.glob(folder_ausw+'/*')
            print(" ")
            print("Die Ergebnisse befinden sich in :")
            print(*subfolders_ausw, sep = "\n") 
    else:
        raise ValueError("Der Zielordner für die Auswertung %s ist nicht vorhanden !!! " %folder_ausw)
    
   
    #%%   Überprüfung der Einstellunegn  
        
    print(" ")
    if n_pic_all==True:
        print('Es werden alle Bilder des Formats (.%s) ausgewertet' %pic_art)
    else:
        print('Es werden die ersten %i Bilder ausgewertet' %n_pic)  
    
    print('Visual?', Visual)
    if Visual==True:
        print('Es werden die ersten %i Bilder mit Tropfen angezeigt' %Visual_anzahl)
    else:
        print('Es werden KEINE Bilder angezeigt')
     
    print('Es werden Tropfen mit über %i mum verworfen' %D_grenze)    
    print('Histogrammerstellung: Im Bereich %i bis %i mum  mit %i bins'  %(range_hist[0], range_hist[1],bins_hist))     
    print('Minimum Score für Detection %f '  %(detection_min_score) )

    if bg_sub==True:
        print('Es wird eine Background Substraktion  mit einer Mittelung über %i Bilder durchgefuehrt' %(files_bg))
        print(" ") 
    
    if filter_mean==True:
        print('Es wird ein Median Filter auf die Bilder angewendet')
        
    
    print(" ") 
    print('Richtige Einstellunegn? Ja=0, Nein=1')
    eing = input()
    
    if int(eing) ==1:
        raise ValueError(" Abbruch durch Benutzer, falsche Auswahl getroffen")   
    return subfolders, subfolders_ausw, bg_sub,name_subfolders,D_grenze,files_bg, n_calib, Visual_anzahl, scale,n_pic, Visual, n_pic_all, detection_min_score, pic_art, bins_hist, range_hist, filter_mean

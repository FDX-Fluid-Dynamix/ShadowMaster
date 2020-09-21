
##
from drop_user_config import configure_NN
##

import os
import sys
import cv2
import glob
import time
import copy
import drops
import scipy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.interactive(True)

# Root directory of the project
ROOT_DIR = os.path.abspath("")
# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import visualize
import mrcnn.model as modellib


#%% Einstellungen durch den Benutzer

config_file='F:/Shadography TU Berlin/config_tu_berlin_m01_m47_m07.ini'
print('Config file vorhanden ?',os.path.isfile(config_file))
config_art='TU_Berlin'
subfolders, subfolders_ausw, bg_sub,name_subfolders,D_grenze,files_bg, n_calib, Visual_anzahl, scale,n_pic, Visual, n_pic_all, detection_min_score, pic_art, bins_hist, range_hist, filter_mean=configure_NN(config_file, config_art)

#%% Funktioenn
def get_ax(rows=1, cols=1, size=16):
    """Return a Matplotlib Axes array to be used in all visualizations in the notebook.
    """
    _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    return ax

def hist_erstellen( D_phy , name, bi , ran , title) :  
    SMD=sum(D_phy**3)/(sum(D_phy**2))
    plt.hist( D_phy, bins=bi, range=ran, density=True, edgecolor="k",color="b") 
    plt.text(plt.xlim()[1]*0.4, plt.ylim()[1]*0.8, 'D_mean=' + "{:.0f}".format(D_phy.mean()) + '$\mu$m', fontsize=10) 
    plt.text(plt.xlim()[1]*0.4, plt.ylim()[1]*0.7, 'SMD=' + "{:.0f}".format(SMD) + '$\mu$m', fontsize=10)
    plt.text(plt.xlim()[1]*0.4, plt.ylim()[1]*0.6, 'D_count=' + "{:.0f}".format(len(D_phy)), fontsize=10) 
    plt.xlabel('Durchmesser in $\mu$m ')       
    plt.savefig(name+str(i_pic)+'.png', bbox_inches='tight', dpi=300)
    plt.title(title)
    plt.show()

#%% Konfiguration

TEST_MODE = "inference"
config = drops.DropConfig()
class InferenceConfig(config.__class__):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
 
 
config = InferenceConfig()
##
config.DETECTION_MIN_CONFIDENCE=detection_min_score
##
config.display()


#%%  Netz laden
DROP_WEIGHTS_PATH = os.path.join(ROOT_DIR, "logs/mask_rcnn_drops.h5")  # Pfad zu den verwendeten Gewichten

##
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
DROP_DIR = os.path.join(ROOT_DIR, "datasets/droplets/")         # Ordner mit Datensatz

dataset = drops.DropDataset()
dataset.load_drop(DROP_DIR , "val")
dataset.prepare()

print("Images: {}\nClasses: {}".format(len(dataset.image_ids), dataset.class_names))

DEVICE = "/cpu:0" 
import tensorflow as tf
with tf.device(DEVICE):
    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

print("Loading weights ", DROP_WEIGHTS_PATH)
model.load_weights(DROP_WEIGHTS_PATH, by_name=True, exclude=None)

#%%

for i_folder in range(0,len(subfolders)) :   
    
    subfolder =subfolders[i_folder]
    ausw      =subfolders_ausw[i_folder]
    print('Folgender Ordner ist in Bearbeitung: ', subfolder)
    print('Abspeicher in                      : ', ausw)
	
    name_sub  =name_subfolders[i_folder]
    name='Shadowgraphy_'+name_sub+'_'
    fileNames=glob.glob(subfolder+'/*.'+pic_art)  
    
    if len(fileNames)==0:
        raise ValueError('Keine Daten in %s gefunden !!!' %subfolder)

    if n_pic_all==False:
        file_end=n_pic 
        fileNames=fileNames[0:file_end]
       
    ERG = pd.DataFrame({'Bildnummer':pd.Series(),'Score':pd.Series(),'Durchmesser_mask':pd.Series() ,'Durchmesser_box': pd.Series() , 'Ratio' :pd.Series()})
    Tropfen_proBild = pd.DataFrame({'Anzahl_Tropfen':pd.Series() ,'Mittlerer Durchmesser': pd.Series() })
    
    ###_______   Background erzeugen __________###
    if len(fileNames)<files_bg:
        files_bg=len(fileNames)
        
    if bg_sub==True:  
        for i_pic_bg in range(0,files_bg):
            image_bg=plt.imread(fileNames[i_pic_bg])
            image_bg=image_bg+abs(image_bg.min())
            image_bg=(image_bg/(image_bg.max()))*255
            if i_pic_bg==0:
                BG=(1/files_bg)*image_bg
                continue  
            if i_pic_bg%10==0:
                print(i_pic_bg)
            BG+=(1/files_bg)*image_bg
    
        plt.imshow(BG, cmap='gray')
        plt.title('Hintergrund')
        plt.savefig(ausw+'/'+name+'Background.png', bbox_inches='tight', dpi=900)
        plt.show()
        plt.close('all') 
    
        
    
    else:
        BG=np.zeros_like(plt.imread(fileNames[0]))

#%%	
    ###_______   Schleife durch Bilder __________###
    i_pic_vis=0
    file_end=len(fileNames)
    for i_pic in range(0,len(fileNames)):     
        
        t1 = cv2.getTickCount()
        
        erg_pro_pic = pd.DataFrame({'Bildnummer':pd.Series(),'Score':pd.Series(),'Durchmesser_mask':pd.Series() ,'Durchmesser_box': pd.Series() , 'Ratio' :pd.Series()})
        image_orginal=plt.imread(fileNames[i_pic])
        image=copy.deepcopy(image_orginal)
        
        image=image+abs(image.min())
        image=(image/(image.max()))*255 
        
        image=image-BG
        image=image+abs(image.min())
        image=(image/(image.max()))*255  
        image=image.astype(float)
        
        ####
        if  filter_mean==True:
            image= scipy.ndimage.median_filter(image, size=2)
            config.MEAN_PIXEL=[image.mean() , image.mean(),image.mean()]
        ###
        image = image[..., np.newaxis]
        results = model.detect([image], verbose=0)  
        r = results[0]

		###_______   Berechnung der Durchmesser, wenn Tropfen erkannt wurden  __________#### 
        if r['scores'].size > 0:
            boxes=r['rois']
            droplets=r['masks']*1
 			
            for i_drop in range(0, len(boxes) ):    
                #Erste Berechnungsart über "bounding box" :
                b1=boxes[i_drop]
                x=b1[2]-b1[0]
                y=b1[3]-b1[1]
                d=((x+y)/2)
                ratio=np.max([x,y])/np.min([x,y])        
                
                #Zweite Berechnugsart  über "mask" (Erkannte Fläche)
                f=np.sum(droplets[:,:,i_drop])
                d_m=np.sqrt( (4*f)/np.pi)
                erg_pro_pic =erg_pro_pic.append(pd.DataFrame({'Bildnummer':pd.Series(i_pic+1),'Score':pd.Series(r['scores'][i_drop]),'Durchmesser_mask':pd.Series(scale*d_m) ,'Durchmesser_box': pd.Series(scale*d) , 'Ratio' :pd.Series(ratio)}), ignore_index=True)
#___________
            erg_pro_pic=erg_pro_pic[erg_pro_pic['Durchmesser_box']<D_grenze]
#___________
            ERG=ERG.append(erg_pro_pic,  ignore_index=True)
   
		    ###_______   Viszalisierung der Ergebnisse  __________#### 
            if Visual:
                if ((i_pic_vis) <Visual_anzahl) :
                    plt.close('all') 
                    
                    plt.imshow(image_orginal[:,:], cmap='gray')
                    plt.axis('off')
                    plt.savefig(ausw+'/'+name+str(i_pic+1)+'_orginal.png', bbox_inches='tight', dpi=900)
                    plt.title('Orginalbild')
                    plt.show()
                    ax = get_ax(1) 
                    visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], dataset.class_names, r['scores'], ax=ax,title="Predictions")
                    plt.savefig(ausw+'/'+name+str(i_pic+1)+'_erkannt.png', bbox_inches='tight', dpi=900)
                    plt.show()
                    i_pic_vis+=1
		
        
        ###_______   Auswertung des eines Bildes __________#### 
        plt.close('all') 
        
        if r['scores'].size > 0:
            anazhl_tropfen=pd.DataFrame({'Anzahl_Tropfen':pd.Series(len(erg_pro_pic['Durchmesser_box'])) ,'Mittlerer Durchmesser': pd.Series(erg_pro_pic['Durchmesser_box'].mean()) })
        else:
            anazhl_tropfen=pd.DataFrame({'Anzahl_Tropfen':pd.Series(0) ,'Mittlerer Durchmesser': pd.Series(0) })
            
        Tropfen_proBild=Tropfen_proBild.append(anazhl_tropfen,  ignore_index=True)

        
        t2 = cv2.getTickCount()
        print('Zeit für  Bild: '+str(i_pic+1)+' : t= '+ str((t2-t1)/cv2.getTickFrequency())[0:4] +' Sekunden')

		###_______   Histogramme  __________#### 
        if Visual and ( len(ERG)>0 and ( i_pic==Visual_anzahl or i_pic==file_end-1 or i_pic==int(file_end/2) or i_pic==i_pic==int(file_end/4)  ) ) :
            plt.close('all') 
            hist_name=ausw+'/'+'Hist_box'+name
            hist_erstellen(ERG['Durchmesser_box'], hist_name, bi=bins_hist, ran=range_hist, title='Histogramm mit Durchmesser ueber boundig box')
            hist_name=ausw+'/'+'Hist_mask'+name
            hist_erstellen(ERG['Durchmesser_mask'], hist_name, bi=bins_hist, ran=range_hist, title='Histogramm mit Durchmesser ueber boundig box')
           
            ###_______   Auswertung pro Bild  __________#### 
            plt.subplot(2,1,1)
            plt.plot(Tropfen_proBild['Anzahl_Tropfen'], '*-')
            plt.title( 'Anzahl Tropfen pro Bild')
            plt.ylabel('Anzahl Tropfen ')  
            plt.subplot(2,1,2)
            plt.plot(Tropfen_proBild['Mittlerer Durchmesser'],'*-', label='Mittlerer Durchmesser pro Bild')
            plt.title( 'Mittlerer Durchmesser pro Bild')
            plt.xlabel('Bild')
            plt.ylabel('Mittlerer Durchmesser in $\mu$m ')  
            plt.savefig(ausw+'/'+'tropfen_pro_bild'+str(i_pic+1)+name)
            plt.show()
            plt.close('all') 
            
            name_h5=ausw+'/Ergebnisse_in _um bis_Bild'+str(i_pic+1)+name+'.h5'
            key=name_sub
            ERG.to_hdf(name_h5, key)
            print('Ergebnisse unter ', name_h5,'mit key=', key)
            

    ###_______   Finales Abspeichern pro Datensatz __________####         
    name_h5=ausw+'/Ergebnisse_in _um bis_Bild'+str(i_pic+1)+name+'.h5'
    key=name_sub
    ERG.to_hdf(name_h5, key)
    print('Ergebnisse unter ', name_h5,'mit key=', key)
    
    name_h5=ausw+'/Tropfen_pro_Bild'+name+'.h5'
    Tropfen_proBild.to_hdf(name_h5, key)
    print('Ergebnisse unter ', name_h5,'mit key=', key)


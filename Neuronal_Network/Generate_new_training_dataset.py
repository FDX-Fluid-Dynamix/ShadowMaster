"""
Algorithm to automatically geneate training dataset

"""

#%%#########################################################
#  Packages
############################################################

import os
import sys
import cv2
import glob
import copy
import drops
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib

matplotlib.interactive(True)

# Root directory of the project
ROOT_DIR = os.path.abspath("")
# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
import mrcnn.model as modellib



#%%#########################################################
#  Manual Settings by the user
############################################################


# Number of train_data
n_train=1
n_val=1


#Parent folder with folders with Shadowgraphy images
folder='data/test1'
#Parent folder for the evaluation

folder_result='datasets_new/droplets/'

image_type='tiff'
fileNames=glob.glob(folder+'/*.'+image_type)  

if len(fileNames)==0:
    raise ValueError('No data in %s found!!!' %folder)
    
if len(fileNames)<(n_train+n_val):
    raise ValueError('Not enough images (%i) for train and validation (%i)' %(len(fileNames), n_train+n_val))




#Minimum Score for the detected drop
detection_min_score=0.9



#Test folder 
if os.path.isdir(folder) !=True :
    raise ValueError("The folder : %s does nit exist !!! \n" %folder)

#Test result folder  
if os.path.isdir(folder_result) !=True :
    raise ValueError("The folder for the final result: %s does not exist!!! " %folder_result)
 
    
#%%#########################################################
#  Configurations of the neuronal network
############################################################
config = drops.DropConfig()
class InferenceConfig(config.__class__):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
 
config = InferenceConfig()
##
config.DETECTION_MIN_CONFIDENCE=detection_min_score
##
config.display()


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


#%%


wdir_main=os.getcwd()   



#%%#########################################################
#  Creating the neuronal network
############################################################
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
DROP_DIR = os.path.join(ROOT_DIR, "datasets/droplets/")         
dataset = drops.DropDataset()
dataset.load_drop(DROP_DIR , "val")
dataset.prepare()
print("Images: {}\nClasses: {}".format(len(dataset.image_ids), dataset.class_names))

DEVICE = "/cpu:0" 
with tf.device(DEVICE):
    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Path to the weights of the network
DROP_WEIGHTS_PATH = os.path.join(ROOT_DIR, "logs/mask_rcnn_drops.h5")  
print("Loading weights ", DROP_WEIGHTS_PATH)
model.load_weights(DROP_WEIGHTS_PATH, by_name=True, exclude=None)



##########################################################
#  Loop over random images
############################################################

for i_random in range(0,n_train+n_val):   
    
    if i_random==0:
        
        folder_png=wdir_main+'/'+folder_result+'train/'
        
        if os.path.isdir(folder_png) !=True :
            os.makedirs(folder_png)
        os.chdir(folder_png)
        
        pic_art='train'
        s_json='{'
        
    if i_random==n_train:
        s_json+='}'
        datei = open('via_region_data.json','w')
        datei.write(s_json)
        datei.close()
        
        
        folder_png=wdir_main+'/'+folder_result+'val/'
        if os.path.isdir(folder_png) !=True :
            os.makedirs(folder_png)
            
        os.chdir(folder_png)
        pic_art='val'
        s_json='{'
    
    
    pic_name='image_%s_%i.png' %(pic_art, i_random)
        
    
    i_pic=np.random.randint(0, len(fileNames)-1)
    
    t1 = cv2.getTickCount()

    erg_pro_pic = pd.DataFrame({'Bildnummer':pd.Series(),'Score':pd.Series(),'Durchmesser_mask':pd.Series() ,'Durchmesser_box': pd.Series() , 'Ratio' :pd.Series()})
    
    
    ###############
    
    path_file=os.path.abspath(wdir_main+'/'+fileNames[i_pic])
    
    image_orginal=plt.imread(path_file)
    
    
    image=copy.deepcopy(image_orginal)
    
    if len(image.shape)>2:
        image=image[:,:,0]
    
    #Flip image:
    np.flip(image,1)
    matplotlib.image.imsave(pic_name, image, cmap='gray')
 
    size_pic=os.path.getsize(pic_name)

    image=image+abs(image.min())
    image=(image/(image.max()))*255 
    

    image = image[..., np.newaxis]
    
    results = model.detect([image], verbose=0)  
    r = results[0]
    
    
    ####
    
    
    if i_random==0 or i_random==n_train:
        s_pic='"%s":{"filename":"%s","size":%i,' %(pic_name+str(size_pic), pic_name, size_pic)
    else:
        s_pic=',"%s":{"filename":"%s","size":%i,' %(pic_name+str(size_pic), pic_name, size_pic)
        
    ######
    

		###_______   Berechnung der Durchmesser, wenn Tropfen erkannt wurden  __________#### 
    if r['scores'].size > 0:
        
        
        boxes=r['rois']
        droplets=r['masks']*1
        
        DROPS=np.sum(droplets, axis=2)
        
        plt.imshow(DROPS)
       
        for i_drop in range(0,len(boxes) ):    

            drop=droplets[:,:,i_drop]
            
            D=(np.gradient(drop, axis=1))
            X=np.where(D>0)[1][::2]
            Y=np.where(D>0)[0][::2]
            X2=np.where(D<0)[1][::2]
            Y2=np.where(D<0)[0][::2]
            
            liste_xps=np.hstack((X,X2[::-1]))
            liste_yps=np.hstack((Y,Y2[::-1]))
            
            
            liste_xp='['
            liste_yp='['
            for x  in liste_xps:
                liste_xp+='%i,'%x
            for  y in  liste_yps:
                liste_yp+='%i,'%y
            
            liste_xp=liste_xp[:-1]+']'
            liste_yp=liste_yp[:-1]+']'
            
            if i_drop==0:
                s_pic+='"regions":[{"shape_attributes":{"name":"polygon","all_points_x":%s,"all_points_y":%s}' %( liste_xp, liste_yp)
            else:
                s_pic+=',"region_attributes":{}},{"shape_attributes":{"name":"polygon","all_points_x":%s,"all_points_y":%s}' %( liste_xp, liste_yp)
            
        s_pic+=',"region_attributes":{}}],"file_attributes":{}}'   
        s_json+=s_pic
        
        if i_drop!=len(boxes)-1:
            s_json+=','
            
    np.delete(fileNames, i_pic)  
    
s_json+='}'
datei = open('via_region_data.json','w')
datei.write(s_json)
datei.close()
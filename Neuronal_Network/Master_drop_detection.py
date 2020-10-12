"""
Algorithm to detect the drop size from Shadowgraphy Images.

"""

#%%#########################################################
#  Packages
############################################################

import os
import sys
import cv2
import glob
import time
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
from mrcnn import visualize
import mrcnn.model as modellib


#%%#########################################################
#  Manual Settings by the user
############################################################

gui=False
bg_sub=False
files_bg=25
range_hist=(0,350)
bins_hist=200
D_grenze=250
detection_min_score=0.9
image_type='tiff'

folder='C:/Users/student/Marla/Test_VGL_NN_Algorithmus'
folder_result='C:/Users/student/Marla/Auswertung_test_vgl/mask_r_cnn/'
n_calib=0
n_pic_all=False
n_pic=20

n_visual=10

calib_pixel=94
scale=250/calib_pixel
Visual=False


#%%#########################################################
#  Testing all settings
############################################################

print("The following settings have been selected: \n")  
print('%i Pixel equivalent to 250 \xb5m-> scale=%.4f \xb5m per pixel \n'%(calib_pixel, scale))

#Test folder 
if os.path.isdir(folder) ==True :
    subfolders=glob.glob( folder+'/*')
    subfolders=subfolders[n_calib:n_calib+1]         
    name_subfolders=subfolders.copy()
    for i_sfol in range(0, len(subfolders)):  
        name_subfolders[i_sfol]=os.path.basename(subfolders[i_sfol])
    print("The following folders are evaluated." )
    print(*subfolders, sep = "\n")
else:
    raise ValueError("The folder : %s does nit exist !!! \n" %folder)

#Test result folder  
if os.path.isdir(folder_result) ==True :
    subfolders_result=glob.glob(folder_result+'/*')
    if len(subfolders_result)!=len(subfolders):
        for names in name_subfolders:
            if os.path.isdir(folder_result+'//'+names)==False:
                os.mkdir(folder_result+'//'+names)
        subfolders_result=glob.glob(folder_result+'/*')
        subfolders_result=subfolders_result[0:len(subfolders)]
        
    print(" ")
    print("The results are saved in the following folders:")
    print(*subfolders_result, sep = "\n") 
else:
    raise ValueError("The folder for the final result: %s does not exist!!! " %folder_result)
 
    
print(" ")

if n_pic_all==True:
     print("All images from the type '.%s' are evaluated. \n" %image_type)
else:
    print('The first %i images are evaluated \n' %n_pic)  

print('Only drops with a minimum score over %.2f will be regard.\n' %detection_min_score)  

if Visual==True:
     print('The result of the first %i images is displayed \n' %n_visual)
else:
    print('No result images are displayed \n')
             
print('Histogramm in the range of %i to %i \xb5m with %i bins.\n'  %(range_hist[0], range_hist[1],bins_hist))     

if bg_sub==True:
    print('A background subtraction is performed with an averaging over %i images. \n' %(files_bg))

print('Are all the settings correct? Yes=0, No=1')
settings_correct= input()
if int(settings_correct) ==1:
    raise ValueError(" Cancellation by user because incorrect settings were selected.")


#%%#########################################################
#  Functions
############################################################
def get_ax(rows=1, cols=1, size=16):
    """Return a Matplotlib Axes array to be used in all visualizations in the notebook.
    """
    _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    return ax

def hist_erstellen( D_phy , name, bi , ran , title) :  
    """Creating a histogram with the results of the drop diameters
    """
    SMD=sum(D_phy**3)/(sum(D_phy**2))
    plt.hist( D_phy, bins=bi, range=ran, density=True, edgecolor="k",color="b") 
    plt.text(plt.xlim()[1]*0.4, plt.ylim()[1]*0.8, 'D_mean=' + "{:.0f}".format(D_phy.mean()) + '\xb5m', fontsize=10) 
    plt.text(plt.xlim()[1]*0.4, plt.ylim()[1]*0.7, 'SMD=' + "{:.0f}".format(SMD) + '\xb5m', fontsize=10)
    plt.text(plt.xlim()[1]*0.4, plt.ylim()[1]*0.6, 'D_count=' + "{:.0f}".format(len(D_phy)), fontsize=10) 
    plt.xlabel('Diameter in \xb5m ')       
    plt.savefig(name+str(i_pic)+'.png', bbox_inches='tight', dpi=300)
    plt.title(title)
    plt.show()



def creating_background_image(fileNames , n_files_bg):
    """Creating a background image as the mean value of n images
    """
    if len(fileNames) < files_bg:
        n_files_bg=len(fileNames)

    for i_pic_bg in range(0,n_files_bg):
        image_bg=plt.imread(fileNames[i_pic_bg])
        image_bg=image_bg+abs(image_bg.min())
        image_bg=(image_bg/(image_bg.max()))*255
        if i_pic_bg==0:
            BG=(1/n_files_bg)*image_bg
            continue  
        BG+=(1/n_files_bg)*image_bg

    plt.imshow(BG, cmap='gray')
    plt.title('Hintergrund')
    plt.savefig(result_folder+'/'+name+'Background.png', bbox_inches='tight', dpi=900)
    plt.show()  
    
    return BG
    

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

#%%#########################################################
#  Main loop over all folders
############################################################

for i_folder in range(0,len(subfolders)) :   
    
    subfolder =subfolders[i_folder]
    result_folder=subfolders_result[i_folder]
    print('The folder %s in progress. ' %subfolder)
    print('The results will be save in %s ' %result_folder)
	

    fileNames=glob.glob(subfolder+'/*.'+image_type)  
    
    if len(fileNames)==0:
        raise ValueError('Keine Daten in %s gefunden !!!' %subfolder)

    if n_pic_all==False and n_pic< len(fileNames):
        file_end=n_pic 
        fileNames=fileNames[0:file_end]
    else:
        file_end=len(fileNames)
    
    name_sub  =name_subfolders[i_folder]
    name='Shadowgraphy_'+name_sub+'_'
    
    Result_panda = pd.DataFrame({'Imagenumber':pd.Series(),'Score':pd.Series(),'Diameter_mask':pd.Series() ,'Diameter_box': pd.Series() , 'Ratio' :pd.Series()})
    Drops_pro_pic = pd.DataFrame({'Number_of_drops':pd.Series() ,'Mean_diameter': pd.Series() })
    
    ###_______   Create background __________###

        
    if bg_sub==True:  
        BG=creating_background_image(fileNames , files_bg)
    else:
        BG=np.zeros_like(plt.imread(fileNames[0]))
    
    i_pic_vis=0
    
#%%#########################################################
#  Loop over all images
############################################################
    
    for i_pic in range(0, int(file_end) ):     
        t1 = cv2.getTickCount()
        
        erg_pro_pic = pd.DataFrame({'Imagenumber':pd.Series(),'Score':pd.Series(),'Diameter_mask':pd.Series() ,'Diameter_box': pd.Series() , 'Ratio' :pd.Series()})
        
        image_org=plt.imread(fileNames[i_pic])
        image=copy.deepcopy(image_org)
        
        image=image+abs(image.min())
        image=(image/(image.max()))*255 
        
        image=image-BG
        image=image+abs(image.min())
        image=(image/(image.max()))*255  
        image=image.astype(float)
        

        image = image[..., np.newaxis]
        results = model.detect([image], verbose=0)  
        r = results[0]

		#Calculation of the drop diameters if drops have been detected 
        if r['scores'].size > 0:
            boxes=r['rois']
            droplets=r['masks']*1
            
            for i_drop in range(0, len(boxes) ):    
                #First way over the "bounding box" 
                b1=boxes[i_drop]
                x=b1[2]-b1[0]
                y=b1[3]-b1[1]
                d=((x+y)/2)
                ratio=np.max([x,y])/np.min([x,y])        
                
                #Second way over the "mask" 
                f=np.sum(droplets[:,:,i_drop])
                d_m=np.sqrt( (4*f)/np.pi)
                
                erg_pro_pic =erg_pro_pic.append(pd.DataFrame({'Imagenumber':pd.Series(i_pic+1),'Score':pd.Series(r['scores'][i_drop]),'Diameter_mask':pd.Series(scale*d_m) ,'Diameter_box': pd.Series(scale*d) , 'Ratio' :pd.Series(ratio)}), ignore_index=True)
    
            Result_panda=Result_panda.append(erg_pro_pic,  ignore_index=True)
   
		    #Visualization of the results
            if Visual:
                if ((i_pic_vis) <n_visual) :
                    plt.close('all') 
                    
                    plt.imshow(image_org[:,:], cmap='gray')
                    plt.axis('off')
                    plt.savefig(result_folder+'/'+name+str(i_pic+1)+'_originall.png', bbox_inches='tight', dpi=900)
                    plt.title('Original image')
                    plt.show()
                    ax = get_ax(1) 
                    visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], dataset.class_names, r['scores'], ax=ax,title="Predictions")
                    plt.savefig(result_folder+'/'+name+str(i_pic+1)+'_result.png', bbox_inches='tight', dpi=900)
                    plt.show()
                    i_pic_vis+=1
		
        
        #Save results per each image
        plt.close('all') 
        
        if r['scores'].size > 0:
            anazhl_tropfen=pd.DataFrame({'Number_of_drops':pd.Series(len(erg_pro_pic['Diameter_box'])) ,'Mean_diameter': pd.Series(erg_pro_pic['Diameter_box'].mean()) })
        else:
            anazhl_tropfen=pd.DataFrame({'Number_of_drops':pd.Series(0) ,'Mean_diameter': pd.Series(0) })
            
        Drops_pro_pic=Drops_pro_pic.append(anazhl_tropfen,  ignore_index=True)

        
        t2 = cv2.getTickCount()
        print('Time for picture %i  %.2f s' %( i_pic+1, (t2-t1)/cv2.getTickFrequency()) )


#%%#########################################################
#  Save and plot results
############################################################
		
        #Histogram creation and saving of the results
        if Visual and  len(Result_panda)>0  and ( i_pic==n_visual  or i_pic==int(file_end/2) or i_pic==i_pic==int(file_end/4) or i_pic==int(file_end-1) )  :
            plt.close('all') 
            hist_name=result_folder+'/'+'Hist_box'+name
            hist_erstellen(Result_panda['Diameter_box'], hist_name, bi=bins_hist, ran=range_hist, title='Histogramm with diameter over boundig box')
            hist_name=result_folder+'/'+'Hist_mask'+name
            hist_erstellen(Result_panda['Diameter_mask'], hist_name, bi=bins_hist, ran=range_hist, title='Histogramm with diameter over boundig box')
           
            
            f, ax_plot=plt.subplots(2,1)
            
            ax_plot[0].plot(Drops_pro_pic['Number_of_drops'], '*-')
            ax_plot[0].set_ylabel('Number of drops ')  
       
            ax_plot[1].plot(Drops_pro_pic['Mean_diameter'],'*-', label='Mean_diameter pro Bild')
            ax_plot[1].set_ylabel( 'Diameter in \xb5m')
            ax_plot[1].set_xlabel('Number of image')

            plt.savefig(result_folder+'/'+name+'drops_per_image_till_'+str(i_pic+1))
            plt.show()
            plt.close('all') 
            
            key=name_sub
            
            name_h5=result_folder+'/'+name+'results_till_image_'+str(i_pic+1)+'.h5'
            Result_panda.to_hdf(name_h5, key)

            name_h5=result_folder+'/'+name+'drops_per_iamge_till_image_'+str(i_pic+1)+'.h5'
            Drops_pro_pic.to_hdf(name_h5, key)
            

#%%#########################################################
#  End of the loop over the images and final save of the results
############################################################           
    name_h5=result_folder+'/'+name+'results_till_image_'+str(i_pic+1)+'.h5'
    Result_panda.to_hdf(name_h5, key)
    print('Results are save as: %s with key=%s' %(name_h5, key) )
    np.savetxt(result_folder+'/results_mask_till_image_'+str(i_pic)+name+'.txt',np.array(Result_panda['Diameter_mask']))
    np.savetxt(result_folder+'/results_box_till_image_'+str(i_pic)+name+'.txt',np.array(Result_panda['Diameter_box']))
    
    


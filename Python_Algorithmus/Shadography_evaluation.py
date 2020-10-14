"""
Algorithmus to detect the drop size from Shadowgraphy Images.

"""

import matplotlib.pyplot as plt
import glob
import cv2
import os
import numpy as np
import functions 

############################################################
#%%  Manual settings by the user
############################################################

#Parent folder with folders with Shadowgraphy images
folder='data/'

#Parent folder for the evaluation
evaluation_path=('evaluation/')


# Number of images for evaluation
evaluate_all_files=True
# Only if evaluate_all_files==False: Number of images for evaluation
file_end=2

# Background subtraction 
sub_bg=False
# Number of images for calculating the background
n_pic_bg=20



#########  Settings for detection and filtering ######### 

# From calibration Picture mum per pixel 
scale=250/94  

# Minimum Droplet radius in pixel   
min_droplet_size=7

# Tresh Value for the binary image
tresh=125  

# Minimal ratio=min(bounding box)/max(bounding box)
min_ratio=0.79  

# Minimum mean value of a drop        
min_mw=60

# Minimum difference between center and edge for a sharp drop
min_dif=90


#Max value in the histogram  
drop_max=350    


#########    Different plot variants ######### 

# Plot the orginal image and the final result
plotten_erg    =True

# Plot all Steps for the Detection
plot_all_steps=False       


# Plot result of the filter over the difference between center and edge of each drop
plot_circ      =False  

# Plot result of the contour and ratio filter for each drop  
plot_cont      =False

# Post-processing , histogram of droplets diameters
postproc       =True 

# Save Images ?
save_img       =True

# Number of Pictures to save for each folder
n_save_img     =12

 

############################################################
# %%  Testing all settings
############################################################

#Test folder 
if os.path.isdir(folder) ==True :
    subfolders=glob.glob( folder+'/*')
    subfolders=subfolders        
    name_subfolders=subfolders.copy()
    for i_sfol in range(0, len(subfolders)):  
        name_subfolders[i_sfol]=os.path.basename(subfolders[i_sfol])
    print("The following folders are evaluated." )
    print(*subfolders, sep = "\n")
else:
    raise ValueError("The folder : %s does nit exist !!! \n" %folder)


if os.path.isdir(evaluation_path) ==True :
    evaluation_folders=glob.glob(evaluation_path+'/*')
    if len(evaluation_folders)!=len(subfolders):
        for names in name_subfolders:
            if os.path.isdir(evaluation_path+'//'+names)==False:
                os.mkdir(evaluation_path+'//'+names)
        evaluation_folders=glob.glob(evaluation_path+'/*')
        evaluation_folders=evaluation_folders[0:len(subfolders)]
    
    print(" ")
    print("The results are saved in the following folders:")
    print(*evaluation_folders, sep = "\n") 
else:
    raise ValueError("The folder for the final result: %s does not exist!!! " %evaluation_path)



Detecion_class=functions.DROP_DETECTION(scale,min_droplet_size, min_ratio, min_mw , min_dif)
Preprocessing=functions.PreProcess(tresh)
plot_func=functions.PLOT_Shadowgraphy()
DROPS=functions.DROPS_class()




print('\n Are all the settings correct? Yes=0, No=1')
settings_correct= input()
if int(settings_correct) ==1:
    raise ValueError(" Cancellation by user because incorrect settings were selected.")

############################################################
#%%  Main part
############################################################

plt.close("all")

for i_folder, subfolder in enumerate(subfolders):

    fileNames=glob.glob(subfolder+'/*.tiff') 
    evaluation_folder=evaluation_folders[i_folder]+'/'
    pic_name=os.path.basename(subfolder)
    
    if i_folder==0:
        test_pic=np.array(cv2.imread(fileNames[0],0))
        Detecion_class.shape1, Detecion_class.shape2=test_pic.shape
    
    
    if len(fileNames)==0:
        print("The folder  %s contains no images !!!"  %folder) 
        continue

    if sub_bg==True:
        Preprocessing.bg_erzeugen(fileNames,  n_pic_bg , evaluation_folder, pic_name)
    
    else:
        Preprocessing.bg_pic=255*np.ones((Detecion_class.shape1, Detecion_class.shape2))
        
    if evaluate_all_files==True:
        file_end=len(fileNames)
    
    
    #%%   ###################################### Loop over each image   ##########################################################################
    
    for i_pic, fileName  in enumerate (fileNames[0:file_end]):
        
        if i_pic > int(n_save_img ):
            save_img=False
            
        t1 = cv2.getTickCount()
     
               
        ######################################   PREPROCESS ####################################################################################
        orimg=np.array(cv2.imread(fileName,0))
        img_n=cv2.normalize(orimg, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)    
        
        image_blur =Preprocessing.background_and_filter(img_n )
    
        image_bin , image_bin_fill = Preprocessing.preprocess_cv(image_blur ) 
               
        ######################################   DETECTION ####################################################################################
        
        droplets, n_droplets = Detecion_class.detect_droplets(image_bin_fill)   
        drops_pro_pic = Detecion_class.compute_droplets_properties(droplets)  
     
        
        ######################################  FIRST FILTER ####################################################################################
      
        drops_pro_pic_f,  image_bin_filtered =Detecion_class.filter_circles(img_n, image_bin_fill, fileName, drops_pro_pic , plot_circ)
        image_bin_filtered=image_bin_filtered.astype(np.uint8)
        area=sum( np.pi*(drops_pro_pic_f.diameters/2)**2)
        
        radius        =drops_pro_pic_f.diameters/2

        ######################################  SECOND FILTER ####################################################################################
        drops_pro_pic_c, contours , area, ratio =Detecion_class.contour_filter(fileName, img_n , image_bin_filtered, scale, plot_cont)
      
        image_with_drops=img_n.copy()
        image_with_drops = cv2.drawContours(image_with_drops, contours, -1,(255, 0, 0), thickness=-1)   
        
        
        radius_contour=drops_pro_pic_c.diameters/2
        
        ######################################  Final Result ####################################################################################
            
        DROPS.radien_in_mu_m(radius, radius_contour, area, ratio, scale)    
    
        if plot_all_steps :
            plot_func.plot_all_steps( i_pic,orimg ,image_blur, image_bin,image_bin_filtered , drops_pro_pic_f.centroids, radius, image_with_drops , save_img , evaluation_folder, pic_name) 
        if plotten_erg:
            plot_func.plot_final_result(i_pic, orimg, image_with_drops , save_img , evaluation_folder, pic_name)
        
        plt.close('all')   
        t2 = cv2.getTickCount()
        print("Time for image %i : %.3f s" %( i_pic+1, (t2-t1)/cv2.getTickFrequency() ) )   
    
        
    #%%    
    if postproc :
    
        plt.close('all')  
        name=pic_name+'contour_filter'
        title_his=r'Drop size over contour in  $\mu$m'
        plot_func.post_processing(evaluation_folder,drop_max, name, DROPS.r_g_c , i_pic+1, title_his)
        
            
        plt.close('all')  
        name=pic_name+'labeled_filter'
        title_his=r'Drop size in  $\mu$m'
        plot_func.post_processing(evaluation_folder,drop_max, name, DROPS.r_g , i_pic+1, title_his)
    
    

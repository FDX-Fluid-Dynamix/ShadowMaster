
"""
All Functions for the script Shadowgraphy_valuation.py

"""


import matplotlib.pyplot as plt
import numpy as np
import mahotas as mh
import mahotas.demos
from skimage import measure
import cv2

plt.close("all")

############################################################
#%%  Classes for results 
############################################################

class DROPS_class:
    def __init__(self ):
        self.r_g = []
        self.r_g_c=[]
        self.area_g_c =[]
        self.ratio_g_c=[]
               
    def radien_in_mu_m(self,radius, radius_contour, area, ratio, scale):   
        self.r_g       =np.append(self.r_g    ,scale*radius)
        self.r_g_c     =np.append(self.r_g_c  ,scale*radius_contour)
        self.area_g_c  =np.append(self.area_g_c ,area)
        self.ratio_g_c =np.append(self.ratio_g_c    ,ratio)
        
        
class Drops_pro_pic:
    def __init__(self,centroids,  diameters ):
        self.centroids=centroids
        self.diameters=diameters


############################################################
#%%  Preprocessing
############################################################


class PreProcess():
    
    """
    Preprocessing 
    
    Create Background image (function bg_erzeugen)
    
    Two Steps:
    
    1. Background Subtraction and blur Filter (function background_and_filter)
    2. Binary image and filled binary image (function preprocess_cv)

    """
    
    def __init__(self, tresh):
        self.bg_pic = []
        
        if tresh >255 or tresh<10:
            print(' Warning: \n The tresh value is not a in the correct range , default value=125 \n')
            self.tresh = int(tresh)
        else:
            self.tresh = int(tresh)

###########################    
    def bg_erzeugen(self, fileNames, size, folder_eval, pic_name):
        
        print('Start creating background image')
        t1 = cv2.getTickCount()
        
        for i in range(0,size) :
            
            im=cv2.imread(fileNames[i] ,0) .astype(int)
            
            # Invert image for better results
            im=cv2.normalize(im, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            im_inv=cv2.bitwise_not(im).astype(int)
          
            if i==0:
                bg_inv=im_inv/size
            else:
                bg_inv+=im_inv/size
            
        bg_pic=-1*(bg_inv-bg_inv.max())
        plt.imshow(bg_pic, cmap='gray')
        plt.axis('off')
        plt.savefig(folder_eval+pic_name+'_Background_image')
        plt.close('all')
        plt.close()
        t2 = cv2.getTickCount()
        print("Time for creating  Background image ", (t2-t1)/cv2.getTickFrequency() )  
        self.bg_pic=bg_pic
        
###########################        

    def background_and_filter(self, orimg ) :

        # Subtract Background image
        img_bg=np.abs(orimg-self.bg_pic)

        # Invert the picture to bright backgroung and black drops
        bck=np.uint8(np.abs(-1*(img_bg-img_bg.max())))

        # Remove local noise
        img_denois = cv2.fastNlMeansDenoising(bck,None,10,7,21) 

        # First Threshold 
        ret,img_trunc = cv2.threshold(img_denois, self.tresh ,255,cv2.THRESH_TRUNC)

        # Blur image
        blur = cv2.GaussianBlur(img_trunc,(5,5),0)  

        # Normailze image in numbers of 0-255
        blur= cv2.normalize(blur, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
  
        return blur
    
###########################        

    def preprocess_cv (self,blur ) :   
        
        # Second Threshold for blurred picture 
        ret, fbin=cv2.threshold(blur, self.tresh,255,cv2.THRESH_BINARY) 

        # Binary Picture with black background and white drops
        fbin=-1*(fbin.astype(int)-255)
            
        # To Secure that Seeding Point for fill the binary image is (0,0)==0 
        if fbin[0:10,0:10].mean()>=200 :
            fbin[0:100, 0:100]=0    
            if fbin[0:250,0:250].mean()>=100 :
                fbin[0:250,0:250]=0
        
        
        # Mask used to flood filling , the size needs to be 2 pixels more than the image
        h, w = fbin.shape[:2]
        mask = np.zeros((h+2, w+2), np.uint8)     
       
        # Floodfill from point (0, 0)
        im_floodfill = fbin.copy()
        print('Bildtype', type(im_floodfill))
        cv2.floodFill(im_floodfill, mask, (0,0), 255)  
    
        # Invert flood filled image
        im_floodfill_inv = cv2.bitwise_not(im_floodfill)    
    
        im_fill = fbin| im_floodfill_inv    # Combine the two images to get the foreground.
        im_fill=(im_fill+256)/255
        
        return fbin, im_fill
    


############################################################
#%%  Detection
############################################################

class DROP_DETECTION(DROPS_class, Drops_pro_pic):
      
    
    """
    Detection and filtering
    
    
    Two possibilities for the Calculation of the radius
    
    1. Binary image and mh.label (function detect_droplets and compute_droplets_properties)
        with filter  over the difference between the middle and the border of the drop (function filter_circles) 
    
    2. Filtered Binary image and cv2.findContours ( function contour_filter )
    

    """

    def __init__(self, scale,min_droplet_size, min_ratio, min_mv, min_dif ):
         
        if isinstance(min_droplet_size, int)==True:
            self.min_droplet_size = min_droplet_size
        else:
            print(' Warning: \n Minimal Droplet size is not a integer, default value=6 pixel \n')
            self.min_droplet_size = 6
        
        
        if type(scale)== int or type(scale)== float:
            self.scale=scale
        else:
            print(' Warning: \n Scale is not a number, default value=1 \xb5m per pixel \n')
            self.scale=1
        
        
        if type(min_ratio)== int or type(min_ratio)== float: 
            self.min_ratio=min_ratio
        else:
            print(' Warning: \n min_ratio is not a number, default value=0 \n')
            self.min_ratio=0
            
        if type(min_mv)== int or type(min_mv)== float:
            self.min_mv=min_mv
        else:
            print(' Warning: \n Minimal mean value is not a number, default value=0 \n')
            self.min_mv=0
        
        if type(min_dif)== int or type(min_dif)== float: 
            self.min_dif=min_dif
        else:
            print(' Warning: \n Minimal difference is not a number, default value=0 \n')
            self.min_dif=0
            
        self.shape1=0
        self.shape2=0
        
        print(' \n Scale=%.2f \xb5m per pixel \n Minimal droplet size= %.1f pixel \n Minimal ratio=%.1f \n Minimal mean value=%.1f \n Minimal difference=%.1f' %(self.scale, self.min_droplet_size, self.min_ratio, self.min_mv, self.min_dif ) )
 

###########################    
    def detect_droplets(self, image):                                                   

        # Detection of connected areas
        labeled_droplets, n_droplets = mh.label(image, np.ones((9,9), bool) )              

        # Remove too large and too small drops
        sizes = mh.labeled.labeled_size(labeled_droplets)                           
        labeled_droplets =  mh.labeled.remove_regions_where(labeled_droplets, sizes < self.min_droplet_size)
        labeled_droplets =  mh.labeled.remove_bordering(labeled_droplets)
        
        relabeled_droplets, n_left = mh.labeled.relabel(labeled_droplets)
        
        return relabeled_droplets, n_left

###########################        
    def compute_droplets_properties(self,droplets):

        # Calculate centroids and diameters 
        properties   = measure.regionprops(droplets)
        
        centroids    = np.asarray([prop.centroid for prop in properties])                       
        diameters    = np.asarray([prop.equivalent_diameter for prop in properties])
        
        drop_pic=Drops_pro_pic(centroids, diameters)
        
        return drop_pic
   
###########################                  
    def filter_circles(self, orimg, fill,  drop_pic,  plot_circ)   : 
        
        '''
        Filter over the difference between the middle and the border  of the drop 
        '''
		
        diameters=drop_pic.diameters
        f_rand=np.zeros(len(drop_pic.centroids))
        f_diff=np.zeros(4)
        fill_new=fill.copy()
        
        Filter_sharp=[]
        
 
        for i_drop in range(0,len(drop_pic.centroids)):
            
            center1=int(round(drop_pic.centroids[i_drop,0]))
            center2=int(round(drop_pic.centroids[i_drop,1]))
               
            d_mikrometer=diameters[i_drop]*self.scale
                
            # Determine the outer radius for filtering
            rad_au =int(round((diameters[i_drop]/2)+2 ))
            
            if  d_mikrometer >720:
                rad_au =rad_au+8
            elif  d_mikrometer > 450:
                rad_au =rad_au+7  
            elif  d_mikrometer > 230 :
                rad_au =rad_au+4
            elif  d_mikrometer > 68:
                rad_au =rad_au+3
            elif  d_mikrometer > 40:
                rad_au =rad_au+2
            elif  d_mikrometer < 40:
                rad_au =rad_au-3
 
            #Removal of the drops at the edge of the binary image
            edege_tol=2
            if center1+rad_au >(self.shape1-edege_tol) or center2+rad_au> (self.shape2-edege_tol) or    center1-rad_au < edege_tol or center2-rad_au < edege_tol :
                sharp=False
                Filter_sharp.append(sharp)
                if center1+rad_au >(self.shape1-edege_tol):
                    fill_new[center1-rad_au:   self.shape1  ,center2 -rad_au:center2 +rad_au]=0
                if center2+rad_au> (self.shape2-edege_tol) :
                    fill_new[center1 -rad_au:center1 +rad_au , center2 -rad_au:self.shape2         ]=0
                if center1-rad_au < edege_tol:
                    fill_new[0:center1 +rad_au               , center2 -rad_au:center2+rad_au]=0
                if center2-rad_au < edege_tol   :
                    fill_new[center1 -rad_au:center1 +rad_au , 0:center2 +rad_au ]=0
                continue 
                
    
            #Darken of the bright middle of the drop 
            fcent= orimg[ center1 ,   center2  ].astype(int)
            if fcent >= 25 :          
                fcent=10
                
                
            #Compute the difference between the center and the edge of the drop
            f_diff[0]=np.abs(orimg[ center1 +rad_au, center2 ].astype(int)-fcent)
            f_diff[1]=np.abs(orimg[ center1 -rad_au, center2 ].astype(int)-fcent)
            f_diff[2]=np.abs(orimg[ center1, center2+rad_au ].astype(int)-fcent)
            f_diff[3]=np.abs(orimg[ center1, center2-rad_au ].astype(int)-fcent)
            
            f_diff_nz=f_diff[f_diff>10]
            if len(f_diff_nz)>=1:
                f_rand[i_drop]=f_diff_nz.mean()
            else:
                f_rand[i_drop]=f_diff.mean()
                            
        
            # Filtering
            sharp=False
            
            
            # Filter out drops with too little difference
            '''
            For better results , it is also possible to set the minimum difference as a function of the diameter
            '''
            if f_rand[i_drop] >self.min_dif:
                sharp=True 
            

            # Filter out drops with too little diameters
            if diameters[i_drop]<self.min_droplet_size:
                sharp=False
                        
                        
            #  Remove blurred drops of the binary image       
            if sharp==False:
                fill_new[center1 -rad_au:center1 +rad_au , center2 -rad_au:center2 +rad_au]=0
                
            Filter_sharp.append(sharp)     
                     
            
            # Plot each drop to check the filter
            if plot_circ  :
                drop=orimg[center1-rad_au:center1+rad_au  ,  center2-rad_au:center2+rad_au ]
                if center1+rad_au >(self.shape1-edege_tol) or center2+rad_au >(self.shape2-edege_tol)  or center1-rad_au < edege_tol or center2-rad_au < edege_tol :
                    continue
                
                plt.imshow(drop, cmap='gray')
                plt.axis('off')
                
                if sharp==True:
                    plt.title( 'Sharp drop: DIF_mean: %.4f, D= %.3f $\mu m$' %(f_diff_nz.mean() , d_mikrometer) )
                else:
                    plt.title( 'Blurred drop: DIF_mean: %.4f, D= %.3f $\mu m$' %(f_diff_nz.mean() , d_mikrometer) )
                
                plt.show()
                plt.close()

      
               
        validDiameters = diameters[Filter_sharp]
        validCentroids= drop_pic.centroids[Filter_sharp]
        drops_pro_pic_f=Drops_pro_pic(validCentroids, validDiameters)

        fill_new=fill_new.astype(np.uint8)
        
        return  drops_pro_pic_f,fill_new    
    
   
###########################        
    def contour_filter(self, orimg , fbin, scale, plot_cont) :
        
        '''
        Compute Diameter with cv2.findContours and filter out drops with too small ratio or too bright drops
        '''
        
        contours, hierarchy = cv2.findContours(fbin,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)  
        
        centerx,centery    =np.zeros( len(contours) ),  np.zeros( len(contours) )
        diameter , mean_val=np.zeros( len(contours) ),  np.zeros( len(contours) )
        area   ,ratio      =np.zeros( len(contours) ),  np.zeros( len(contours) )
        extent             =np.zeros( len(contours) )
        Filter_sharp ,contours_sharp      =[], []
    
        edege_tol=3
            
    #-------------------------------------- Calculation -------------------------------------- #
        for i, cnt in enumerate (contours ) :    
            sharp=True
            M = cv2.moments(cnt)
            
            if M['m00']==0:
                sharp=False
                Filter_sharp.append(sharp)
                continue
            
            cy = int(M['m10']/M['m00'])
            cx = int(M['m01']/M['m00'])
            
            centerx[i] =cx
            centery[i] =cy
            area[i]    =cv2.contourArea(cnt)
            diameter[i]=np.sqrt(4*area[i]/np.pi)
            
            x,y,w,h = cv2.boundingRect(cnt)
            ratio[i] = float( min(w,h)/max(w,h) )
            

            rect_area = w*h
            extent[i] = float(area[i])/rect_area
            
            cx_max=max( cnt[:,0,1])
            cx_min=min( cnt[:,0,1])
            cy_max=max( cnt[:,0,0])
            cy_min=min( cnt[:,0,0])
            
            drop= orimg[ cx_min:cx_max , cy_min:cy_max]
            
            
            # Filter out too small drops
            if drop.shape[0]<=self.min_droplet_size or drop.shape[1]<=self.min_droplet_size:
                sharp=False
                Filter_sharp.append(sharp)
                continue
            
            
            # Filter out drops at the edge of the picture
            if cx_max >(self.shape1-edege_tol) or cy_max > (self.shape2-edege_tol)  or cx_min < edege_tol or cy_min < edege_tol  :
                sharp=False
                Filter_sharp.append(sharp)
                continue
            
            else:    
                mask= fbin[ cx_min:cx_max , cy_min:cy_max]
                mask=np.uint8(mask)
                
                mean = cv2.mean(drop,mask = mask)
                mean_val[i]=mean[0]
    
                # Filter out too small ratios
                if ratio[i] < self.min_ratio :
                    sharp= False
    
                # Filter out too bright drops
                if mean_val[i] > self.min_mv:
                    sharp=False
                
                Filter_sharp.append(sharp)
                if sharp==True:
                    contours_sharp.append(cnt)

            if plot_cont : 
                    plt.imshow(drop, cmap='gray')
                    plt.axis('equal')
                    plt.axis( 'off')
                    if sharp==True:           
                        plt.title( 'Sharp drop: MW %.4f , D=%.4f $\mu m $, ratio=%.4f: ' %( mean_val[i], self.scale*diameter[i], ratio[i]  ) )
                    else:
                        plt.title( 'Blurred drop: MW %.4f , D=%.4f $\mu m $, ratio=%.4f: ' %( mean_val[i], self.scale*diameter[i], ratio[i]  ) )
                    plt.show()
                    plt.close()
                
        valid_ratio   =ratio   [Filter_sharp]        
        valid_diameter=diameter[Filter_sharp]
        valid_area    =area    [Filter_sharp]
        centerx       =centerx [Filter_sharp]
        centery       =centery [Filter_sharp]
        
        center=np.vstack((centerx,centery)).T
        
        drops_pro_pic_c=Drops_pro_pic(center, valid_diameter)
                
        return drops_pro_pic_c , contours_sharp , valid_area , valid_ratio
   
   
############################################################
#%%  Different plots
############################################################
class PLOT_Shadowgraphy:
    
    '''
        Different variants for plotting the results
    '''
    
    def plot_final_result(self,j,orimg, img_c ,  save_img , folder_eval, pic_name) :
        
        plt.imshow(orimg, cmap='gray') 
        plt.title( 'Original image')
        plt.axis('off')
        plt.show()

		
        plt.imshow(img_c, cmap='gray')
        plt.title( 'Result')
        plt.axis('off')
        
        if save_img :
            filename=folder_eval+pic_name+'_result_contour'+str(j+1)+ '.png'
            plt.savefig(filename, bbox_inches='tight' , dpi=300)
            
        plt.show()
        plt.close('all')
        
   
###########################    
    def plot_all_steps(self,j,  orimg , blur, i_bin , i_bin_f, validCentroids, radius, img_mit_contour , save_img , folder_eval ,pic_name) :

        plt.close()
        plt.figure(j)
        
        fig,ax=plt.subplots(1, 5)
        
        ax[0].imshow(orimg, cmap='gray')
        ax[0].set_title( 'Original image', fontsize=8)
        ax[0].axis('off')
        
        ax[1].imshow(blur, cmap='gray')
        ax[1].set_title( 'Preprocessed image', fontsize=8)
        ax[1].axis('off')
        
        ax[2].imshow(i_bin, cmap='gray')
        ax[2].set_title( 'Binary image', fontsize=8)
        ax[2].axis('off')
        
        ax[3].imshow(i_bin_f, cmap='gray')
        ax[3].set_title( 'Filtered Binary image', fontsize=8)
        ax[3].axis('off')
        
        
        if not len(validCentroids)==0 :
            y_i=validCentroids[:,0]
            x_i=validCentroids[:,1]
  
            for i in range(0, len(radius)):
                c = plt.Circle((x_i[i], y_i[i]), radius[i], color='green', linewidth=1, fill=False)
                ax[4].add_artist(c)
        
        ax[4].imshow(img_mit_contour, cmap='gray')
        ax[4].set_title( 'Final Result', fontsize=8)
        ax[4].axis('off')
        
        
        if save_img :
            filename=folder_eval+pic_name+'_Filter_bis'+str(j+1)+ '.png'
            plt.savefig(filename, bbox_inches='tight', dpi=300)
            
        plt.show()
        plt.close('all')
        return
            
###########################    
    def post_processing(self, folder_eval, drop_max, pic_name, radius_in_mum , j, title_his) :
        
        plt.close()
        
        Dsize =2*radius_in_mum   
        
        name=folder_eval+'diameter_'+ pic_name+'until_image_'+str(j+1)+'.txt'
        np.savetxt(name, Dsize)
        
      
        Dcount = len(Dsize)
        SMD=sum(Dsize**3) /(sum(Dsize**2 ) )


        plt.hist(Dsize,  bins=120, density=True, edgecolor="k",color="b")
        plt.title(title_his)
        plt.xlabel(r"Value in $\mu$m")
        plt.ylabel("Probability")
        plt.xlim(0,drop_max)
        plt.text( plt.xlim()[1]*0.55, plt.ylim()[1]*0.85  , 'Mean Value  %.2f $\mu$m' %(Dsize.mean()), fontsize=12)
        plt.text( plt.xlim()[1]*0.55, plt.ylim()[1]*0.75 ,  'SMD   %.2f: '    %SMD + ' $\mu$m', fontsize=12)  
        plt.text( plt.xlim()[1]*0.55, plt.ylim()[1]*0.65  , 'Count:  %.2f' %Dcount, fontsize=12)
        plt.text( plt.xlim()[1]*0.55, plt.ylim()[1]*0.55  , 'D_min:  %.3f und D_max %.2f ' %(Dsize.min(),Dsize.max()), fontsize=12)
        filename=folder_eval+'Histogramm_'+ pic_name+str(j)+ '.png'
        plt.savefig(filename, bbox_inches='tight', dpi=300)
        plt.show()
        plt.close()
            
#%%


           

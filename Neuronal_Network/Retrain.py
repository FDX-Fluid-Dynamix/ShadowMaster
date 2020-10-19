
"""
Python skript for retrain the neuronal network

The evaluation of the training with tensorboard is done by the following command in the console: 
tensorboard --logdir=\samples\droplets\logs\ --host localhost --port 8088

"""

#%%#########################################################
#  Packages
############################################################

import os
import sys
import tensorflow as tf


# Import Mask RCNN
ROOT_DIR = os.path.abspath("")
sys.path.append(ROOT_DIR)  # To find local version of the library
import mrcnn.model as modellib
import drops


#%%#########################################################
# Settings
############################################################

weight_start="mask_rcnn_drops.h5"
weight_new  ="mask_rcnn_drops_new.h5"


DROP_DIR = os.path.join(ROOT_DIR, "datasets_new/droplets/")


# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
weights_path=os.path.join(MODEL_DIR, weight_start)

config = drops.DropConfig()


class InferenceConfig(config.__class__):
    # Run detection on one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    STEPS_PER_EPOCH =200         #DEAFULT=1000
    EPOCHS=30                    #DEAFULT=30
    LEARNING_RATE= 0.001         #DEAFULT=0.001    
     
  
config = InferenceConfig()
config.display()

#%%#########################################################
#  Build network
############################################################

DEVICE = "/gpu:0"  

with tf.device(DEVICE):

    model = modellib.MaskRCNN(mode="training", model_dir=MODEL_DIR, config=config)     
    model.load_weights(weights_path, by_name=True)

    # Train dataset
    dataset_train = drops.DropDataset()
    dataset_train.load_drop(DROP_DIR, "train")
    dataset_train.prepare()
    # Validation dataset
    dataset_val = drops.DropDataset()
    dataset_val.load_drop(DROP_DIR, "val")
    dataset_val.prepare()

    print("Training network heads")
    model.train(dataset_train, dataset_val, learning_rate=config.LEARNING_RATE, epochs=config.EPOCHS, layers='heads')
 
    
    
#%%#########################################################
#  Save weights
############################################################

model_path = os.path.join(MODEL_DIR, weight_new)
model.keras_model.save_weights(model_path)


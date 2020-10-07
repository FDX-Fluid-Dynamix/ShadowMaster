"""
Mask R-CNN Subcalss
"""

import os
import sys
import json
import numpy as np
import skimage.draw

# Root directory of the project
ROOT_DIR = os.path.abspath(" ")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_drops_15.11.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")


#%%

############################################################
#  Configurations
############################################################

class DropConfig(Config):
    
    # Give the configuration a recognizable name
    NAME = "drop"

    # If GPU is used
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  

############################################################
#  Dataset
############################################################

class DropDataset(utils.Dataset):

    def load_drop(self, dataset_dir, subset):

        self.add_class("drop", 1, "drop")

        # Train or validation dataset?
        assert subset in ["train", "val"]
        dataset_dir = os.path.join(dataset_dir, subset)

        # Load annotations
        annotations = json.load(open(os.path.join(dataset_dir, 'via_region_data.json')))
        annotations = list(annotations.values())  # don't need the dict keys
        
        annotations = [a for a in annotations if a['regions']]# The VIA tool saves images in the JSON even if they don't have any annotations. Skip unannotated images.

        # Add images
        for a in annotations:
            # Get the x, y coordinaets of points of the polygons that make up the outline of each object instance. These are stores in the
            # shape_attributes (see json format above) .
            if type(a['regions']) is dict:
                polygons = [r['shape_attributes'] for r in a['regions'].values()]
            else:
                polygons = [r['shape_attributes'] for r in a['regions']] 

            # load_mask() needs the image size to convert polygons to masks. Unfortunately, VIA doesn't include it in JSON, so we must read the image. This is only managable since the dataset is tiny.
            image_path = os.path.join(dataset_dir, a['filename'])
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]

            self.add_image( "drop", image_id=a['filename'],  # use file name as a unique image id
                path=image_path,
                width=width, height=height,
                polygons=polygons)

    def load_mask(self, image_id):

        image_info = self.image_info[image_id]
        if image_info["source"] != "drop":
            return super(self.__class__, self).load_mask(image_id)

        # Convert polygons to a bitmap mask of shape  [height, width, instance_count]
        info = self.image_info[image_id]
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])], dtype=np.uint8)
        for i, p in enumerate(info["polygons"]):
            # Get indexes of pixels inside the polygon and set them to 1
            rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
            mask[rr, cc, i] = 1

        # Return mask, and array of class IDs of each instance. Since we have  one class ID only, we return an array of 1s
        return mask.astype(np.bool), np.ones([mask.shape[-1]], dtype=np.int32)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "droplets":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)


def train(model, data_dir , config):
    # Training dataset.
    dataset_train = DropDataset()
    dataset_train.load_drop(data_dir, "train")
    dataset_train.prepare()
    # Validation dataset
    dataset_val = DropDataset()
    dataset_val.load_drop(data_dir, "val")
    dataset_val.prepare()

    print("Training network heads")
    model.train(dataset_train, dataset_val, learning_rate=config.LEARNING_RATE, epochs=config.EPOCHS, layers='heads')

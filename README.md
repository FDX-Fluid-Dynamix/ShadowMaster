# ShadowMaster

In this GitHub Repository two different variants are presented for the evaluation of droplet sizes from a Shadowgraphy measurement.

* [Shadowgraphy](#shadowgraphy)
* [Python Algorithm](#python-algorithm)
* [Neuronal Network](#neuronal-network)

## Shadowgraphy

	
## Python Algorithm

This version to evaluate the Shadowgraphy images is inspired by a Binary thresholding function. The python package
[@OpenCV ](https://opencv-python-tutroals.readthedocs.io/en/latest/index.html) is used. \
If necessary it is possible to generate a background image as the mean value of a specific number of images.
Background subtraction reduces the influence of scratches and dirt on the lens. \

For better results the [@cv2.fastNlMeansDenoising](https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_photo/py_non_local_means/py_non_local_means.html) filter is used to filter out local noise. \
Afterwards with a thresholding function with a fix thresh value a binary image is made.

The [@„mh.labeled“](https://mahotas.readthedocs.io/en/latest/labeled.html) function is used to detect possible drops in the form of connected areas.
This drops are not always real drops, thus it is necessary to filter out bad possible drops.
It is easy to filter out drops that are too small or too large. 
Even drops that are too strongly deformed can be easily filtered out via a maximum allowed aspect ratio of the expansion in x and y direction. \
The distinction between sharp and unsharp drops is very difficult because there is no universal filter to filter out sharp objects. \
For this problem, a filter inspired by human vision was developed. 
For humans, an object appears sharp if there is a large color difference between the average value and the close environment.
To simplify, the filter uses the difference between the center (usually the darkest point) and a radius increased by a few pixels. 

To get more precise information about the detected drops, for example the area of the drops, the contours in the filtered binary image are determined.
The diameter of the detected drops is calculated as the equivalent circle diameter of the determined area.


## Neuronal Network

The structure of the network is the [@Mask R-CNN](https://arxiv.org/abs/1703.06870) network from 2017 by Kaiming He.
The implementation of the network was done by [@matterport](https://github.com/matterport/Mask_RCNN), where a good introduction to the structure of the neural network can be found. 

The Python script drops.py contains the necessary changes to the configuration of the neural network. 
Furthermore there are three other scripts. 

The first (Master_drop_detection.py) allows the evaluation of Shadowgraphy images. The neural network creates a bounding box and a mask for each possible drop. Therefore the diameter can be determined in two ways. Once as the average of the two sides of the bounding box or using the mask and the equivalent circle diameter .

The second (Generate_new_training_dataset.py) allows the automatic generation of new training data with the result of the previous weight. 
These labelled images can be checked and improved by hand using the [@VGG Image Anonator](http://www.robots.ox.ac.uk/~vgg/software/via/).

The third skript ( Retrain.py) makes it possible to train the weights of the neuronal network with the new dataset.

### Setup conda enviroment

The specific python packages For the neuronal network are listed in the files "Requirements.txt" for cpu and for gpu "Requirements_gpu.txt" . 
The following commands are necessary to create the conda Enviroment: 

conda create -n Env_Name \
conda activate Env_Name \
conda install --file Requirements.txt \
conda install -c conda-forge imgaug 

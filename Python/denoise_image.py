import numpy as np 
import cv2 
from matplotlib import pyplot as plt 
import imageio

filepath = "data/ILSVRC2012_img_val/ILSVRC2012_val_00000001.JPEG"
# Reading image from folder where it is stored 
img = cv2.imread(filepath) 
#img = imageio.imread('data/ILSVRC2012_img_val/ILSVRC2012_val_00000001.JPEG')

# denoising of image saving it into dst image 
dst = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 15) 
  
# Plotting of source and destination image 
plt.subplot(121), plt.imshow(img) 
plt.subplot(122), plt.imshow(dst) 
  
plt.show()

def denoiseColor(filepath):
    img = cv2.imread(filepath) 
    dst = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 15) 
    return dst



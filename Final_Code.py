#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from tkinter import filedialog
from tkinter import *

root = Tk()
root.filename =  filedialog.askopenfilename(initialdir = "/",title = "Select file",filetypes = (("png files","*.png"),("all files","*.*")))
filepath =root.filename
print (root.filename)
img= cv2.imread(filepath,3)

#-----------------------Data Conversion------------------------------------------------
 #importing image class from PIL package
from PIL import Image
import numpy as np
#import numpy.ndarray
from scipy import io

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
#------------------to show the image
plt.imshow(mpimg.imread(filepath,3))
plt.title("Input Image"), plt.xticks([]), plt.yticks([])
plt.show()
data=mpimg.imread(filepath,3)
from numpy import asarray
# asarray() class is used to convert
# PIL images into NumPy arrays
data_1 =np.frombuffer(data,dtype="float",count=-1)
arr=np.array(data_1)
io.savemat('arr.mat',{"vec":arr})
mydata = io.loadmat('arr.mat')
print(mydata)
xmed=mydata['vec']
#--------------------------pre-processing using sample input images --------------
import cv2
#img= cv2.imread(filepath,3)
blur = cv2.GaussianBlur(img, (15,15), 0)
cv2.imwrite('Filtered_Image.png', blur);
img2= cv2.imread("image_09.png")
plt.imshow(img2)
plt.title(" Gaussian Filtered Image"), plt.xticks([]), plt.yticks([])
plt.tight_layout()
plt.show()
#------------------------edge detection--------------------------------------------
# Detecting Edges on the Image using the argument ImageFilter.FIND_EDGES
#--------------------Canny edge detection --------------
# Canny Edge Detection
edge = cv2.Canny(img, 20, 30)
fig, ax = plt.subplots(1, 2, figsize=(18, 6), dpi=150)
ax[0].imshow(img, cmap='gray')
ax[1].imshow(edge, cmap='gray')
plt.title("Canny Edge  Filtered Image"), plt.xticks([]), plt.yticks([])
plt.tight_layout()
plt.show()
#--------------------------------------------------------------------------------

import numpy as np
import cv2
from matplotlib import pyplot as plt
img= cv2.imread(filepath,3)
b,g,r = cv2.split(img)
rgb_img = cv2.merge([r,g,b])
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
b,g,r = cv2.split(img)
rgb_img = cv2.merge([r,g,b])
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
# noise removal
kernel = np.ones((2,2),np.uint8)
#opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)
closing = cv2.morphologyEx(thresh,cv2.MORPH_CLOSE,kernel, iterations = 2)
# sure background area
sure_bg = cv2.dilate(closing,kernel,iterations=3)
# Finding sure foreground area
dist_transform = cv2.distanceTransform(sure_bg,cv2.DIST_L2,3)
# Threshold
ret, sure_fg = cv2.threshold(dist_transform,0.1*dist_transform.max(),255,0)
# Finding unknown region
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg,sure_fg)
# Marker labelling
ret, markers = cv2.connectedComponents(sure_fg)
# Add one to all labels so that sure background is not 0, but 1
markers = markers+1
# Now, mark the region of unknown with zero
markers[unknown==255] = 0
markers = cv2.watershed(img,markers)
img[markers == -1] = [255,0,0]
plt.subplot(211),plt.imshow(rgb_img)
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(212),plt.imshow(thresh, 'gray')
plt.imsave(r'thres_hold.png',thresh)
plt.title("Otsu's binary threshold"), plt.xticks([]), plt.yticks([])
plt.tight_layout()
plt.show()
plt.subplot(211),plt.imshow(closing,'gray')
plt.title("morphologyEx:Closing:2x2"), plt.xticks([]), plt.yticks([])
plt.subplot(212),plt.imshow(sure_bg,'gray')
plt.imsave(r'dil_a.tion.png',sure_bg)
plt.title("Dilation"), plt.xticks([]), plt.yticks([])
plt.tight_layout()
plt.show()
plt.subplot(211),plt.imshow(dist_transform,'gray')
plt.title("Distance Transform"), plt.xticks([]), plt.yticks([])
plt.subplot(212),plt.imshow(sure_fg, 'gray')
plt.title("Thresholding"), plt.xticks([]), plt.yticks([])
plt.tight_layout()
plt.show()
plt.subplot(211),plt.imshow(unknown,'gray')
plt.title("Unknown"), plt.xticks([]), plt.yticks([])
plt.subplot(212),plt.imshow(img, 'gray')
plt.title("Result from Watershed"), plt.xticks([]), plt.yticks([])
plt.tight_layout()
plt.show()
#-------------------------histogram using calculation
# find frequency of pixels in range 0-255 
import numpy as np
import cv2
from matplotlib import pyplot as plt
img= cv2.imread(filepath,3)
histr = cv2.calcHist([img],[0],None,[256],[0,256]) 
# show the plotting graph of an image 
plt.plot(histr)
plt.title("Histogram of image"), plt.xticks([]), plt.yticks([])
plt.tight_layout()
plt.show() 
# alternative way to find histogram of an image 
plt.hist(img.ravel(),256,[0,256])
plt.title("GENERATING HISTOGRAM OF ORIENTED GRADIENTS"), plt.xticks([]), plt.yticks([])
plt.show()
#-----------------------accuracy sensitivity -------------
#Import the necessary libraries

from tamil import utf8
#string = u"எஃகு"
string =u"அ ஆ இ ஈ உ ஊ எ ஏ ஐ ஒ ஓ ஔ க் க	கா  கி	கீ  கு	கூ  கெ	கே கை	கொ  கோ கௌ"
letters = utf8.get_letters(string)
print(len(letters))
# 3. Not 4. 
print(letters)
 #[u'\u0b8e', u'\u0b83', u'\u0b95\u0bc1']
for letter in letters:
    print(letter)


# In[2]:





# In[1]:


pip show tamil  # Check the installed version
pip install --upgrade tamil  # Upgrade the tamil library to the latest version


# In[2]:


pip show tamil


# In[3]:


pip install --upgrade tamil


# In[1]:


import scipy.io
scipy.io.loadmat("arr.mat")


# In[ ]:


get_ipython().system('')


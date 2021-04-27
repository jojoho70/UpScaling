import cv2
import numpy as np
import glob 
import os 
from PIL import Image, ImageEnhance,ImageFilter
import operator
from imutils import paths
import argparse
import imageio
import math
import mahotas as mh
import colorcorrect
import colorcorrect.algorithm as cca
from colorcorrect.util import from_pil, to_pil

# totallength=2048
# totalwidth=1668
# length=0
# width=0
# def ConvertpiltoOpencv(image):
#     firstimage=image.convert('RGB')
#     open_cv_image = np.array(firstimage)
#     open_cv_image = open_cv_image[:, :, ::-1].copy() 
#     return open_cv_image
# xn=[]

# path = glob.glob("*png")
# for x in path:
#     length+=2048
#     width+=1558
#     man = cv2.imread(x)
#     xn.append(man)
def denoiserv3(image):

    sharpen_kernel = np.array([[-.5,-.5,-.5], [-.5,5,-.5], [-.5,-.5,-.5]])
    sharpen = cv2.filter2D(image, -1, sharpen_kernel)
    # converted_img = cv2.cvtColor(sharpen , cv2.COLOR_GRAY2BGR)
    denoise = cv2.fastNlMeansDenoising(sharpen, None, 20, 20)

    return denoise
def ConvertOpencvToPill(image):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    return  img

def variance_of_laplacian(image):
	# compute the Laplacian of the image and then return the focus
	# measure, which is simply the variance of the Laplacian
	return cv2.Laplacian(image, cv2.CV_64F).var()

def ConvertpiltoOpencv(image):
    firstimage=image.convert('RGB')
    open_cv_image = np.array(firstimage)
    open_cv_image = open_cv_image[:, :, ::-1].copy() 
    return open_cv_image
def denoiser(image):
    denoise = cv2.fastNlMeansDenoising(image, None, 15, 15)
    return denoise

def antialiasing(image):
    sharpen_kernel = np.array([[-1.1,-1.1,-1.1], [-1.1,10,-1.1], [-1.1,-1.1,-1.1]])
    sharpen = cv2.filter2D(image, -1, sharpen_kernel)
    return sharpen
def denoiserz(image):
    denoise = cv2.fastNlMeansDenoising(image, None, 50, 50)
    return denoise

exposure_times = np.array([15.0, 2.5, 0.25, 0.0333], dtype=np.float32)

img1 = Image.open("owo.png")

#img2 = Image.open("68.png")

# suppose img2 is to be shifted by `shift` amount 
#shift = (0, 0)

# compute the size of the panorama
#nw, nh = map(max, map(operator.add, img2.size, shift), img1.size)
#offsetx= round(nw-(nw*.1))
#print(nh)
#print(nw)
# paste img1 on top of img2
#newimg1 = Image.new('RGBA', size=(16000, 10000), color=(0, 0, 0, 0))
#newimg1.paste(img2, (offsetx, 0))
#newimg1.paste(img1, (0, 0))
#
## paste img2 on top of img1
#newimg2 = Image.new('RGBA', size=(16000, 10000), color=(0, 0, 0, 0))
#newimg2.paste(img1, (0, 0))
#newimg2.paste(img2, (offsetx, 0))

# blend with alpha=0.5
#result = Image.blend(newimg1, newimg2, alpha=0.5)
# img1= to_pil(cca.automatic_color_equalization(from_pil(img1), slope=5, limit=500, samples=250))
x= ConvertpiltoOpencv(img1)
x=cv2.cvtColor(x,cv2.COLOR_BGR2GRAY)
x = cv2.equalizeHist(x)
#

x= denoiserv3(x)
# x=antialiasing(x)
# x=ConvertOpencvToPill(x)

# x=Image.fromarray(x)
# # x = cca.automatic_color_equalization(from_pil(x),2.5)

# x=to_pil(x)
# x=x.filter(ImageFilter.SHARPEN)
# x=x.filter(ImageFilter.SMOOTH_MORE)

# x=ConvertpiltoOpencv(x)
#x=cv2.cvtColor(x,cv2.COLOR_BGR2GRAY)
# x=denoiserz(x)

cv2.imwrite("3233234.png",x)

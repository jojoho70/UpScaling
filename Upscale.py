import cv2
import numpy as np
import glob 
import os 
from PIL import Image, ImageEnhance,ImageFilter
import operator
from imutils import paths
import argparse
import imageio
import mahotas as mh
import colorcorrect
import colorcorrect.algorithm as cca
from colorcorrect.util import from_pil, to_pil
path="/Users/joseph/Desktop/Stitch/Newdata/"


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
def firstdenoiser(image):

    sharpen_kernel = np.array([[-.4,-.4,-.4], [-.4,5,-.4], [-.4,-.4,-.4]])
    sharpen = cv2.filter2D(image, -1, sharpen_kernel)
    # converted_img = cv2.cvtColor(sharpen , cv2.COLOR_GRAY2BGR)
    denoise = cv2.fastNlMeansDenoising(sharpen, None, 20, 20)
    cv2.imshow("denoise",denoise)
    return denoise
def antialiasing(image):
    sharpen_kernel = np.array([[-1.13,-1.13,-1.13], [-1.13,10,-1.13], [-1.13,-1.13,-1.13]])
    sharpen = cv2.filter2D(image, -1, sharpen_kernel)
    return sharpen
def denoiserz(image):
    denoise = cv2.fastNlMeansDenoising(image, None, 10, 10)
    return denoise

files = glob.glob("*png")
startnum=1
for i in files:
    print("Current file is: "+i)
    print(files)
    result=Image.open(i)

    x= ConvertpiltoOpencv(result)
    x=firstdenoiser(x)
    x=ConvertOpencvToPill(x)
    cv2.imwrite(str(startnum)+".png",x)
    print(i+" was sucessefully upscaled!")
    startnum+=1
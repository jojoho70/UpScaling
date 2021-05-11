import cv2
import numpy as np
import glob
import os
from PIL import Image, ImageEnhance, ImageFilter
import operator
from imutils import paths
import argparse
import imageio
import mahotas as mh
import colorcorrect
import colorcorrect.algorithm as cca
from colorcorrect.util import from_pil, to_pil

# set your path.
path = "/Users/joseph/Desktop/Stitch/Newdata/"


def ConvertOpencvToPill(image):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return img


def variance_of_laplacian(image):
    # compute the Laplacian of the image and then return the focus
    # measure, which is simply the variance of the Laplacian
    # this will be used in a future version
    return cv2.Laplacian(image, cv2.CV_64F).var()


def pilToOpencv(image):
    firstimage = image.convert('RGB')
    open_cv_image = np.array(firstimage)
    open_cv_image = open_cv_image[:, :, ::-1].copy()
    return open_cv_image


def denoiser(image):

    sharpen_kernel = np.array(
        [[-.4, -.4, -.4], [-.4, 5, -.4], [-.4, -.4, -.4]])
    sharpen = cv2.filter2D(image, -1, sharpen_kernel)
    sharpen = antialiasing(sharpen)
    # converted_img = cv2.cvtColor(sharpen , cv2.COLOR_GRAY2BGR)
    denoise = cv2.fastNlMeansDenoising(sharpen, None, 20, 20)
    cv2.imshow("denoise", denoise)
    return denoise


def antialiasing(image):
    sharpen_kernel = np.array(
        [[-1.13, -1.13, -1.13], [-1.13, 10, -1.13], [-1.13, -1.13, -1.13]])
    sharpen = cv2.filter2D(image, -1, sharpen_kernel)
    return sharpen


files = glob.glob("*png")
startNum = 1
for i in files:
    print("Current file is: "+i)
    result = Image.open(i)
    img = pilToOpencv(result)
    img = denoiser(img)
    img = ConvertOpencvToPill(img)
    cv2.imwrite(str(startNum)+".png", img)
    print(i+" was sucessefully upscaled!")
    startNum += 1

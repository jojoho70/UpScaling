import cv2
import numpy as np
import glob
import os
from PIL import Image
import operator
from imutils import paths
import argparse
import imageio
import math
length = 1768
width = 2048


def variance_of_laplacian(image):
    # compute the Laplacian of the image and then return the focus
    # measure, which is simply the variance of the Laplacian
    return cv2.Laplacian(image, cv2.CV_64F).var()


def ConvertpiltoOpencv(image):
    firstimage = image.convert('RGB')
    open_cv_image = np.array(firstimage)
    open_cv_image = open_cv_image[:, :, ::-1].copy()
    return open_cv_image


def denoiser(image):

    sharpen_kernel = np.array(
        [[-.4, -.4, -.4], [-.4, 5, -.4], [-.4, -.4, -.4]])
    sharpen = cv2.filter2D(image, -1, sharpen_kernel)
    # converted_img = cv2.cvtColor(sharpen , cv2.COLOR_GRAY2BGR)
    denoise = cv2.fastNlMeansDenoising(sharpen, None, 20, 20)
    cv2.imshow("denoise", denoise)
    return denoise


def ConvertOpencvToPill(image):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return img


path = glob.glob("*png")
cv_img = []
for img in path:
    l = cv2.imread(img)
    # maner = l.size
    length += 2048
    width += 1768
    print(length)
    print(width)
    n = cv2.imread(img)
    cv_img.append(n)
path = np.sort(path)
rowvalues = round(math.sqrt(len(cv_img)))
print(path)
# only overlay on the x on beginning layer
# then x and y above next layer

# calc all the sizes of image or dataset
# suppose img2 is to be shifted by `shift` amount
overlay = 0.1
offsetx = width-round(width*overlay)
offsety = length-round((length*2)*length)
shift = (0, 0)
y = 0
x = 0
# compute the size of the panorama
shift = (offsetx, 0)
# paste img1 on top of img2
finaleimage = Image.new('RGBA', size=(length, width), color=(0, 0, 0, 0))
for j in range(rowvalues):
    for i in range(rowvalues):
        if i == 0:
            newimg1 = Image.new('RGBA', size=(1768, 2048), color=(0, 0, 0, 0))
            firstimage = Image.open(path[0])
            firstimage.show("image num:"+str(0))
            newimg1.paste(firstimage, (x, y))
            img1 = Image.open(path[i+1])
            img1 = ConvertpiltoOpencv(img1)
            img1 = denoiser(img1)
            img1 = ConvertOpencvToPill(img1)
            img = Image.fromarray(img1)
            newimg1.paste(img, (offsetx, offsety))

            newimg2 = Image.new('RGBA', size=(1768, 2048), color=(0, 0, 0, 0))
            firstimage = Image.open(path[i])
            firstimage = ConvertpiltoOpencv(firstimage)
            firstimage = denoiser(firstimage)
            firstimage = ConvertOpencvToPill(firstimage)
            firstimage = Image.fromarray(firstimage)
            newimg1.paste(firstimage, (x, y))
            firstimage.show("image num:"+str(i))
            img1 = Image.open(path[i+1])
            img1 = ConvertpiltoOpencv(img1)
            img1 = denoiser(img1)
            img1 = ConvertOpencvToPill(img1)

            newimg1.paste(img, (offsetx, offsety))
            x += offsetx
            # blend with alpha=0.4
            firstimage.show("image num:"+str(i))
            offsetx += offsetx
    y += offsety
finaleimage.show("man")

finaleimage.save("oof.png")


# grabimage=Image.open("000000000.png")

# grabimage1=Image.open("000000001.png")
# grabimage2=Image.open("000000002.png")
# grabimage3=Image.open("000000003.png")
# print(grabimage.size)
# backgroundimage = Image.new('RGBA', size=(length*2,width*2), color=(0, 0, 0, 0))
# backgroundimage.paste(grabimage)
# backgroundimage.paste(grabimage1,(width-offsetx, 0))
# backgroundimage.paste(grabimage2,(0, length-offsety))
# backgroundimage.paste(grabimage3,(width-offsetx,length-offsety))
# backgroundimage.show("man")
# backgroundimage.save("man.png")

# cv2.imshow('full image', sharpen)
# cv2.imshow("first",img)
cv2.waitKey(0)

# cv2.waitKey(2000)

cv2.destroyAllWindows()

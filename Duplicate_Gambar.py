import cv2
import numpy as np
import glob
from PIL import Image, ImageFilter, ImageOps, ImageEnhance
from numpy import *
import os
import imutils


def scale(MAIN_IMAGE_PATH, IMAGE_SAVE, batas_scale_bawah, batas_scale_atas, step_scale):
    for image_path in glob.glob(MAIN_IMAGE_PATH):
        s = 0
        for scale_percent in np.arange(batas_scale_bawah, batas_scale_atas+step_scale, step_scale):
            s += 1
            img = cv2.imread(image_path)
            image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            width = int(image.shape[1])
            height = int(image.shape[0])
            center = (width / 2, height / 2)
            M = cv2.getRotationMatrix2D(center, 0, scale_percent/100)
            image2 = cv2.warpAffine(image, M, (width, height))

            Image.fromarray(image2).save(f"{IMAGE_SAVE}/"+os.path.basename(image_path).split('.')[0]+"_scale_"+str(scale_percent)+"_"+str(s)+".jpg")

def rotate(MAIN_IMAGE_PATH, IMAGE_SAVE, batas_derajat_bawah, batas_derajat_atas, step_derajat):
    for image_path in glob.glob(MAIN_IMAGE_PATH):
        r = 0
        for angle in np.arange(batas_derajat_bawah, batas_derajat_atas+step_derajat, step_derajat):
            r += 1
            img = cv2.imread(image_path)
            image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            (h, w) = image.shape[:2]
            center = (w / 2, h / 2)
            M = cv2.getRotationMatrix2D(center, 0, 1)
            image2 = cv2.warpAffine(image, M, (w, h))
            rotated = imutils.rotate_bound(image2, angle=angle)

            Image.fromarray(rotated).save(f"{IMAGE_SAVE}/"+os.path.basename(image_path).split('.')[0]+"_rotate_"+str(r)+".jpg")

def translasi(MAIN_IMAGE_PATH, IMAGE_SAVE):
    t = 0
    for image_path in glob.glob(MAIN_IMAGE_PATH):
        t += 1
        bil_random_w = np.random.uniform(-0.4, 0.4, 1)
        bil_random_h = np.random.uniform(-0.4, 0.4, 1)
        img = cv2.imread(image_path)
        image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        width = int(image.shape[1])
        height = int(image.shape[0])
        M = np.float32([[1, 0, bil_random_w[0] * width/2], [0, 1, bil_random_h[0] * height/2]])
        image2 = cv2.warpAffine(image, M, (width, height))

        Image.fromarray(image2).save(f"{IMAGE_SAVE}/"+os.path.basename(image_path).split('.')[0]+"_translasi_"+str(t)+".jpg")

def crop_image(MAIN_IMAGE_PATH, IMAGE_SAVE):
    c = 0
    for image_path in glob.glob(MAIN_IMAGE_PATH):
        c += 1
        bil_random_kr = np.random.uniform(0.1, 0.25, 1)
        bil_random_at = np.random.uniform(0.75, 1, 1)
        bil_random_kn = np.random.uniform(0.75, 1, 1)
        bil_random_bw = np.random.uniform(0.1, 0.25, 1)
        image = Image.open(image_path)
        width, height = image.size
        kiri = int(width * bil_random_kr)
        atas = int(height * bil_random_at)
        kanan = int(width * bil_random_kn)
        bawah = int(height * bil_random_bw)
        image2 = image.crop((kiri, bawah, kanan, atas))

        image2.save(f"{IMAGE_SAVE}/"+os.path.basename(image_path).split('.')[0]+"_crop_"+str(c)+".jpg")

# TAMBAH ALFEND
def flip_image(MAIN_IMAGE_PATH, IMAGE_SAVE):
    f = 0
    for image_path in glob.glob(MAIN_IMAGE_PATH):
        f += 1
        image = Image.open(image_path)
        image2 = ImageOps.flip(image)

        image2.save(f"{IMAGE_SAVE}/"+os.path.basename(image_path).split('.')[0]+"_flip_"+str(f)+".jpg")

def blur_image(MAIN_IMAGE_PATH, IMAGE_SAVE):
    b = 0
    for image_path in glob.glob(MAIN_IMAGE_PATH):
        b += 1
        image = Image.open(image_path)
        image2 = image.filter(ImageFilter.BLUR)

        image2.save(f"{IMAGE_SAVE}/"+os.path.basename(image_path).split('.')[0]+"_blur_"+str(b)+".jpg")

""" 
def invert_image(MAIN_IMAGE_PATH, IMAGE_SAVE):
    i = 0
    for image_path in glob.glob(MAIN_IMAGE_PATH):
        i += 1
        image = Image.open(image_path)
        image2 = ImageOps.invert(image)

        image2.save(f"{IMAGE_SAVE}/"+os.path.basename(image_path).split('.')[0]+"_invert_"+str(i)+".jpg")

def solarize_image(MAIN_IMAGE_PATH, IMAGE_SAVE):
    s = 0
    for image_path in glob.glob(MAIN_IMAGE_PATH):
        s += 1
        image = Image.open(image_path)
        image2 = ImageOps.solarize(image)

        image2.save(f"{IMAGE_SAVE}/"+os.path.basename(image_path).split('.')[0]+"_solarize_"+str(s)+".jpg")
"""

def posterize_image(MAIN_IMAGE_PATH, IMAGE_SAVE):
    p = 0
    for image_path in glob.glob(MAIN_IMAGE_PATH):
        p += 1
        image = Image.open(image_path)
        image2 = ImageOps.posterize(image, 4)

        image2.save(f"{IMAGE_SAVE}/"+os.path.basename(image_path).split('.')[0]+"_posterize_"+str(p)+".jpg")

def sharpen_image(MAIN_IMAGE_PATH, IMAGE_SAVE):
    sh = 0
    for image_path in glob.glob(MAIN_IMAGE_PATH):
        sh += 1
        image = Image.open(image_path)
        image2 = ImageEnhance.Sharpness(image).enhance(3.0)

        image2.save(f"{IMAGE_SAVE}/"+os.path.basename(image_path).split('.')[0]+"_sharpen_"+str(sh)+".jpg")

def color_image(MAIN_IMAGE_PATH, IMAGE_SAVE):
    co = 0
    for image_path in glob.glob(MAIN_IMAGE_PATH):
        co += 1
        image = Image.open(image_path)
        image2 = ImageEnhance.Color(image).enhance(1.5)

        image2.save(f"{IMAGE_SAVE}/"+os.path.basename(image_path).split('.')[0]+"_color_"+str(co)+".jpg")

def contrast_image(MAIN_IMAGE_PATH, IMAGE_SAVE):
    co = 0
    for image_path in glob.glob(MAIN_IMAGE_PATH):
        co += 1
        image = Image.open(image_path)
        image2 = ImageEnhance.Contrast(image).enhance(1.5)

        image2.save(f"{IMAGE_SAVE}/"+os.path.basename(image_path).split('.')[0]+"_contrast_"+str(co)+".jpg")

def brightness_image(MAIN_IMAGE_PATH, IMAGE_SAVE):
    br = 0
    for image_path in glob.glob(MAIN_IMAGE_PATH):
        br += 1
        image = Image.open(image_path)
        image2 = ImageEnhance.Brightness(image).enhance(1.5)

        image2.save(f"{IMAGE_SAVE}/"+os.path.basename(image_path).split('.')[0]+"_brightness_"+str(br)+".jpg")

def resize_image(MAIN_IMAGE_PATH, IMAGE_SAVE):
    rs = 0
    for image_path in glob.glob(MAIN_IMAGE_PATH):
        rs += 1
        image = Image.open(image_path)
        image2 = image.resize((224, 224))

        image2.save(f"{IMAGE_SAVE}/"+os.path.basename(image_path).split('.')[0]+"_resize_"+str(rs)+".jpg")



MAIN_IMAGE_PATH = "D:/Dataset TA Baru/Defect 2/*.jpg" 
IMAGE_SAVE = "D:/Dataset Insulation Trafo 2/Train/Defect" 

crop_image(MAIN_IMAGE_PATH, IMAGE_SAVE)
translasi(MAIN_IMAGE_PATH, IMAGE_SAVE)
rotate(MAIN_IMAGE_PATH, IMAGE_SAVE, batas_derajat_bawah = 30, batas_derajat_atas = 270, step_derajat = 30)
scale(MAIN_IMAGE_PATH, IMAGE_SAVE, batas_scale_bawah = 60, batas_scale_atas = 140, step_scale = 10)
# TAMBAH ALFEND
flip_image(MAIN_IMAGE_PATH, IMAGE_SAVE)
blur_image(MAIN_IMAGE_PATH, IMAGE_SAVE)
#invert_image(MAIN_IMAGE_PATH, IMAGE_SAVE)
#solarize_image(MAIN_IMAGE_PATH, IMAGE_SAVE)
posterize_image(MAIN_IMAGE_PATH, IMAGE_SAVE)
sharpen_image(MAIN_IMAGE_PATH, IMAGE_SAVE)
color_image(MAIN_IMAGE_PATH, IMAGE_SAVE)
contrast_image(MAIN_IMAGE_PATH, IMAGE_SAVE)
brightness_image(MAIN_IMAGE_PATH, IMAGE_SAVE)
resize_image(MAIN_IMAGE_PATH, IMAGE_SAVE)

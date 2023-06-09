from math import sqrt, log10
import numpy as np
from numpy.fft import fft2, ifft2
from scipy.signal import gaussian, convolve2d
import matplotlib.pyplot as plt
from skimage.restoration import denoise_wavelet
from copy import deepcopy
import random



# Calculate MSE
def mse(img, filtred_img):
   

    sum1 = 0.0
    sum2 = 0.0
    img_copy = deepcopy(img)
    mse = 0.0
    psnr = 0.0
    for i in range(img.shape[0]):
         for j in range(img.shape[1]):
              sum1 = ((img[i][j] - filtred_img[i][j])**2)
              img_copy[i][j] = sum1
              
    sum2 = np.sum(img_copy)
    mse = sum2 / (np.array(img_copy).size)

    return mse






# Calculate PSNR
def psnr(img, filtred_img):
    

    sum1 = 0.0
    sum2 = 0.0
    img_copy = deepcopy(img)
    mse = 0.0
    psnr = 0.0
    for i in range(img.shape[0]):
         for j in range(img.shape[1]):
              sum1 = ((img[i][j] - filtred_img[i][j])**2)
              img_copy[i][j] = sum1
              
    sum2 = np.sum(img_copy)
    mse = sum2 / (np.array(img_copy).size)
    if(mse == 0):
         
         return 100
    max_pixel = 255.0  # for 8bit integer
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr




# Calculate SNR
def snr(unnoised,origin):
    som1 = 0.0
    som2 = 0.0
    for i in range(origin.shape[0]):
        for y in range(origin.shape[1]):
            som1 = som1 + (origin[i][y])**2
            som2 = som2 + (unnoised[i][y] - (origin[i][y]))**2
    return 10*log10(som1/som2)


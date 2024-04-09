import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from projeto_funcs import *

fig = cv2.imread("Imagens_DESENVOLVIMENTO2/riscada2.png", cv2.IMREAD_COLOR)
fig_gray = cv2.imread("Imagens_DESENVOLVIMENTO2/riscada1.png", cv2.IMREAD_GRAYSCALE)

#transforma em escala de cinza

fig_S = cv2.cvtColor(fig,cv2.COLOR_BGR2HLS)[:,:,2]

sobelX = cv2.Sobel(fig_S, cv2.CV_16S, 1, 0)
sobelX = np.uint8(np.absolute(sobelX))
sobelX = cv2.convertScaleAbs(sobelX)

sobelX2 = cv2.Sobel(fig_gray, cv2.CV_16S, 1, 0)
sobelX2 = np.uint8(np.absolute(sobelX2))
sobelX2 = cv2.convertScaleAbs(sobelX2)

sobelY = cv2.Sobel(fig_gray, cv2.CV_16S, 0, 1)
sobelY = np.uint8(np.absolute(sobelY))
sobelY = cv2.convertScaleAbs(sobelY)

sobel = sobelX2+sobelY

riscos = np.where(sobel>40,255,0).astype(np.uint8)
#print(risco.sum())

#Plots
pic = plt.figure(figsize = (6,3))

pic.add_subplot(1, 3, 1)
plt.imshow(sobelX2+sobelY,cmap='gray')

pic.add_subplot(1, 3, 2)
plt.imshow(cv2.cvtColor(fig,cv2.COLOR_BGR2HLS)[:,:,1],cmap='gray')

pic.add_subplot(1, 3, 3)
plt.imshow(riscos,cmap='gray')


plt.show()

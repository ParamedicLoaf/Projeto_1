import cv2
import matplotlib.pyplot as plt
import numpy as np


fig_boa = cv2.imread("Arquivos_DESENVOLVIMENTO/_boa_01.png", cv2.IMREAD_GRAYSCALE)
fig_boa2 = cv2.imread("Arquivos_DESENVOLVIMENTO/_boa_02.png", cv2.IMREAD_GRAYSCALE)
fig_boa3 = cv2.imread("Arquivos_DESENVOLVIMENTO/_boa_03.png", cv2.IMREAD_GRAYSCALE)
fig_boa4 = cv2.imread("Arquivos_DESENVOLVIMENTO/_boa_04.png", cv2.IMREAD_GRAYSCALE)
fig_boa5 = cv2.imread("Arquivos_DESENVOLVIMENTO/_boa_05.png", cv2.IMREAD_GRAYSCALE)

fig = cv2.imread("Arquivos_DESENVOLVIMENTO/quebrada_01.png", cv2.IMREAD_GRAYSCALE)
fig2 = cv2.imread("Arquivos_DESENVOLVIMENTO/quebrada_02.png", cv2.IMREAD_GRAYSCALE)
fig3 = cv2.imread("Arquivos_DESENVOLVIMENTO/quebrada_03.png", cv2.IMREAD_GRAYSCALE)

fig4 = cv2.imread("Arquivos_DESENVOLVIMENTO/amassada1.png", cv2.IMREAD_GRAYSCALE)
fig5 = cv2.imread("Arquivos_DESENVOLVIMENTO/amassada2.png", cv2.IMREAD_GRAYSCALE)
fig6 = cv2.imread("Arquivos_DESENVOLVIMENTO/amassada3.png", cv2.IMREAD_GRAYSCALE)

figs = [fig_boa,fig_boa2,fig_boa3,fig_boa4,fig_boa5,fig,fig2,fig3,fig4,fig5,fig6]

if fig is None:
    print("File not found. Bye!")
    exit(0)
#else:
    #plt.imshow(fig)
    #plt.show()

def binariza(fig):
    thresh = 150
    fig_bw = cv2.threshold(fig, thresh, 255, cv2.THRESH_BINARY)[1]
    fig_bw = cv2.erode(fig_bw, np.ones((3, 3), np.uint8), iterations=2)
    fig_bw = cv2.dilate(fig_bw, np.ones((5, 5), np.uint8), iterations=2)

    return fig_bw

def compara_area(fig):
    fig_bw = np.where((fig<130),255,0).astype(np.uint8)

    contours, _ = cv2.findContours(fig_bw, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    for cnt in contours:
        if (cv2.contourArea(cnt)>5000): #filtra ruídos
            hull = cv2.convexHull(cnt)
            ellipse = cv2.fitEllipse(cnt)
            x,y,w,h = cv2.boundingRect(cnt)
            difference = abs(cv2.contourArea(hull)-w*h)
            difference2 = abs(cv2.contourArea(cnt))
            img_out = cv2.drawContours(fig, [cnt], -1, (255,255,0), 2)

    print(difference)
    print('w: ',w,' h: ',h)
    
    plt.imshow(img_out,cmap='grey')
    plt.show()

    lista_areas = []
    for contour in contours:
        area = cv2.contourArea(contour)
        lista_areas.append(area)

    if len(lista_areas) > 0:
        last_area = lista_areas[-1]
        if last_area > 260000 or last_area < 240000:
            return "Peça amassada"
        else:
            return "Peça não amassada"

i = 0
for fig in figs:
    print(i)
    x=compara_area(fig)
    print(' ')
    i = i+1
#print(x)


#plt.imshow(fig)
    #plt.show()
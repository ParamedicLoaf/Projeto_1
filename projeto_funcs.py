import cv2
import matplotlib.pyplot as plt
import numpy as np


def rot(teta):
    TH_rot = np.array( [[np.cos(teta),-np.sin(teta),0],
                        [np.sin(teta), np.cos(teta),0],
                        [0           ,0           ,1]], dtype = "float")
    return TH_rot

def reverse_mapping(fig,TH):
    (h,w,z) = fig.shape
    TH_inv = np.linalg.inv(TH)

    #output
    out = np.zeros((h,w,z), dtype = np.uint8)

    for u in range(w):
        for v in range(h):

            p1 = np.array([u,v,1]).transpose()
            p0 = np.matmul(TH_inv,p1)

            x = int(p0[0]/p0[2])
            y = int(p0[1]/p0[2])

            if (x>=0) and(x<w) and (y>=0) and (y<h):
                out[v,u] = fig[y,x]
        
    return out

def extrai_pilula(image, angle, x,y,wi,hi): #extrai a pílula da imagem e corrige rotação

    if wi>hi:
        angle = angle-90
        w = wi
        h = hi
    else:
        w = hi
        h = wi

    rot_mat = cv2.getRotationMatrix2D((x,y), angle, 1.0)

    sup_dir = np.array([x+w/2,y-h/2,1]).transpose().astype(np.uint32)
    sup_esq = np.array([x-w/2,y-h/2,1]).transpose().astype(np.uint32)
    inf_esq = np.array([x-w/2,y+h/2,1]).transpose().astype(np.uint32)
    inf_dir = np.array([x+w/2,y+h/2,1]).transpose().astype(np.uint32)

    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result[sup_esq[1]:inf_esq[1]+1,sup_esq[0]:sup_dir[0]+1,:],w,h,x,y

def cor_certa(pill): #checa se a cor está certa
    pillHue = cv2.cvtColor(pill,cv2.COLOR_BGR2HLS)[:,:,0]
    vermelho = np.where((pillHue<8)&(pillHue>0),255,0).astype(np.uint8)

    contours, hierarchy = cv2.findContours(vermelho, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    cor = False
    altura = 0
    if len(contours)>0:
        for cnt in contours:
            if (cv2.contourArea(cnt)>1000): #filtra ruídos
                x,y,h,w = cv2.boundingRect(cnt) #obtém retângulo que enclausula a parte colorida do remédio
                altura = w
                cor = True

    return vermelho,cor,altura

def esta_inteira(pill):
    inteira = False

    #transforma em escala de cinza
    fig_grey = cv2.cvtColor(pill,cv2.COLOR_BGR2GRAY)
    pillHue = cv2.cvtColor(pill,cv2.COLOR_BGR2HLS)[:,:,0]

    #obtém uma máscara e contornos
    pill_mask = np.where(fig_grey>225,255,0).astype(np.uint8)

    preto = np.where(fig_grey<60,255,0).astype(np.uint8)
    contours, hierarchy = cv2.findContours(preto, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours2 = max(contours, key=lambda x: cv2.contourArea(x))

    i = 0
    for cnt in contours:
        if  (cv2.contourArea(cnt))>100: #filtra ruídos
            i = i+1

    if np.sum(pill_mask)<5 and (cv2.contourArea(cv2.convexHull(contours2))-cv2.contourArea(contours2))<1900 and i==1:
        inteira = True

    print(np.sum(pill_mask))
    print((cv2.contourArea(cv2.convexHull(contours2))-cv2.contourArea(contours2)))
    print(i)

    return pill_mask,inteira,preto



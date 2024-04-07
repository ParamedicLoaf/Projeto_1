import cv2
import matplotlib.pyplot as plt
import numpy as np


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

def esta_inteira(pill): #verifica se está quebrada
    inteira = False

    #transforma em escala de cinza
    fig_grey = cv2.cvtColor(pill,cv2.COLOR_BGR2GRAY)
    pillHue = cv2.cvtColor(pill,cv2.COLOR_BGR2HLS)[:,:,0]

    #obtém uma máscara e contornos
    pill_mask = np.where(fig_grey>225,255,0).astype(np.uint8) #procura por áreas de pó branco dentro da figura

    preto = np.where(fig_grey<60,255,0).astype(np.uint8) #analisa a área preta da pílula
    contours, hierarchy = cv2.findContours(preto, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours2 = max(contours, key=lambda x: cv2.contourArea(x))

    i = 0
    for cnt in contours: # Conta quantos contornos pretos há na figura. Se for mais de 1, está quebrado
        if  (cv2.contourArea(cnt))>100: #filtra ruídos
            i = i+1

    if np.sum(pill_mask)<5 and (cv2.contourArea(cv2.convexHull(contours2))-cv2.contourArea(contours2))<5000 and i==1:
        inteira = True

    #print(np.sum(pill_mask))
    #print((cv2.contourArea(cv2.convexHull(contours2))-cv2.contourArea(contours2)))
    #print(i)

    return pill_mask,inteira,preto



def forma_ok(fig): # verifica se está amassada
    fig_grey = cv2.cvtColor(fig,cv2.COLOR_BGR2GRAY)

    h,w = fig_grey.shape #dimensão da figura 
    fig_bw = np.where((fig_grey<130),255,0).astype(np.uint8) #pega apenas a pilular para ver seu formato

    contours, _ = cv2.findContours(fig_bw, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    for cnt in contours:
        if (cv2.contourArea(cnt)>6000): #filtra ruídos
            hull = cv2.convexHull(cnt)
            difference = abs(cv2.contourArea(hull)-w*h) #analisa a differença de área entre a área total e a área da pílula

            #print('diferenca: ',difference)

            if difference > 25000 or difference < 21000: #Como a imagem fig vêm de extrai pílula, os valores entre a área total e a área da pílula são conhecidos
                return False
            else:
                return True
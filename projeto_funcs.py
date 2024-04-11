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
    pill_mask = np.where(fig_grey>237,255,0).astype(np.uint8) #procura por áreas de pó branco dentro da figura

    preto = np.where(fig_grey<60,255,0).astype(np.uint8) #analisa a área preta da pílula
    
    '''
    kernel = np.ones((3,3),np.uint8)
    preto = cv2.dilate(preto,kernel,iterations = 3) 
    preto = cv2.erode(preto,kernel,iterations = 1)'''

    contours, hierarchy = cv2.findContours(preto, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours2 = max(contours, key=lambda x: cv2.contourArea(x))

    i = 0
    for cnt in contours: # Conta quantos contornos pretos há na figura. Se for mais de 1, está quebrado
        if  (cv2.contourArea(cnt))>500: #filtra ruídos
            i = i+1

    distances =[]
    hull = cv2.convexHull(contours2, returnPoints=False)
    defects = cv2.convexityDefects(contours2,hull)

    for j in range(defects.shape[0]):  # calculate the angle
        s, e, f, d = defects[j][0]
        distances.append(d)

    if np.sum(pill_mask)<5 and max(distances)<4000 and i==1:
        inteira = True

    
    #cv2.imshow("Img1 with keypoints",preto)
    #cv2.waitKey(0)
    #print(np.sum(pill_mask))
    #print(max(distances))
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

            if difference > 6750 or difference < 5250: #Como a imagem fig vêm de extrai pílula, os valores entre a área total e a área da pílula são conhecidos
                return False
            else:
                return True
            
def sem_riscos(fig):
    
    fig_H = cv2.cvtColor(fig,cv2.COLOR_BGR2HLS)[:,:,0] #Hue
    fig_gray = cv2.cvtColor(fig,cv2.COLOR_BGR2GRAY) #cinza
    
    #contornos em derivada
    sobelX = cv2.Sobel(fig_gray, cv2.CV_16S, 1, 0)
    sobelX = np.uint8(np.absolute(sobelX))
    sobelX = cv2.convertScaleAbs(sobelX)

    sobelY = cv2.Sobel(fig_gray, cv2.CV_16S, 0, 1)
    sobelY = np.uint8(np.absolute(sobelY))
    sobelY = cv2.convertScaleAbs(sobelY)

    sobel = sobelX+sobelY

    
    # pega a região vermelha
    fig_vermelho = np.where((fig_H>3) & (fig_H<8),255,0).astype(np.uint8)
    
    fig_vermelho = cv2.dilate(fig_vermelho,np.ones((2,2),np.uint8),iterations=3)
    fig_vermelho = cv2.erode(fig_vermelho,np.ones((3,3),np.uint8),iterations=5)

    # considera riscos apenas na região vermelha
    riscos = np.where((sobel>40)&(fig_vermelho>100),255,0).astype(np.uint8)

    contours, _ = cv2.findContours(riscos, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    areas = []
    for cnt in contours:
        areas.append(cv2.contourArea(cnt))

    # ignora as 6 maiores áreas (provenientes dos números e contornos da pílula)
    result = sorted(areas, reverse=True)
    result = result[6:]

    # soma as áreas e usa o resultado para determinar riscadas
    if sum(result) > 800:
        return False
    else:
        return True

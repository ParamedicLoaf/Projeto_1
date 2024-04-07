import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from projeto_funcs import *

fig = cv2.imread("Arquivos_DESENVOLVIMENTO/quebrada_03.png", cv2.IMREAD_COLOR)

#transforma em escala de cinza
fig_grey = cv2.cvtColor(fig,cv2.COLOR_BGR2GRAY)

#obtém uma máscara e contornos
pill_mask = np.where((fig_grey<120) & (fig_grey>0),255,0).astype(np.uint8)
contours, hierarchy = cv2.findContours(pill_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

i=0

data = []
for cnt in contours:
    if (cv2.contourArea(cnt)>1500): #filtra ruídos
        i = i+1
        
        (x,y),(h,w),angle = cv2.minAreaRect(cnt) #obtém retângulo que enclausula o remédio
        #print('angle:',angle, ' x:',x,' y:',y,' w:',w,' h:',h) #DEBUG

        pill = extrai_pilula(fig,angle,x,y,w,h) #extrai a pílula, corrigindo rotação

        condicao = 'OK'
        cor = cor_certa(pill[0]) #verifica se a cor está certa
        
        if cor[1]==False:
            condicao = 'COR'
        else:
            forma = forma_ok(pill[0])
            if forma==False:
                condicao = 'AMASSADA'
            else:
                inteira = esta_inteira(pill[0]) #verifica se está inteira
                if inteira[1]==False:
                    condicao = 'QUEBRADA'

        if condicao=='OK':
            w = pill[1]
            h1 = pill[2]
            h2 = cor[2]
            x = pill[3]
            y = pill[4]

            data.append([condicao,x,y,w,h1,h2])
        else:
            x = pill[3]
            y = pill[4]
            data.append([condicao,x,y,' ',' ',' '])

print('pililas: ',i)

# Create the pandas DataFrame
df = pd.DataFrame(data, columns=['Status', 'Pos X', 'Pos Y', 'W','H1','H2'])
 
# print dataframe.
print(df)

df.to_excel("output.xlsx")
'''
pic = plt.figure(figsize = (8,5))
pic.add_subplot(2,3,1)
plt.imshow(fig_grey,cmap='gray')

pic.add_subplot(2,3,2)
plt.imshow(pill_mask,cmap='gray')

pic.add_subplot(2,3,3)
plt.imshow(cv2.cvtColor(pill[0],cv2.COLOR_BGR2RGB))

pic.add_subplot(2,3,4)
plt.imshow(cv2.cvtColor(fig,cv2.COLOR_BGR2HLS)[:,:,0],cmap='gray')

pic.add_subplot(2,3,5)
plt.imshow(cv2.cvtColor(cor[0],cv2.COLOR_BGR2RGB))

pic.add_subplot(2,3,6)
plt.imshow(inteira[2],cmap='gray')

plt.show()'''
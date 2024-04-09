import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from projeto_funcs import *

fig = cv2.imread("Imagens_TESTE_VALIDACAO/Imagem_VALIDACAO_2.png", cv2.IMREAD_COLOR)

#transforma em escala de cinza
fig_grey = cv2.cvtColor(fig,cv2.COLOR_BGR2GRAY)

#obtém uma máscara e contornos
pill_mask = np.where((fig_grey<120) & (fig_grey>0),255,0).astype(np.uint8)
contours, hierarchy = cv2.findContours(pill_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

i=1

data = []
for cnt in contours:
    if (cv2.contourArea(cnt)>1500): #filtra ruídos
        i = i+1
        
        (x,y),(h,w),angle = cv2.minAreaRect(cnt) #obtém retângulo que enclausula o remédio
        #print('angle:',angle, ' x:',x,' y:',y,' w:',w,' h:',h) #DEBUG

        pill = extrai_pilula(fig,angle,x,y,w,h) #extrai a pílula, corrigindo rotação
        cv2.imshow("Img1 with keypoints",pill[0])
        cv2.waitKey(0)

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
                else:
                    nao_riscada = sem_riscos(pill[0])
                    if nao_riscada==False:
                        condicao = 'RISCADA'

        if condicao=='OK':
            w = int(pill[1])
            h1 = int(pill[2])
            h2 = int(cor[2])
            x = int(pill[3])
            y = int(pill[4])

            data.append([condicao,x,y,w,h1,h2])
        else:
            x = int(pill[3])
            y = int(pill[4])
            data.append([condicao,x,y,' ',' ',' '])

# Create the pandas DataFrame
df = pd.DataFrame(data, columns=['Status', 'Pos X', 'Pos Y', 'W','H1','H2'])
 
# print dataframe.
print(df)

df.to_excel("output.xlsx")

plt.imshow(cv2.cvtColor(fig,cv2.COLOR_BGR2RGB))
plt.show()

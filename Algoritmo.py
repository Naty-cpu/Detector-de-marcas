import cv
import numpy as np

img = cv.imread('Atividades_escaneadas (2)_page-0002.jpg',1) #Imagem de busca 
gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

template = cv.imread('template_direito.jpg',cv.IMREAD_GRAYSCALE)
w, h = template.shape[::-1]

result = cv.matchTemplate(gray_img, template, cv.TM_CCOEFF_NORMED)
loc = np.where(result >= 0.54)

for pt in zip(*loc[::-1]):
    cv.rectangle(img, pt, (pt[0] + w, pt[1] + h), (0, 255, 0), 3)

template2 = cv.imread('template_esquerdo.jpg',cv.IMREAD_GRAYSCALE)
r, f = template2.shape[::-1]

result2 = cv.matchTemplate(gray_img, template2, cv.TM_CCOEFF_NORMED)
loc2 = np.where(result2 >= 0.54)

for pt in zip(*loc2[::-1]):
    cv.rectangle(img, pt, (pt[0] + r, pt[1] + f), (0, 255, 0), 3)

img = cv.resize(img, (1080, 660))
cv.imshow("img", img)
cv.waitKey(0)

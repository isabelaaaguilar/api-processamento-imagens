import cv2 as cv
import numpy as np
img = cv.imread('resized_image1.jpg',0)  

img2 = cv.imread('resized_image1.jpg') #Colorida
template = cv.imread('mousecrop.jpg',0)
w, h = template.shape[::-1]

# faz a correlação cruzada
res = cv.matchTemplate(img, template, eval('cv.TM_CCOEFF_NORMED') )

# pega algumas variaveis para montar o retangulo da area de detecção
min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
top_left = max_loc
bottom_right = (top_left[0] + w, top_left[1] + h)
# monta retangulo na iamgem
cv.rectangle(img2,top_left, bottom_right, (0, 0, 255), 2)

cv.imshow("Image", img2)

cv.waitKey(0)
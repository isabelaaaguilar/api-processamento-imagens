
"""
Ciência da Computação - Unidade Praça da Liberdade

Integrantes:

Isabela Regina Aguilar 706002
Lucas Milard de Souza Freitas 704593
Rossana de Oliveira Souza  705085

"""



import cv2 as cv

img = cv.imread('resized_image1.jpg', 0)  # Imagem em tons de cinza (cv2.IMREAD_GRAYSCALE)
img2 = cv.imread('resized_image1.jpg') # Imagem Colorida
template = cv.imread('mousecrop.jpg', 0)
w, h = template.shape[::-1]

# faz a correlação cruzada
res = cv.matchTemplate(img, template, eval('cv.TM_CCOEFF_NORMED') )

# extrai algumas variáveis e coordenadas para montar o retângulo da área de detecção
min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
top_left = max_loc
bottom_right = (top_left[0] + w, top_left[1] + h)

# constrói o retângulo na primeira imagem lida
cv.rectangle(img2,top_left, bottom_right, (0, 0, 255), 2)

# Exibe a imagem já sinalizada no local com melhor correspondência com o corte de imagem recebido
cv.imshow("Image", img2)
cv.waitKey(0)
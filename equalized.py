
"""
Ciência da Computação - Unidade Praça da Liberdade

Integrantes:

Isabela Regina Aguilar 706002
Lucas Milard de Souza Freitas 704593
Rossana de Oliveira Souza  705085

"""


# Teste realizado para equalizar imagens em python
import cv2 as cv
image_path = 'download.png'
imge = cv.imread(image_path, 0) # Converte imagem em tons de cinza
equ = cv.equalizeHist(imge) # Função da biblioteca opencv para realizar a equalização de histograma
cv.imwrite(image_path,equ) # Salva imagem equalizada
# Teste realizado para equalizar imagens em python
import cv2 as cv
image_path = 'download.png'
imge = cv.imread(image_path, 0) # Converte imagem em tons de cinza
equ = cv.equalizeHist(imge) # Função da biblioteca opencv para realizar a equalização de histograma
cv.imwrite(image_path,equ) # Salva imagem equalizada
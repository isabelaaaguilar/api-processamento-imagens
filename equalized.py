
import cv2 as cv
image_path = 'download.png'
imge = cv.imread(image_path,0 )
equ = cv.equalizeHist(imge)
cv.imwrite(image_path,equ)
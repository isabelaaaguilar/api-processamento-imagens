import os
import shutil
from PIL import Image
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


def renomeiaESalva():
    newFolder = "grau_artrose_crop/"
    for x in range(5):
        folder = "xgboost/" + str(x) + "/"
        count = 1
        crop = "crop" + str(x) +".png"

        for file_name in os.listdir(folder):
            source = folder + file_name

            newFileName = "imagem_" + str(count) + "_grau_" + str(x) + ".jpg"
            destination = folder + newFileName


            basewidth = 500 # Redimensionando imagem
            img = Image.open(source)
            wpercent = (basewidth/float(img.size[0]))
            hsize = int((float(img.size[1])*float(wpercent)))
            img=img.resize((basewidth, hsize), Image.ANTIALIAS)
            img.save(source)

            img = Image.open(crop)
            wpercent = (basewidth/float(img.size[0]))
            hsize = int((float(img.size[1])*float(wpercent)))
            img=img.resize((basewidth, hsize), Image.ANTIALIAS)
            img.save(crop)

            imageCompare =  cv.imread(source, 0)
            print(crop)

            template = cv.imread(crop, 0)
            w, h = template.shape[::-1]

            # faz a correlação cruzada
            res = cv.matchTemplate(imageCompare, template, eval('cv.TM_CCOEFF_NORMED'))

            # extrai algumas variáveis e coordenadas para montar o retângulo da área de detecção
            _, _, _, maxLoc = cv.minMaxLoc(res)

            cropimage = imageCompare[maxLoc[1]:maxLoc[1]+h, maxLoc[0]:maxLoc[0]+w]
            cv.imwrite(source, cropimage)

            # Renomeia e salva cada imagem
            os.rename(source, destination)

            # Faz a cópia da imagem original para a pasta grau artrose
            newDestination = newFolder + newFileName
            shutil.copy(destination, newDestination)

            # Faz a transposição da imagem original
            img = Image.open(destination)
            transposeImage(destination, folder, newFolder, x, count, img, "_transpose")

            imge = cv.imread(destination,0)
            equ = cv.equalizeHist(imge)

            newFileName = "imagem_" + str(count) + "_equalizada_grau_" + str(x) + ".jpg"
            newdestination = folder + newFileName

            #equ.save(newdestination)
            cv.imwrite(newdestination,equ)
            equ =  Image.open(newdestination)

            # Faz a imagem equalizada, salva na pasta original e faz uma cópia na pasta grau artrose
            #imagemEqualizada = equalizationHistogram(destination, count, x, folder, newFolder)

            # Faz a transposição da imagem equalizada
            transposeImage(destination, folder, newFolder, x, count, equ, "_transpose_equalized")
           
            count += 1

def equalizationHistogram(destination, count, x, folder, newFolder): 
    # carrega a imagem em memória principal com a biblioteca openCv
    imagem = plt.imread(destination)
    # Converte a imagem para uma matriz de tons de cinza
    imagem = convert_to_gray(imagem, False)
    # Inicia o histograma - objeto com 255 valores num dicionário
    histograma = instantiate_histogram()
    # Quanta vezes cada valor de intensidade do histograma aparece na imagem
    histograma = count_intensity_values(histograma, imagem)
    # Calcula a probabilidade de cada pixel
    n_pixels = imagem.shape[0] * imagem.shape[1]
    hist_proba = get_hist_proba(histograma, n_pixels)
    # Calcula a probabilidade acumulada 
    accumulated_proba = get_accumulated_proba(hist_proba)
    # Novo objeto para mapear os valores de cinza
    new_gray_value = get_new_gray_value(accumulated_proba)
    # Aplica os novos valores na imagem original
    imagem = equalize_hist(imagem, new_gray_value)
    # Renomeia e salva a imagem equalizada
    newFileName = "imagem_" + str(count) + "_equalizada_grau_" + str(x) + ".jpg"
    destination = folder + newFileName
    # Converte a matriz de volta numa imagem
    im = Image.fromarray(imagem).convert('RGB')
    # Salva a imagem equalizada
    im.save(destination)
    # Faz a cópia da imagem equalizada para a pasta grau artrose
    newDestination = newFolder + newFileName
    shutil.copy(destination, newDestination)

    return im

def transposeImage(destination, folder, newFolder, x, count, img, nomeImagem):
   
    img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

    # transpoe a imagem e salva
    transposeFileName = "imagem_" + str(count) + str(nomeImagem) + "_grau_" + str(x) + ".jpg"
    destination = folder + transposeFileName
    img.save(destination)

    # faz a cópia para a pasta grau artrose
    newDestination = newFolder + transposeFileName
    shutil.copy(destination, newDestination)

def convert_to_gray(image, luma=False):
    if luma:
        params = [0.299, 0.589, 0.114]
    else:
        params = [0.2125, 0.7154, 0.0721]    
    gray_image = np.ceil(np.dot(image[...,:3], params))
 
    # Saturando os valores em 255
    gray_image[gray_image > 255] = 255
    
    return gray_image

def instantiate_histogram():
    hist_array = []
    
    for i in range(0,256):
        hist_array.append(str(i))
        hist_array.append(0)
    
    hist_dct = {hist_array[i]: hist_array[i + 1] for i in range(0, len(hist_array), 2)} 
    
    return hist_dct

def count_intensity_values(hist, img):
    for row in img:
       for column in row:
            hist[str(int(column))] = hist[str(int(column))] + 1
     
    return hist

def get_hist_proba(hist, n_pixels):
    hist_proba = {}
    for i in range(0, 256):
        hist_proba[str(i)] = hist[str(i)] / n_pixels
    
    return hist_proba

def get_accumulated_proba(hist_proba): 
    acc_proba = {}
    sum_proba = 0
    
    for i in range(0, 256):
        if i == 0:
            pass
        else: 
            sum_proba += hist_proba[str(i - 1)]
            
        acc_proba[str(i)] = hist_proba[str(i)] + sum_proba
    return acc_proba

def get_new_gray_value(acc_proba):
    new_gray_value = {}
    
    for i in range(0, 256):
        new_gray_value[str(i)] = np.ceil(acc_proba[str(i)] * 255)
    return new_gray_value

def equalize_hist(img, new_gray_value):
    for row in range(img.shape[0]):
        for column in range(img.shape[1]):
            img[row][column] = new_gray_value[str(int(img[row] [column]))]
            
    return img


def main():
    
    renomeiaESalva()
    
if __name__ == "__main__":
    main()



import os
import shutil
from PIL import Image
import cv2 as cv
import numpy as np
import matplotlib.image as image

#import matplotlib.pyplot as plt 
#from skimage import data



def renomeiaESalva():
    newFolder = "grau_artrose/"
    for x in range(5):
        folder = "imagens/" + str(x) + "/"
        count = 1

        for file_name in os.listdir(folder):
            source = folder + file_name

            newFileName = "imagem_" + str(count) + "_grau_" + str(x) + ".jpg"
            destination = folder + newFileName

            # renomeia cada imagem
            os.rename(source, destination)

            # faz a cópia para a pasta grau artrose
            newDestination = newFolder + newFileName
            shutil.copy(destination, newDestination)

            # faz a transposição da imagem
            transposeImage(destination, folder, newFolder, x, count)




            # carrega a imagem em memória principal com a biblioteca openCv
            imagem = image.imread(destination)
            #imagem = data.coffee()

            # Inicia o histograma - objeto com 255 valores num dicionário
            histograma = instantiate_histogram()
            # Quanta vezes cada valor de intensidade do histograma aparece na imagem
            histograma = count_intensity_values(histograma, imagem)

            n_pixels = imagem.shape[0] * imagem.shape[1]
            hist_proba = get_hist_proba(histograma, n_pixels)


            
            count += 1

def transposeImage(destination, folder, newFolder, x, count):
    img = Image.open(destination)
    img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

    # transpoe a imagem e salva
    transposeFileName = "imagem_" + str(count) + "_transpose_grau_" + str(x) + ".jpg"
    destination = folder + transposeFileName
    img.save(destination)

    # faz a cópia para a pasta grau artrose
    newDestination = newFolder + transposeFileName
    shutil.copy(destination, newDestination)

def instantiate_histogram():
    hist_array = []
    
    for i in range(0,256):
        hist_array.append(str(i))
        hist_array.append(0)
    
    hist_dct = {hist_array[i]: hist_array[i + 1] for i in range(0, len(hist_array), 2)} 
    
    return hist_dct

def count_intensity_values(hist, img):
    print(img)
    for row in img:
       for column in row:
            hist[str(int(column))] = hist[str(int(column))] + 1
     
    return hist

def get_hist_proba(hist, n_pixels):
    hist_proba = {}
    for i in range(0, 256):
        hist_proba[str(i)] = hist[str(i)] / n_pixels
    
    return hist_proba


def main():
    
    renomeiaESalva()
    
if __name__ == "__main__":
    main()



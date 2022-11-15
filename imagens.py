import os
import shutil
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt


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
            imagem = plt.imread(destination)
            # Converte a imagem para uma matriz de tons de cinza
            imagem = convert_to_gray(imagem, False)
            # Inicia o histograma - objeto com 255 valores num dicionário
            histograma = instantiate_histogram()
            # Quanta vezes cada valor de intensidade do histograma aparece na imagem
            histograma = count_intensity_values(histograma, imagem)
            # 
            n_pixels = imagem.shape[0] * imagem.shape[1]
            hist_proba = get_hist_proba(histograma, n_pixels)
            # Calcula a probabilidade acumulada 
            accumulated_proba = get_accumulated_proba(hist_proba)
            # Novo objeto para mapear os valores de cinza
            new_gray_value = get_new_gray_value(accumulated_proba)
            # Aplica os novos valores na imagem original
            imagem = equalize_hist(imagem, new_gray_value)

            # Renomei e salva a imagem equalizada
            newFileName = "imagem_" + str(count) + "_equalizada_grau_" + str(x) + ".jpg"
            destination = folder + newFileName

            im = Image.fromarray(imagem).convert('RGB')

            im.save(destination)



            
           
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



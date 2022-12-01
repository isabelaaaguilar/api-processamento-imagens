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

            # Renomeia e salva cada imagem
            os.rename(source, destination)

            # Faz a cópia da imagem original para a pasta grau artrose
            newDestination = newFolder + newFileName
            shutil.copy(destination, newDestination)

            # Faz a transposição da imagem original
            img = Image.open(destination)
            transposeImage(destination, folder, newFolder, x, count, img, "_transpose")

            # Faz a imagem equalizada, salva na pasta original e faz uma cópia na pasta grau artrose
            imagemEqualizada = equalizationHistogram(destination, count, x, folder, newFolder)

            # Faz a transposição da imagem equalizada
            transposeImage(destination, folder, newFolder, x, count, imagemEqualizada, "_transpose_equalized")
           
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

    # Transpõe a imagem e salva
    transposeFileName = "imagem_" + str(count) + str(nomeImagem) + "_grau_" + str(x) + ".jpg"
    destination = folder + transposeFileName
    img.save(destination)

    # Faz a cópia para a pasta grau artrose
    newDestination = newFolder + transposeFileName
    shutil.copy(destination, newDestination)

def convert_to_gray(image, luma=False):
    # Converte a imagem para tons de cinza e equaliza os valores de 0 a 255
    if luma:
        params = [0.299, 0.589, 0.114]
    else:
        params = [0.2125, 0.7154, 0.0721]    
    gray_image = np.ceil(np.dot(image[...,:3], params))
 
    # Saturando os valores em 255
    gray_image[gray_image > 255] = 255
    
    return gray_image

def instantiate_histogram():
    # Cria o histograma da imagem no formato de um array para os pixels de valor de 0 a 255
    hist_array = []
    
    for i in range(0,256):
        hist_array.append(str(i))
        hist_array.append(0)
    # O índice do vetor indica o valor do pixel (0-255) e o valor naquele índice é a quantidade de vezes em que o pixel aparece na imagem
    hist_dct = {hist_array[i]: hist_array[i + 1] for i in range(0, len(hist_array), 2)} 
    
    return hist_dct

def count_intensity_values(hist, img):
    # 
    for row in img:
       for column in row:
            hist[str(int(column))] = hist[str(int(column))] + 1
     
    return hist

def get_hist_proba(hist, n_pixels):
    #  Faz o histograma de probabilidade do pixel na imagem em função do histograma da própria imagem
    hist_proba = {}
    for i in range(0, 256):
        hist_proba[str(i)] = hist[str(i)] / n_pixels
    
    return hist_proba

def get_accumulated_proba(hist_proba): 
    # Faz a propabilidade acumulada em função do histograma de probabilidades
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
    # Em função da probabilidade acumulada os valores dos pixels são recalculados, ainda variando de 0 a 255, com objetivo de aumentar o contraste da imagem
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



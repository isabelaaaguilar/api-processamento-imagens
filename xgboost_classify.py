from os import path, listdir
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import os
import numpy as np

# Extrai qual grau de artrose a imagem tem
def extract_degree(img):
    imgName = img.split(".")
    degree = imgName[0][-1]
    return degree

# Extrai se é uma imagem com ou sem artrose
def binary_arthrosis(img):
    imgName = img.split(".")
    degree = imgName[0][-1]
    if degree == "0":
        return 0
    return 1

# Trata nome da imagem
def extract_name():
    all_image_path = [path.join('grau_artrose', p) for p in listdir('grau_artrose') if path.isfile(path.join('grau_artrose', p))]
    all_images = [path.replace('.',' ').replace("\\",' ').split()[1] for path in all_image_path if os.path.isfile(path)]
    return all_image_path, all_images


all_image_path, all_images = extract_name()
degrees = [extract_degree(img) for img in all_image_path]
has_arthrosis = [binary_arthrosis(img) for img in all_images]

#Data Frame separado por graus de artrose
dataDegree = list(zip(all_images, degrees))
imgDegree = pd.DataFrame(dataDegree, columns=['Image', 'Degree'])

#Data Frame separado em imagens com ou sem artrose
dataBinary = list(zip(all_images, has_arthrosis))
imgBinaryArthrosis = pd.DataFrame(dataBinary, columns=['Image', 'WithArthrosis'])

print(imgDegree)
print(imgBinaryArthrosis)
""" 
y = imgBinaryArthrosis["WithArthrosis"]
X = imgBinaryArthrosis["Image"]
# dividir entre conjuntos de treino e teste
train_X, test_X, train_y, test_y = train_test_split(X.values, y.values, test_size=0.2)


# instanciar o modelo XGBoost
model = XGBClassifier()
# chamar o fit para o modelo
model.fit(train_X, train_y, verbose=False)
# fazer previsões em cima do dataset de teste
predictions = model.predict(test_X)
print("Acurácia: {:.2f}".format(accuracy_score(predictions, test_y))) """
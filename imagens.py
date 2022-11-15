import os
import shutil
from PIL import Image


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

            transposeImage(destination, folder, newFolder, x, count)
            
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


def main():
    
    renomeiaESalva()
    
if __name__ == "__main__":
    main()



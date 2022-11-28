from flask import Flask, request, make_response
from PIL import Image
from flask_cors import CORS
import cv2 as cv
import PIL
import base64
import os

app = Flask(__name__)
CORS(app)

@app.route("/upload-image", methods=["POST"])
def home():
        # Imagem que deseja buscar o corte
        imageCompare = request.files["imageFiles"] 

        # Corte de uma imagem
        imageCrop = request.files["imageCrop"] 

        # Salva o path para a imagem
        upload_imageCompare_path, upload_imageCrop_path = savePath(imageCompare, imageCrop)
        
        # Redimensionando imagem
        resizeImage(upload_imageCompare_path, imageCompare, upload_imageCrop_path, imageCrop)
        
        # Trata imagens em escala de cinza e colorida
        imageCompareGray, imageCompareColorful, template, w, h = colorAndImageTemplate(imageCompare, upload_imageCrop_path)
        
        # faz a correlação cruzada
        res = cv.matchTemplate(imageCompareGray, template, eval('cv.TM_CCOEFF_NORMED'))

        # pega algumas variaveis para montar o retangulo da area de detecção
        imageCompareColorful = detectionArea(res, w ,h, imageCompareColorful)
        
        # Cria a imagem resultado que será devolvida para o front end da aplicação
        cv.imwrite("resultado.jpg", imageCompareColorful)

        # Tranforma o file da imagem de resultado numa string na base 64 para retornar para o front
        with open("resultado.jpg", "rb") as f:
                image_binary = f.read()
        response = make_response(base64.b64encode(image_binary))

        #remove os arquivos utilizados no último teste
        os.remove(imageCompare.filename)
        os.remove(imageCrop.filename)
        os.remove("resultado.jpg")

        # retorna a string para o front
        return response

 # Salva o path para a imagem
def savePath(imageCompare, imageCrop):
        upload_imageCompare_path = os.path.join(imageCompare.filename)
        imageCompare.save(upload_imageCompare_path)
        upload_imageCrop_path = os.path.join(imageCrop.filename)
        imageCrop.save(upload_imageCrop_path)
        return upload_imageCompare_path, upload_imageCrop_path

# Redimensionando imagem
def resizeImage(upload_imageCompare_path, imageCompare, upload_imageCrop_path, imageCrop):
        basewidth = 500 
        img = Image.open(upload_imageCompare_path)
        wpercent = (basewidth/float(img.size[0]))
        hsize = int((float(img.size[1])*float(wpercent)))
        img=img.resize((basewidth, hsize), PIL.Image.ANTIALIAS)
        img.save(imageCompare.filename)

        """     mgCrop = Image.open(upload_imageCrop_path)
        wpercent = (basewidth/float(imgCrop.size[0]))
        hsize = int((float(imgCrop.size[1])*float(wpercent)))
        imgCrop= imgCrop.resize((basewidth, hsize), PIL.Image.ANTIALIAS)
        imgCrop.save(imageCrop.filename)
 """
        imageCompareGray = cv.imread(imageCompare.filename, 0) # Escala de cinza
        imageCompareColorful = cv.imread(imageCompare.filename) # Colorida

        template = cv.imread(upload_imageCrop_path, 0) #cria o template
        w, h = template.shape[::-1]
        return imageCompareGray, imageCompareColorful, template, w, h

        # faz 0 coeficiente de correlação 
        res = cv.matchTemplate(imageCompareGray, template, eval('cv.TM_CCOEFF_NORMED'))

        # pega algumas variaveis para montar o retangulo da area de detecção
        min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
        top_left = max_loc
        bottom_right = (top_left[0] + w, top_left[1] + h)

        # monta retangulo na imagem
        cv.rectangle(imageCompareColorful,top_left, bottom_right, (0, 0, 255), 2)

        return imageCompareColorful

        with open("resultado.jpg", "rb") as f:
                image_binary = f.read()
        response = make_response(base64.b64encode(image_binary))

        # os.remove("crop.png")
        # os.remove("resultado.jpg")


        return response

app.run(debug=True)
from flask import Flask, request, send_file, make_response
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

        imageCompare = request.files["imageFiles"]        
         # Imagem que deseja buscar o corte
       
        imageCrop = request.files["imageCrop"]
        # Corte de uma imagem

        upload_imageCompare_path = os.path.join(imageCompare.filename)
        imageCompare.save(upload_imageCompare_path)
        upload_imageCrop_path = os.path.join(imageCrop.filename)
        imageCrop.save(upload_imageCrop_path)

        basewidth = 500 # Redimensionando imagm
        img = Image.open(upload_imageCompare_path)
        wpercent = (basewidth/float(img.size[0]))
        hsize = int((float(img.size[1])*float(wpercent)))
        img=img.resize((basewidth, hsize), PIL.Image.ANTIALIAS)
        img.save(imageCompare.filename)

        imgCrop = Image.open(upload_imageCrop_path)
        wpercent = (basewidth/float(imgCrop.size[0]))
        hsize = int((float(imgCrop.size[1])*float(wpercent)))
        imgCrop= imgCrop.resize((basewidth, hsize), PIL.Image.ANTIALIAS)
        imgCrop.save(imageCrop.filename)

        imageCompareGray = cv.imread(imageCompare.filename, 0) # Escala de cinza
        imageCompareColorful = cv.imread(imageCompare.filename) # Colorida
        template = cv.imread(upload_imageCrop_path, 0)
        w, h = template.shape[::-1]

        # faz a correlação cruzada
        res = cv.matchTemplate(imageCompareGray, template, eval('cv.TM_CCOEFF_NORMED'))

        # pega algumas variaveis para montar o retangulo da area de detecção
        min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
        top_left = max_loc
        bottom_right = (top_left[0] + w, top_left[1] + h)
        # monta retangulo na iamgem
        cv.rectangle(imageCompareColorful,top_left, bottom_right, (0, 0, 255), 2)

        cv.imshow("Image", imageCompareColorful)
        cv.imwrite("resultado.jpg", imageCompareColorful)

        with open("resultado.jpg", "rb") as f:
                image_binary = f.read()
        response = make_response(base64.b64encode(image_binary))
        return response
#send_file("resultado.jpg", mimetype='image/jpg')
app.run(debug=True)
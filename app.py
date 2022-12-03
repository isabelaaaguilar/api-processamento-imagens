from flask import Flask, request, send_file, jsonify
from PIL import Image
import cv2 as cv
from flask_cors import CORS
import tensorflow as tf
import os
import base64
import time

app = Flask(__name__)
cors = CORS(app)


IMAGE_SIZE = 224

def correlation(imageCompare_path, imageCompare, imageCrop_path):
        

        basewidth = 500 # Redimensionando imagm
        img = Image.open(imageCompare_path)
        wpercent = (basewidth/float(img.size[0]))
        hsize = int((float(img.size[1])*float(wpercent)))
        img=img.resize((basewidth, hsize), Image.ANTIALIAS)
        img.save(imageCompare.filename)

        imageCompareGray = cv.imread(imageCompare.filename, 0) # Escala de cinza
        imageCompareColorful = cv.imread(imageCompare.filename) # Colorida
        template = cv.imread(imageCrop_path, 0)
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

        with open("resultado.jpg", "rb") as image_file:
         encoded_string = base64.b64encode(image_file.read()).decode("utf-8")

        return encoded_string



def preprocess_image(image):
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [IMAGE_SIZE, IMAGE_SIZE])
    image /= 255.0  # normalize to [0,1] range

    return image


# Read the image from path and preprocess
def load_and_preprocess_image(path):
    image = tf.io.read_file(path)
    return preprocess_image(image)





# Predict & classify image
def classify(model, image_path, type):
    preprocessed_imgage = load_and_preprocess_image(image_path)
    preprocessed_imgage = tf.reshape(
        preprocessed_imgage, (1, IMAGE_SIZE, IMAGE_SIZE, 3)
    )

    inicio = time.time()
    prob = model.predict(preprocessed_imgage)
    #print(prob[0][0])
    #print("Probabilidade:  %5.2f" % (prob[0][0]))
    aux_prob = 0
    label = ''

    if type == "binary":
        label = "sem artrose" if prob[0][0] >= 0.5 else "artrose"
        aux_prob =  prob[0][0] if prob[0][0] >= 0.5 else 1 - prob[0][0]
    elif type == "degrees":
        if prob[0][0] >= 0.7:
            label = "grau 0"
            aux_prob = prob[0][0]
        elif prob[0][0] >= 0.5:
            label = "grau 1"
            aux_prob = prob[0][0]
        elif prob[0][0] >= 0.3:
            label = "grau 2"
            aux_prob = 1- prob[0][0]
        elif prob[0][0] >= 0.15:
            label = "grau 3"
            aux_prob = 1- prob[0][0]
        else:
            label = "grau 4"
            aux_prob = 1- prob[0][0]
        
    fim = time.time()
    execution_time = round(fim-inicio,2)

    classified_prob  = '{:.3f}'.format(aux_prob)
        
    #os.remove(image_path)
    return label, classified_prob, execution_time


def cnn_classify(image_path):
        models = ['50', '50V2', '101', '101V2', '152', '152V2'] # modelo CNN
        folder = 'models/resNet'
        binary = [] 
        degrees = []

        for i in range(6):
            source = folder + models[i]
            for model in os.listdir(source):
                print(source, model)
                cnn_model = tf.keras.models.load_model(source +'/' +model)
                type = model.split('.')[0]
                label, classified_prob, execution_time = classify(cnn_model, image_path, type) # classificação
                if(type == "binary"):
                    binary.append({
                        'model': 'resNet'+models[i], 
                        'label': label,
                        'prob': classified_prob,
                        'time': execution_time
                         })
                else:
                     degrees.append({
                        'model': 'resNet'+models[i],
                        'label': label,
                        'prob': classified_prob,
                        'time': execution_time
                         })
                

        #classifications = Classifications(binary, degrees)

        return binary, degrees



@app.route("/", methods=["POST"])
def home():
        imageCompare = request.files["imageFiles"] # Imagem que deseja buscar o corte
        imageCrop = request.files["imageCrop"] # Corte de uma imagem

        # salvar imagens
        imageCompare_path = os.path.join(imageCompare.filename)
        imageCompare.save(imageCompare_path)
        imageCrop_path = os.path.join(imageCrop.filename)
        imageCrop.save(imageCrop_path)

        binary, degrees = cnn_classify(imageCompare_path)
        correlation_image = correlation(imageCompare_path, imageCompare, imageCrop_path)
        response = {
                'correlation' : correlation_image,
                'classifications' : {
                        'cnn': {
                            'binary': binary,
                            'degrees': degrees
                        },
                        'random_forest': '',
                        'XGboost': ''
                }
        }

        return jsonify(response)

app.run(debug=True)
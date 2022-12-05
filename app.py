from flask import Flask, request, jsonify
from PIL import Image
import cv2 as cv
from flask_cors import CORS
from skimage.io import imread
import tensorflow as tf
import os
import base64
import time
import pickle
from skimage.transform import resize

app = Flask(__name__)
cors = CORS(app)


IMAGE_SIZE = 224

def correlation(imageCompare_path, imageCompare, imageCrop_path):
    
        basewidth = 500 # Redimensionando imagem
        img = Image.open(imageCompare_path)
        wpercent = (basewidth/float(img.size[0]))
        hsize = int((float(img.size[1])*float(wpercent)))
        img=img.resize((basewidth, hsize), Image.ANTIALIAS)
        img.save(imageCompare.filename)

        imageCompareGray = cv.imread(imageCompare.filename, 0) # Imagem em tons de cinza (cv2.IMREAD_GRAYSCALE)
        imageCompareColorful = cv.imread(imageCompare.filename) # Imagem Colorida
        template = cv.imread('crop2.png', 0)
        w, h = template.shape[::-1]

        # faz a correlação cruzada
        res = cv.matchTemplate(imageCompareGray, template, eval('cv.TM_CCOEFF_NORMED'))

        # extrai algumas variáveis e coordenadas para montar o retângulo da área de detecção
        _, _, _, maxLoc = cv.minMaxLoc(res)
        top_left = maxLoc
        bottom_right = (top_left[0] + w, top_left[1] + h)

        # constrói o retângulo na primeira imagem lida
        cv.rectangle(imageCompareColorful,top_left, bottom_right, (0, 0, 255), 2)

        # Realiza o corte na imagem na área de interesse encontrada por meio da correlação cruzada
        cropimage = imageCompareColorful[maxLoc[1]:maxLoc[1]+h, maxLoc[0]:maxLoc[0]+w]
        cv.imwrite("crop.jpg", cropimage)
        cv.imshow("Image", imageCompareColorful)
        cv.imwrite("resultado.jpg", imageCompareColorful)

        with open("resultado.jpg", "rb") as image_file:
         encoded_string = base64.b64encode(image_file.read()).decode("utf-8")

        return encoded_string

# Realiza o pré-processamento (redimensionamento e normalização) na imagem que será testada
def preprocess_image(image):
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [IMAGE_SIZE, IMAGE_SIZE])
    image /= 255.0

    return image

def load_and_preprocess_image(path):
    image = tf.io.read_file(path)
    return preprocess_image(image)


# Realiza a classificação para a CNN
def classify(model, image_path, type):
    preprocessed_imgage = load_and_preprocess_image(image_path)
    preprocessed_imgage = tf.reshape(
        preprocessed_imgage, (1, IMAGE_SIZE, IMAGE_SIZE, 3)
    )

    inicio = time.time()
    prob = model.predict(preprocessed_imgage)
    aux_prob = 0
    label = ''

    # Retorna as métricas de classificação para o Resnet de acordo com o tipo de classificação
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
    classified_prob  = '{:.2f}'.format(aux_prob*100)

    return label, classified_prob, execution_time


def cnn_classify(image_path):
        models = ['50', '50V2', '101', '101V2', '152', '152V2'] # Lista de todos os modelos CNN (ResNet) gerados 
        folder = 'models/resNet'
        binary = [] 
        degrees = []

        # Realiza a classificação para os seis modelos gerados
        for i in range(6):
            source = folder + models[i]
            for model in os.listdir(source):
                print(source, model)
                cnn_model = tf.keras.models.load_model(source +'/' +model) # Carrega o modelo
                type = model.split('.')[0]
                label, classified_prob, execution_time = classify(cnn_model, image_path, type) # Realiza a classificação para a CNN
                # Gera a lista de resultados que será retornada para o front-end
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
                
        return binary, degrees

# Retorna as métricas de classificação para o XGBoost
def XGBoostClassify(model_path, Categories, image):
     model=pickle.load(open(f'{model_path}','rb'))
     inicio = time.time()
     probability=model.predict_proba(image.reshape(1,-1))
    
     label = ''
     max_prob = 0
     for ind,val in enumerate(Categories):
        prob = probability[0][ind]*100
        if prob > max_prob:
            max_prob = prob
            label = val
     fim = time.time()
     execution_time = round(fim-inicio,2)
     
     prob = '{:.2f}'.format(max_prob)

     return label, prob, execution_time


def XGboost(image_path):

    img=imread(image_path)

    img_resize=resize(img,(150,150,3))
    image=img_resize.flatten() 

    xgboost_list = []
    random_list = []
    # caminho do modelo
    CategoriesG =['grau 0','grau 1','grau 2','grau 3','grau 4']
    CategoriesB = ['artrose', 'sem artrose']

    #classificação por grau com xgboost 
    label, max_prob, execution_time = XGBoostClassify('models/xgboost_model_best.p', CategoriesG, image)
    xgboost_list.append({'type': "degrees", 'label': label, 'prob': max_prob, 'time': execution_time})

    #classificação binária com xgboost
    label, max_prob, execution_time = XGBoostClassify('models/xgboost_model_best.p', CategoriesB, image)
    xgboost_list.append({'type': "binary",'label': label, 'prob': max_prob, 'time': execution_time})

    #classificação por grau com randomForest
    label, max_prob, execution_time = XGBoostClassify('models/randomForest.p', CategoriesG, image)
    random_list.append({'type': "degrees", 'label': label, 'prob': max_prob, 'time': execution_time})

     #classificação binária com xgboost
    label, max_prob, execution_time = XGBoostClassify('models/randomForestB.p', CategoriesB, image)
    random_list.append({'type': "binary",'label': label, 'prob': max_prob, 'time': execution_time})

    return  xgboost_list, random_list

# Rota que retorna todas as classificações para o front-end
@app.route("/", methods=["POST"])
def home():
        imageCompare = request.files["imageFiles"] # Imagem que deseja buscar o corte
        imageCrop = request.files["imageCrop"] # Corte de uma imagem

        # Salva as imagens
        imageCompare_path = os.path.join(imageCompare.filename)
        imageCompare.save(imageCompare_path)

        # Realiza a equalização da imagem para melhorar a classificação
        image_equalized_path = 'equalized.png'
        imge = cv.imread(imageCompare_path,0)
        equ = cv.equalizeHist(imge)
        cv.imwrite(image_equalized_path,equ)

        imageCrop_path = os.path.join(imageCrop.filename)
        imageCrop.save(imageCrop_path)

        binary, degrees = cnn_classify(image_equalized_path) # Classificação Resnet
        # Aplicação da correlação cruzada e corte de região de interesse para melhor a classificação
        # do XGBooost e do Random Forest
        correlation_image = correlation(image_equalized_path, imageCompare, imageCrop_path)
        xgboost_list, random_list = XGboost(image_equalized_path) # Classificação XGBoost e Random Forest

        # Montagem do response contendo todas as classificações que serão retornadas para o front-end
        response = {
                'correlation' : correlation_image,
                'classifications' : {
                        'cnn': {
                            'binary': binary,
                            'degrees': degrees
                        },
                        'randomForest': random_list,
                        'xgboost': xgboost_list
                }
        }

        return jsonify(response)

app.run(debug=True)
from os import path, listdir, makedirs
from imagens import renomeiaESalva
import tensorflow as tf
import keras

STATIC_FOLDER = "static"
IMAGE_SIZE = 224

if not path.isdir('grau_artrose'):
        makedirs("grau_artrose")
        renomeiaESalva()

all_image_path = [path.join('grau_artrose', p) for p in listdir('grau_artrose') if path.isfile(path.join('grau_artrose', p))]

# Realiza o pré-processamento (redimensionamento e normalização) em todas as imagens
def load_and_preprocess_image(path):
    file = tf.io.read_file(path)
    image = tf.image.decode_jpeg(file , channels=3)
    image = tf.image.resize(image, [224, 224])
    image /= 255.0
    image = 2*image-1
    return image

# Realiza a categorização das imagens da base de acordo com o que sera classificaso
def categorize_images(x):

    all_images = [load_and_preprocess_image(path) for path in all_image_path]

    if x:  
        dict = {'0': 0, '1': 1, '2': 1, '3':1, '4':1} # Classificação Binária (com ou sem artrose)
    else: 
        dict = {'0': 0, '1': 1, '2': 2, '3':3, '4':4} # Classificação por grau de artrose

    labels = [path.split('.')[0][-1:] for path in all_image_path] 
    all_image_labels = [dict[label] for label in labels]

    ds = tf.data.Dataset.from_tensor_slices((all_images, all_image_labels))

    return ds, all_image_labels

# Criação do modelo de classificação baseado no ResNet
def generate_model(x):
    ds, all_image_labels = categorize_images(x)

    res_net = tf.keras.applications.ResNet50V2(input_shape=(224, 224, 3), include_top=False) # CNN que nos proporcionou melhores resultados
    res_net.trainable=False

    if x:
        range = 2
    else:
        range = 5

    # Construção do modelo cnn
    cnn_model = keras.models.Sequential([
        res_net,
        keras.layers.MaxPooling2D(),  
        keras.layers.Flatten(), 
        keras.layers.Dense(64, activation="relu"), # fully-connected hidden layer 
        keras.layers.Dense(range, activation="softmax") # output layer
    ])

    # Preparação dos dados para realizar o treinamento do modelo
    BATCH_SIZE = 32
    AUTOTUNE = tf.data.experimental.AUTOTUNE

    train_ds = ds.shuffle(buffer_size = len(all_image_labels))
    train_ds = train_ds.repeat()
    train_ds = train_ds.batch(BATCH_SIZE)
    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)

    # Etapa de treinamento do modelo 
    cnn_model.compile(optimizer=tf.keras.optimizers.Adam(),
                loss='sparse_categorical_crossentropy',
                metrics=["accuracy"])

    steps_per_epoch=tf.math.ceil(len(all_image_path)/BATCH_SIZE).numpy()
    cnn_model.fit(train_ds, epochs=12, steps_per_epoch=steps_per_epoch)

    # Salvando o modelo treinado
    if x:
        cnn_model.save('classify_binary.h5')
    else:
         cnn_model.save('classify.h5')

# Realiza o pré-processamento (redimensionamento e normalização) na imagem que será testada
def preprocess_image(image):
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [IMAGE_SIZE, IMAGE_SIZE])
    image /= 255.0

    return image


def load_and_preprocess_image(path):
    image = tf.io.read_file(path)
    return preprocess_image(image) 


# Realiza a classificação em uma imagem de teste
def classify(model, image_path):
    preprocessed_imgage = load_and_preprocess_image(image_path)
    preprocessed_imgage = tf.reshape(
        preprocessed_imgage, (1, IMAGE_SIZE, IMAGE_SIZE, 3)
    )

    prob = model.predict(preprocessed_imgage)
    print(prob[0][0])
    print("Probabilidade:  %5.2f" % (prob[0][0]))
    label = "sem artrose" if prob[0][0] >= 0.5 else "artrose"
    classified_prob = prob[0][0] if prob[0][0] >= 0.5 else 1 - prob[0][0]
    #os.remove(image_path)
    return label, classified_prob 

def main():
    generate_model(False) # Gera modelo de classificação por graus de artrose 
    generate_model(True) # Gera modelo de classificação binária (com ou sem artrose)

    # Teste unitário de classificação no modelo gerado anteriormente
    #cnn_model = tf.keras.models.load_model("classify.h5")
    #label, prob = classify(cnn_model, 'artrose_grau3 (3).png')
    #prob = round((prob * 100), 2)
    #print("Label: "+ label +"  Probabilidade:  %5.2f" % (prob))
    
    
if __name__ == "__main__":
    main()

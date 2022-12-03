from os import path, listdir, makedirs
import tensorflow as tf
import keras
STATIC_FOLDER = "static"


IMAGE_SIZE = 224

from imagens import renomeiaESalva

if not path.isdir('grau_artrose'):
        makedirs("grau_artrose")
        renomeiaESalva()

all_image_path = [path.join('grau_artrose', p) for p in listdir('grau_artrose') if path.isfile(path.join('grau_artrose', p))]

def load_and_preprocess_image(path):
    file = tf.io.read_file(path)
    image = tf.image.decode_jpeg(file , channels=3)
    image = tf.image.resize(image, [224, 224]) # resize all images to the same size.
    image /= 255.0  # normalize to [0,1] range
    image = 2*image-1  # normalize to [-1,1] range
    return image


def categorize_images(x):

    all_images = [load_and_preprocess_image(path) for path in all_image_path]


    sizeOfDemoList = len(all_images)

    print("The length of the list using the len() method is: " + str(sizeOfDemoList))

    if x:  
        dict = {'0': 0, '1': 1, '2': 1, '3':1, '4':1} 
    else: 
        dict = {'0': 0, '1': 1, '2': 2, '3':3, '4':4}

    # path.split('.')[0][-3:] return the name of the image ('dog' or 'cat')
    labels = [path.split('.')[0][-1:] for path in all_image_path] 
    all_image_labels = [dict[label] for label in labels]

    ds = tf.data.Dataset.from_tensor_slices((all_images, all_image_labels))

    return ds, all_image_labels


def generate_model(x):
    ds, all_image_labels = categorize_images(x)

    res_net = tf.keras.applications.ResNet50V2(input_shape=(224, 224, 3), include_top=False)
    res_net.trainable=False # this told the model not to train the res_net.

    if x:
        range = 2
    else:
        range = 5

    cnn_model = keras.models.Sequential([
        res_net, # res_net is low-level layers
        keras.layers.MaxPooling2D(),  
        keras.layers.Flatten(), 
        keras.layers.Dense(64, activation="relu"), # fully-connected hidden layer 
        keras.layers.Dense(range, activation="softmax") # output layer
    ])

    BATCH_SIZE = 32
    AUTOTUNE = tf.data.experimental.AUTOTUNE

    train_ds = ds.shuffle(buffer_size = len(all_image_labels))
    train_ds = train_ds.repeat()
    train_ds = train_ds.batch(BATCH_SIZE)
    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)

    cnn_model.compile(optimizer=tf.keras.optimizers.Adam(),
                loss='sparse_categorical_crossentropy',
                metrics=["accuracy"])

    steps_per_epoch=tf.math.ceil(len(all_image_path)/BATCH_SIZE).numpy()
    cnn_model.fit(train_ds, epochs=12, steps_per_epoch=steps_per_epoch)

    if x:
        cnn_model.save('classify_binary.h5')
    else:
         cnn_model.save('classify.h5')


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
    generate_model(False)
    generate_model(True)
    #cnn_model = tf.keras.models.load_model("classify.h5")
    #label, prob = classify(cnn_model, 'artrose_grau3 (3).png')
    #prob = round((prob * 100), 2)

    #print("Label: "+ label +"  Probabilidade:  %5.2f" % (prob))
    
    
if __name__ == "__main__":
    main()

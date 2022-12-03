import numpy as np 
import glob
import cv2
from sklearn.model_selection import train_test_split
import os
from sklearn import preprocessing
import xgboost as xgb
from sklearn.metrics import accuracy_score

SIZE = 256  #Resize images

train_images = []
train_labels = []

for directory_path in glob.glob("images/train/*"):
    label = directory_path.split("\\")[-1]
    for img_path in glob.glob(os.path.join(directory_path)):
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)   
        img = cv2.resize(img, (SIZE, SIZE))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        train_images.append(img)
        train_labels.append(label)

train_images = np.array(train_images)
train_labels = np.array(train_labels)

test_images = []
test_labels = [] 
for directory_path in glob.glob("images/test/*"):
    fruit_label = directory_path.split("\\")[-1]
    for img_path in glob.glob(os.path.join(directory_path)):
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (SIZE, SIZE))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        test_images.append(img)
        test_labels.append(fruit_label)


test_images = np.array(test_images)
test_labels = np.array(test_labels)


le = preprocessing.LabelEncoder()
le.fit(test_labels)
test_labels_encoded = le.transform(test_labels)
le.fit(train_labels)
train_labels_encoded = le.transform(train_labels)

X_train, X_test, y_train, y_test = train_test_split(train_images, train_labels_encoded, test_size=0.50, random_state=42)

model = xgb.XGBClassifier(max_depth=3, learning_rate=0.1, n_estimators=500, booster='gbtree')


X_train = X_train.reshape(X_train.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)

model = model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_train, y_pred)
print("Accuracy = ", (accuracy * 100.0), "%")
import pandas as pd
from sklearn import svm
from sklearn.model_selection import GridSearchCV
import os
import matplotlib.pyplot as plt
from skimage.transform import resize
from skimage.io import imread
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,accuracy_score,confusion_matrix
import pickle

def generate_model():
    Categories=['0','1','2','3','4']

    flat_data_arr=[]
    target_arr=[]

    datadir='xgboost/'
    for i in Categories:
        print(f'loading... category : {i}')
        path=os.path.join(datadir,i)
        for img in os.listdir(path):
            img_array=imread(os.path.join(path,img))
            img_resized=resize(img_array,(150,150,3))
            flat_data_arr.append(img_resized.flatten())
            target_arr.append(Categories.index(i))
        print(f'loaded category:{i} successfully')
    flat_data=np.array(flat_data_arr)
    target=np.array(target_arr)
    df=pd.DataFrame(flat_data)
    df['Target']=target
    df

    x=df.iloc[:,:-1]
    y=df.iloc[:,-1]
    x_train,x_test,y_train,y_test=train_test_split(x.values,y.values,test_size=0.20,random_state=77,stratify=y)
    print('Splitted Successfully')

    #from xgboost import RandomForestClassifier

    from sklearn.ensemble import RandomForestClassifier

    xgboost = RandomForestClassifier(n_estimators=40)
    xgboost.fit(x_train,y_train)

    y_pred_xgboost = xgboost.predict(x_test)
    print("The predicted Data from the xgboost is :")
    y_pred_xgboost

    print("The actual data is:")
    np.array(y_test)


    print(classification_report(y_pred_xgboost,y_test))
    print(f"The model is {accuracy_score(y_pred_xgboost,y_test)*100}% accurate")
    confusion_matrix(y_pred_xgboost,y_test)


    # change model_name as per requirement
    model_name = 'random_forest_model'
    # saving the trained model
    pickle.dump(xgboost, open(f'{model_name}.p','wb'))
    print("Pickle is dumped successfully")

generate_model()

# change the model name accordingly to load and predict the image
model_name = 'random_forest_model'

model=pickle.load(open(f'{model_name}.p','rb'))
Categories=['0','1','2','3','4']

# url=input('Enter URL of Image')
img=imread('download.png')

img_resize=resize(img,(150,150,3))
l=img_resize.flatten()  #img_resize.reshape(1,-1) #[img_resize.flatten()]
probability=model.predict_proba(l.reshape(1,-1))

for ind,val in enumerate(Categories):
    print(f'{val} = {probability[0][ind]*100}%')
print("The predicted image is : "+Categories[model.predict(l.reshape(1,-1))[0]])
from sklearn.metrics import classification_report,accuracy_score,confusion_matrix
from sklearn.model_selection import train_test_split
from skimage.transform import resize
from skimage.io import imread
import pandas as pd
import numpy as np
import pickle
import os

def generate_model():
    categories=['0','1']
    # categories=['0','1'] # Categorias para gerar o modelo de classificação binária
    data_arr=[]
    target_arr=[]
    datadir='binary/'

    # Carregamento de todas as imagens presentes no diretório especificado
    for i in categories:
        path=os.path.join(datadir, i)
        for img in os.listdir(path):
            img_array=imread(os.path.join(path, img))
            img_resized=resize(img_array,(150, 150, 3))
            data_arr.append(img_resized.flatten())
            target_arr.append(categories.index(i))
    # Montagem do dataframe que será utilizado na criação do modelo
    data=np.array(data_arr)
    target=np.array(target_arr)
    df=pd.DataFrame(data)
    df['Target']=target

    # Divisão os dados em dados de treinamento e teste
    x=df.iloc[:,:-1]
    y=df.iloc[:,-1]
    x_train,x_test,y_train,y_test=train_test_split(x.values,y.values,test_size=0.20,random_state=77,stratify=y)

    # Criação do modelo de classificação baseado no XGBoost
    from xgboost import XGBClassifier

    xgboost = XGBClassifier(max_depth=2, n_estimators=40) # Parâmetros utilizados que nos proporcionaram melhores resultados
    xgboost.fit(x_train, y_train)

    # Resultados obtidos após o treinamento do modelo 
    y_pred_xgboost = xgboost.predict(x_test)
    print("Os dados previstos pelo XGBoost são :")
    y_pred_xgboost

    np.array(y_test)

    print(classification_report(y_pred_xgboost, y_test))
    print(f"O modelo possui {accuracy_score(y_pred_xgboost, y_test)*100}% de acurácia")
    confusion_matrix(y_pred_xgboost, y_test)

    # Salvando o modelo treinado
    model_name = 'xgboost_model_binary'
    pickle.dump(xgboost, open(f'{model_name}.p','wb'))

generate_model()

# Teste unitário de classificação no modelo gerado anteriormente

model_name = 'xgboost_model_binary'
model=pickle.load(open(f'{model_name}.p','rb'))

categories=['0','1']
# categories=['0','1'] # Categorias para realizar a classificação binária

# Tratamento da imagem que será testada 
img=imread('equalized_crop.png')
img_resize=resize(img,(150, 150, 3))
l=img_resize.flatten()
probability=model.predict_proba(l.reshape(1,-1))

# Resultados obtidos após o teste de classificação
for ind,val in enumerate(categories):
    print(f'{val} = {probability[0][ind]*100}%')
print("A imagem pertence a categoria: "+categories[model.predict(l.reshape(1,-1))[0]])
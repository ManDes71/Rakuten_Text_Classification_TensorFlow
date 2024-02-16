import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import itertools

import pickle
from joblib import dump,load
import tensorflow as tf
import configparser
#from langdetect import detect_langs
import json
from tensorflow.keras.preprocessing.text import tokenizer_from_json


#def get_pickle_version():
#    return configparser.__v



config = configparser.ConfigParser()
config.read('/content/Rakuten_Text_Classification_TensorFlow/Rakuten_config_colab.ini')

def get_RACINE_DOSSIER() :
    return config['DOSSIER']['RACINE_DOSSIER']
    
def get_RACINE_IMAGES() :
    return config['DOSSIER']['RACINE_IMAGES']    
    
def get_RACINE_SAUVEGARDE() :
    return config['DOSSIER']['RACINE_SAUVEGARDE']



DATAFRAME_X = get_RACINE_DOSSIER() + 'X_train_update.csv'
DATAFRAME_X_TEST = get_RACINE_DOSSIER() + 'X_test_update.csv'
DATAFRAME_Y = get_RACINE_DOSSIER() + 'Y_train_CVw08PX.csv'
DATAFRAME_LANGUE = get_RACINE_DOSSIER() + 'df_langue.csv'
DATAFRAME_NOMENCLATURE = get_RACINE_DOSSIER() + 'NOMENCLATURE.csv'
DATAFRAME_STOPWORDS = get_RACINE_DOSSIER() + 'stopwords_FR_02.csv'
#    ***********    DEL    *****************
#RACINE_DOSSIER = 'C:\\Users\\DESPLANCHES.DOMAMP\\Datascientest\\'
#RACINE_IMAGES = 'C:\\Users\\DESPLANCHES.DOMAMP\\Datascientest\\images\\images\\'
#RACINE_SAUVEGARDE = 'C:\\Users\\DESPLANCHES.DOMAMP\\Datascientest\\fichiers\\'

#    ***********    SHADOW   *****************
#RACINE_DOSSIER = 'C:\\Users\\Shadow\\anaconda3\\envs\\tf\\PROJET\\'
#RACINE_IMAGES = 'C:\\Users\\Shadow\\anaconda3\\envs\\tf\\PROJET\\images\\'
#RACINE_SAUVEGARDE = 'C:\\Users\\Shadow\\anaconda3\\envs\\tf\\PROJET\\fichiers\\'

#    ***********    PC SIMON    *****************
#RACINE_DOSSIER = 'E:\\Manuel\\PROJET\\'
#RACINE_IMAGES = 'E:\\Manuel\\PROJET\\images\\'
#RACINE_SAUVEGARDE = 'E:\\Manuel\\PROJET\\fichiers\\'

print("section : ",config.sections())
DOSSIER_IMAGES_TRAIN = get_RACINE_IMAGES() + 'image_train'
DOSSIER_IMAGES_TEST = get_RACINE_IMAGES() +  'image_test'

REPORT_40_ACC = get_RACINE_DOSSIER() + 'df_report_accuracy_40.csv'
REPORT_40_VALACC = get_RACINE_DOSSIER() + 'df_report_val_acc_40.csv'
REPORT_40_PRED = get_RACINE_DOSSIER() + 'df_report_predict_40.csv'


import tensorflow as tf
from tensorflow.keras.utils import to_categorical

        

def save_ndarray(Xarray,name_sav) :
    with open(get_RACINE_SAUVEGARDE()+ name_sav+ '.pkl', 'wb') as f:
        pickle.dump(Xarray, f)
        
def load_ndarray(name_sav) :
    with open(get_RACINE_SAUVEGARDE() + name_sav+ '.pkl', 'rb') as f:
        Xarray = pickle.load(f)
    return   Xarray  
def save_dataframe(df,name_sav) :
    df.to_csv(get_RACINE_DOSSIER() + name_sav)
def load_dataframe(name_sav) :
    df = pd.read_csv(get_RACINE_DOSSIER() + name_sav)    
    return df
def save_model(model,name_sav) :
    print(get_RACINE_DOSSIER() + name_sav+'.h5')
    model.save_weights(get_RACINE_DOSSIER() + name_sav+'.h5')
def load_model(model,name_sav) :
    model.load_weights(get_RACINE_DOSSIER() + name_sav+'.h5')    
    
    
def joblib_dump(model,name_sav) :
    dump(model,get_RACINE_DOSSIER() + name_sav+'.joblib')
def joblib_load(name_sav) :
    return load(get_RACINE_DOSSIER() + name_sav+'.joblib')        
    
def save_dataset(dataset,name_sav) :
    dataset.save(get_RACINE_SAUVEGARDE()+name_sav)  
    
def save_tokenizer(tokenizer,name_sav) :
    tokenizer_json = tokenizer.to_json()
    with open(get_RACINE_SAUVEGARDE() + name_sav+ '.json', 'w', encoding='utf-8') as f:
        f.write(json.dumps(tokenizer_json, ensure_ascii=False))

def load_tokenizer(name_sav) :
    with open(get_RACINE_SAUVEGARDE() + name_sav+ '.json', 'r', encoding='utf-8') as f:
        json_data = json.load(f)
        tokenizer = tokenizer_from_json(json_data)    
    return tokenizer
  
    
def detection_langue(texte) :
    x=detect_langs(texte)   
    print("x=",x)
    #print("x[0]=",str(x[0]).split(':'))
    return str(x[0]).split(':')[0]
        
              
def plot_fit(train_acc,val_acc,tloss,tvalloss) :
    plt.figure(figsize=(12,4))
    plt.subplot(121)
    plt.plot(tloss)
    plt.plot(tvalloss)
    plt.title('Model loss by epoch')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='right')

    plt.subplot(122)
    plt.plot(train_acc)
    plt.plot(val_acc)
    plt.title('Model acc by epoch')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='right')
    plt.show()
 
def get_def_prediction(y_orig, y_pred,cat):
    top5_df = pd.DataFrame({'prdtypecode': y_orig,'predict': y_pred})
      
    df_cross=pd.crosstab(top5_df['prdtypecode'], top5_df['predict'],normalize='index')
        
    df_pred = pd.DataFrame()
    for c in cat:
        s = df_cross.loc[c].sort_values(ascending=False)[:5]
        df_temp = pd.DataFrame([{'Categorie':c,'predict':s.index[0],'pourc':s.values[0],'predict2':s.index[1],'pourc2':s.values[1],'predict3':s.index[2],'pourc3':s.values[2]}])
        df_pred = pd.concat([df_pred, df_temp], ignore_index=True)
        
    return df_pred
def get_df_crosstab(y_orig, y_pred):   
     top5_df = pd.DataFrame({'prdtypecode': y_orig,'predict': y_pred})
     df_cross=pd.crosstab(top5_df['prdtypecode'], top5_df['predict'],normalize='index')
     return df_cross
     
def Afficher_repartition(df_cross,cat,catdict):
        for c in cat:
            print(c,'   ------   ', catdict[c] )    
            s=df_cross.loc[c].sort_values(ascending=False)[:5]
            for index, value in s.iteritems():
                print(f"  : {index},  : {np.round(value*100,2)} % , {catdict[index]}")    
    
def get_classification_report(y_orig, y_pred):
        
    acc_score = accuracy_score(y_orig, y_pred)*100
    classif = classification_report(y_orig, y_pred)
        
    print("Précision de la prédiction:", acc_score, '%')
    print("Evaluation détaillée de la Classification par RDF :\n \n" ,(classif))
            
    return acc_score,classif  

def show_confusion_matrix(y_orig, y_pred):
    
    
    cnf_matrix = confusion_matrix(y_orig, y_pred,labels=sorted(list(set(y_orig))))
    
    #classes = [10,2280,2403,2705,40,50,2462,1280,1281]
    classes=sorted(list(set(y_orig)))
    b=list(set(y_orig))

    plt.figure(figsize=(15,15))

    plt.imshow(cnf_matrix, interpolation='nearest',cmap='Blues')
    plt.title("Matrice de confusion")
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes,rotation=90)
    plt.yticks(tick_marks, classes)

    for i, j in itertools.product(range(cnf_matrix.shape[0]), range(cnf_matrix.shape[1])):
        plt.text(j, i, cnf_matrix[i, j],
                 horizontalalignment = "center",
                 color = "white" if cnf_matrix[i, j] > ( cnf_matrix.max() / 2) else "black")

    plt.ylabel('Vrais labels')
    plt.xlabel('Labels prédits')
    plt.show()
    


 
def ajout_REPORT_40_VALACC(val_acc,MODELE,LIBBELLE,ID):
        
    if os.path.exists(REPORT_40_VALACC):
        bool_valacc = True
    else:
        bool_valacc = False
    
    if bool_valacc :
        df_val_accuracy= pd.read_csv(REPORT_40_VALACC)
    
    df_val_accuracy2 = pd.DataFrame(list(val_acc)[:10],index=np.arange(1 , 11, 1),columns =['val_accuracy'])
    #print(df_val_accuracy2.head())
    #print('*******************')
    df_val_accuracy2['modele']=MODELE.strip()
    df_val_accuracy2['libelle']=LIBBELLE.strip()
    df_val_accuracy2['id']= ID.strip()
    df_val_accuracy2.reset_index(inplace=True)
    df_val_accuracy2 = df_val_accuracy2.rename(columns={"index": "Epoch"})    

    if bool_valacc :
        df_val_accuracy=df_val_accuracy[df_val_accuracy['id'] != ID]
        df_val_accuracy = pd.concat([df_val_accuracy,df_val_accuracy2])
    else :
        df_val_accuracy=df_val_accuracy2.copy()
    
    #print(df_val_accuracy.head())
    df_val_accuracy.to_csv(REPORT_40_VALACC)    
    
def ajout_REPORT_40_ACC(train_acc,MODELE,LIBBELLE,ID):
        
    if os.path.exists(REPORT_40_ACC):
        bool_acc = True
    else:
        bool_acc = False
    
    if bool_acc :
        df_val_accuracy= pd.read_csv(REPORT_40_ACC)
    
    df_val_accuracy2 = pd.DataFrame(list(train_acc)[:10],index=np.arange(1 , 11, 1),columns =['val_accuracy'])
    #print(df_val_accuracy2.head())
    #print('*******************')
    df_val_accuracy2['modele']=MODELE
    df_val_accuracy2['libelle']=LIBBELLE
    df_val_accuracy2['id']=ID
    df_val_accuracy2.reset_index(inplace=True)
    df_val_accuracy2 = df_val_accuracy2.rename(columns={"index": "Epoch"})    

    if bool_acc :
        df_val_accuracy=df_val_accuracy[df_val_accuracy['id'] !=ID]
        df_val_accuracy = pd.concat([df_val_accuracy,df_val_accuracy2])
    else :
        df_val_accuracy=df_val_accuracy2.copy()
    
    df_val_accuracy.to_csv(REPORT_40_ACC)    

def ajout_REPORT_40_PRED(y_orig, y_pred,MODELE,LIBBELLE,ID):    
    if os.path.exists(REPORT_40_PRED):
        bool_pred = True
    else:
        bool_pred = False
    if bool_pred :
        df_report= pd.read_csv(REPORT_40_PRED)
    
 
    report_dict = classification_report(y_orig, y_pred, output_dict=True)

    # Conversion du rapport de classification en DataFrame
    df_report2 = pd.DataFrame(report_dict).transpose()
    df_report2['programme']=MODELE
    df_report2['libelle']=LIBBELLE
    df_report2['id']=ID
    df_report2.reset_index(inplace=True)
    df_report2 = df_report2.rename(columns={"index": "Categorie"})
    if bool_pred :
        df_report=df_report[df_report['id'] != ID]
        df_report = pd.concat([df_report,df_report2])
        df_report.drop('Unnamed: 0',axis=1,inplace=True)
    else:    
        df_report=df_report2.copy()
    df_report.to_csv(REPORT_40_PRED)  

class DS_Model:
    
     def __init__(self, nom_modele):
            
        self._STRATEGIE = "sampling_strategy_INCEPTION_V03.csv"
        self._IMGSIZE       = 400    # Taille de l'image en input
        self._EPOCH         = 20     # nombre d'epoch 
        self._BATCH_SIZE    = 32     # traitement par batch d'images avant la descente de gradient
        self._FREEZE_LAYERS = 305    # Nb couches freezées pour les modèles pré-entrainés
        self._NB_PAR_LABEL_MAX = 5000 # nombre maximun d'enregistrement par label = undersampling
        self._NB_PAR_LABEL_MIN = 2000 # nombre minimum d'enregistrement par label = undersampling
        self._TRAIN         = True   # Entrainement ou utilisation d'un réseau déjà entrainé
       
        
        self.__nom_modele = nom_modele
        self.__df = pd.DataFrame()
        self.__df_test = pd.DataFrame()
        self.__stopwordFR = pd.DataFrame()
        self.__catdict = {}
        self.__cat = []
        self.train_filter = np.array([], dtype=bool)
        self.test_filter = np.array([], dtype=bool)     
        
        
        self.__chargement_fichiers()
        
     def __chargement_fichiers(self):
        
        #config = configparser.ConfigParser()
        #config.read('Rakuten_config_2.ini')   # Simon
        #config.read('Rakuten_config_3.ini')  # dell
        #config.read('Rakuten_config_2.ini')  # shadow
        
        #RACINE_DOSSIER = config['DOSSIER']['RACINE_DOSSIER']
        #RACINE_IMAGES = config['DOSSIER']['RACINE_IMAGES']
        #RACINE_SAUVEGARDE = config['DOSSIER']['RACINE_SAUVEGARDE']


        df_feats=pd.read_csv(DATAFRAME_X)
        df_target=pd.read_csv(DATAFRAME_Y)
        df_test=pd.read_csv(DATAFRAME_X_TEST)
        
        df_langue=pd.read_csv(DATAFRAME_LANGUE)
        df_langue.drop(['Id','prdtypecode','design_long','descrip_long'],axis=1,inplace=True)
        #print(df_langue.info())
        df_feats=pd.merge(df_feats,df_langue,on='Unnamed: 0',how='outer')
        #print(df_feats.info())
        
        self.__df=df_feats.merge(df_target,on='Unnamed: 0',how='inner')
        self.__df.rename(columns={'Unnamed: 0': 'Id'}, inplace=True)
        
        self.__df_test=df_test
        
        self.__cat=df_target['prdtypecode'].sort_values().unique()
        nomenclature=pd.read_csv(DATAFRAME_NOMENCLATURE,header=0,encoding='utf-8',sep=';',index_col=0)
        self.__catdict=nomenclature.to_dict()['definition']
        
        self.__df['nom_image']=self.__df.apply(lambda row: "image_" +  str(row['imageid']) 
                                     + "_product_" + str(row['productid']) + ".jpg",axis=1)
        
        folder_path = DOSSIER_IMAGES_TRAIN
        self.__df['filepath']=self.__df['nom_image'].apply(lambda x : os.path.join(folder_path, x))
        
        self.__stopwordFR = pd.read_csv(DATAFRAME_STOPWORDS)
        
     def get_DF(self):
        return self.__df
    
     def get_DF_TEST(self):
       return self.__df_test
   
     def get_DF_TEST_DESCRIPTION(self,nom_image):
          
         
        print(nom_image)    
        nom_image_sans_jpg = nom_image.split('.')[0]   
        print(nom_image_sans_jpg) 
        # Diviser la chaîne en utilisant le caractère de soulignement ('_') comme séparateur
        parties = nom_image_sans_jpg.split('_')

        numero_image = int(parties[1])
        numero_produit = int(parties[3])
        
        print(numero_image,numero_produit)

        subset = self.__df_test[(self.__df_test.imageid == numero_image) & (self.__df_test.productid == numero_produit)][['designation','description'] ].iloc[0]    
        designation = subset[0]
        description = subset[1]
        return designation,description

        
     def restore_fit_arrays(self):
        train_acc = load_ndarray(self.__nom_modele+'_accuracy')
        val_acc = load_ndarray(self.__nom_modele+'_val_accuracy')
        tloss = load_ndarray(self.__nom_modele+'_loss')
        tvalloss = load_ndarray(self.__nom_modele+'_val_loss')
        
        return train_acc,val_acc,tloss,tvalloss
    
     def restore_predict_arrays(self):
        y_orig = load_ndarray(self.__nom_modele+'_y_orig')
        y_pred = load_ndarray(self.__nom_modele+'_y_pred')   

        return  y_orig,y_pred 
        
     def restore_predict_dataframe(self):    
        df_predict = load_dataframe(self.__nom_modele+'_df_predict')        
        return df_predict    
        
        
     def save_filtres(self):
        np.save('train_filter.npy', self.train_filter.to_numpy())
        np.save('test_filter.npy', self.test_filter.to_numpy())
     
     def restore_filtres(self):
        self.train_filter_loaded = np.load('train_filter.npy')
        self.test_filter_loaded = np.load('test_filter.npy')
       
     def set_BATCH_SIZE(self,batchsize): 
        self._BATCH_SIZE = batchsize
        
     def get_cat(self):
        return self.__cat
        
     def get_catdict(self):
        return self.__catdict
        
     def get_stopwordFR(self):
        return self.__stopwordFR
        
        
     def get_Serie_prdtypecode(self):
        return pd.Series(self.__df.prdtypecode).value_counts()
    
     
     def Train_Test_Split_(self,train_size=0.8, random_state=1234): 
                           
        X = self.__df.drop('prdtypecode', axis=1)
        y = self.__df['prdtypecode']                   
    
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size,
                                    random_state=random_state, stratify=y,shuffle=True)
                                    
        self.train_filter = self.__df.index.isin(X_train.index)
        self.test_filter = self.__df.index.isin(X_test.index)                            
                                    
        return X_train, X_test, y_train, y_test
     


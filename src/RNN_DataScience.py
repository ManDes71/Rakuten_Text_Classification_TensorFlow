import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder
from keras.initializers import Constant

from nltk.tokenize import PunktSentenceTokenizer
from nltk.tokenize.regexp import RegexpTokenizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from  nltk.stem.snowball import FrenchStemmer,EnglishStemmer,GermanStemmer,ItalianStemmer,DutchStemmer,SpanishStemmer,ItalianStemmer


from tensorflow.keras import Sequential,Input, Model
from tensorflow.keras.layers import Embedding, Dense, GlobalAveragePooling1D, RNN, GRUCell
from tensorflow.keras.layers import  Dropout ,Conv1D,Flatten,Bidirectional,LSTM,BatchNormalization

from keras.optimizers import Adam
from tensorflow.keras import callbacks
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC,LinearSVC
from sklearn.metrics import f1_score

import re
import nltk
import pickle
import unicodedata
from nltk.corpus import stopwords
import swifter
import spacy



import tensorflow as tf
from tensorflow.keras.utils import to_categorical

import Bibli_DataScience_3 as ds

def unicode_to_ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn')

def stemming(mots,porter_stemmer) :
    sortie = []
    for string in mots :
        radical = porter_stemmer.stem(string)
        if (radical not in sortie) : sortie.append(radical)
    return sortie
    
def Stemmer_sentence(sentence,langue):
    if langue == 'en':
        porter_stemmer = EnglishStemmer()
    elif langue == 'fr':
        porter_stemmer = FrenchStemmer()
    elif langue == 'de':
        porter_stemmer = GermanStemmer()
    elif langue == 'ca':
        porter_stemmer = FrenchStemmer()
    elif langue == 'nl':
        porter_stemmer = DutchStemmer()
    elif langue == 'it':
        porter_stemmer = ItalianStemmer()
    elif langue == 'es':
        porter_stemmer = SpanishStemmer()
    else:
        porter_stemmer = FrenchStemmer()
    
    # Pour chaque mot de la phrase (dans l'ordre inverse)
    sentence = stemming(sentence,porter_stemmer)
    return sentence
    


    
class DS_RNN(ds.DS_Model):
    
     def __init__(self, nom_modele):
     
        super().__init__(nom_modele)
            
        
        self.__nom_modele = nom_modele
        self.__model = Sequential()
        
        self.__df_pred = pd.DataFrame()
        self.__df_feats = pd.DataFrame()
        self.__df_target = pd.DataFrame()
        self.__y_orig =[]
        self.__y_pred = []
        self.__X = np.array([])
        self.__y = np.array([])
        self.__report_ID = "RNN"
        self.__report_MODELE = nom_modele
        self.__report_LIBBELLE = "RNN"
        self.recuperer_dataframes()
        self.NUM_WORDS = 70000  # 70000
        self.MAXLEN= 600
        self.EMBEDDING_DIM = 300
       
        nltk.download('punkt')
        
        self.nlp_fr = spacy.load('fr_core_news_md')
        self.nlp_en = spacy.load('en_core_web_md')
        self.nlp_de = spacy.load('de_core_news_md')
        self.nlp_es = spacy.load('es_core_news_md')
        self.nlp_it = spacy.load('it_core_news_md')
        self.nlp_nl = spacy.load('nl_core_news_md')
        self.nlp_ca = spacy.load('ca_core_news_md')
      
     def get_df_feats(self):
        return self.__df_feats
        
     def get_df_target(self):
        return self.__df_target

     def get_df_pred(self):
        return self.__df_pred

     def set_df_pred(self,pred):
        self.__df_pred = pred
        
     def set_y_orig(self,orig):
        self.__y_orig = orig 

     def set_y_pred(self,pred):
        self.__y_pred = pred        
        
     def get_y_orig(self) :
        return self.__y_orig
        
     def get_y_pred(self) :
        return self.__y_pred
        
     def get_model(self) :
        return self.__model
     def set_model(self,model):
        self.__model = model         
        
     def get_REPORT_ID(self) :
        return self.__report_ID
     def set_REPORT_ID(self,id):
        self.__report_ID = id           
     def get_REPORT_MODELE(self) :
        return self.__report_MODELE
     def set_REPORT_MODELE(self,modele):
        self.__report_MODELE = modele          
     def get_REPORT_LIBELLE(self) :
        return self.__report_LIBBELLE  
     def set_REPORT_LIBELLE(self,libelle):
        self.__report_LIBBELLE = libelle      

     def get_vector(self ,word, lang):
        if lang == 'fr':
            return self.nlp_fr(word).vector
        elif lang == 'en':
            return self.nlp_en(word).vector
        elif lang == 'de':
            return self.nlp_de(word).vector
        elif lang == 'ca':
            return self.nlp_ca(word).vector
        elif lang == 'nl':
            return self.nlp_nl(word).vector
        elif lang == 'it':
            return self.nlp_it(word).vector  
        elif lang == 'es':
            return self.nlp_es(word).vector              
        else :
            return self.nlp_fr(word).vector
       
                     
     def recuperer_dataframes(self):
        df = self.get_DF()
        self.__df_feats = df[['designation','description','productid','imageid']]
        self.__df_target = df['prdtypecode']
        
     def add_traitement(self,mots,pays_langue) :
        return mots
     
     def preprossessing_X(self,row):
        w = row['designation']
        pays_langue = row['PAYS_LANGUE']
        
        #print("w=",w)
        #print("pays_langue=",pays_langue)
     
        stop_words = self.get_stopwordFR()
        w = unicode_to_ascii(w.lower().strip())
        # creating a space between a word and the punctuation following it
        # eg: "he is a boy." => "he is a boy ."
        w = re.sub(r"([?.!,¿])", r" \1 ", w)
        w = re.sub(r'[" "]+', " ", w)
        # replacing everything with space except (a-z, A-Z,0-9, ".", "?", "!", ",","°")
        w = re.sub(r"[^a-zA-Z0-9âéè°]+", " ", w)
        w = re.sub(r'\b\w{0,2}\b', '', w)

        # remove stopword
        mots = word_tokenize(w.strip())
        mots = [mot for mot in mots if mot not in stop_words]
        mots = self.add_traitement(mots,pays_langue)
        #print("mot final 1 = ",mots)
        #print("mot final 2 = ",' '.join(mots).strip())
        #return ' '.join(mots).strip()
        return {'designation': ' '.join(mots).strip(), 'PAYS_LANGUE': pays_langue}
    
     def traiter_phrases(self,design,descrip):
        DESCRIP = []
        partie_design = design if type(design) == str else ''
        partie_descrip = descrip if type(descrip) == str else ''
        s = (partie_design + ' ' + partie_descrip) if len(partie_descrip) > 0 else partie_design
        row = pd.Series(s)
        print("s = ",s)
        langue=ds.detection_langue(s)
        print("langue = ",langue)
        chaine_designation = s
        chaine_pays_langue = langue[0]
        
        row = pd.Series({'designation': chaine_designation, 'PAYS_LANGUE': chaine_pays_langue})
        X_test = self. preprossessing_X(row)
        
        print("X_test 1 = ",X_test)
        X_test= pd.Series(X_test['designation'])
        print("X_test 11 = ",X_test)
        print("type = ",type(X_test))
        
        
        tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=self.NUM_WORDS)
        # Mettre à jour le dictionnaire du tokenizer
        tokenizer = ds.load_tokenizer(self.__nom_modele+'_tokenizer')
        X_train = ds.load_ndarray(self.__nom_modele+'_X_train')
        print("type((X_train['designation'])",type(X_train['designation']))
        tokenizer.fit_on_texts(X_train['designation'])
        
        #word2idx = tokenizer.word_index
        #idx2word = tokenizer.index_word
        #vocab_size = tokenizer.num_words
      
        X_test = tokenizer.texts_to_sequences(X_test)
        print("X_test 2 = ",X_test)
                
        X_test = tf.keras.preprocessing.sequence.pad_sequences(X_test, maxlen=self.MAXLEN, padding='post', truncating='post')
        print("X_test 2 = ",X_test)     
        
        return X_test   
        
     def preprossessing_Y(self,y_train,y_test):
          label_encoder = LabelEncoder()
          y_classes_converted = label_encoder.fit_transform(y_train)
          y_train_Network = to_categorical(y_classes_converted)
          y_classes_converted = label_encoder.transform(y_test)
          y_test_Network = to_categorical(y_classes_converted)
          print(y_train_Network.shape)
          print(y_test_Network.shape)
          #ds.save_ndarray(y_train_Network,self.__nom_modele+'_y_train')
          #ds.save_ndarray(y_test_Network,self.__nom_modele+'_y_test')
          #ds.save_ndarray(label_encoder,self.__nom_modele+'_label_encoder')
          return y_train_Network,y_test_Network,label_encoder   
        
     def Train_Test_Split_(self,train_size=0.8, random_state=1234, fic="None"):    
     
        if fic == "Load" :
            print("Récupération de jeu d'entrainement")
            X_train = ds.load_ndarray(self.__nom_modele+'_X_train')
            X_test = ds.load_ndarray(self.__nom_modele+'_X_test')
            y_train_avant = ds.load_ndarray(self.__nom_modele+'_y_train')
            y_test_avant = ds.load_ndarray(self.__nom_modele+'_y_test')
            #label_encoder = ds.load_ndarray(self.__nom_modele+'_label_encoder')
            print("load y_train_avant.shape ",y_train_avant.shape)
            
            return X_train, X_test, y_train_avant, y_test_avant
        
           
        X_train_avant, X_test_avant, y_train_avant, y_test_avant = super().Train_Test_Split_(train_size, random_state)
        
        DESCRIP_train = []
        for design, descrip in zip( X_train_avant['designation'],  X_train_avant['description']):
            partie_design = design if type(design) == str else ''
            partie_descrip = descrip if type(descrip) == str else ''
            s = (partie_design + ' ' + partie_descrip) if len(partie_descrip) > 0 else partie_design
            DESCRIP_train.append(s)
        
        DESCRIP_test = []
        for design, descrip in zip( X_test_avant['designation'],  X_test_avant['description']):
            partie_design = design if type(design) == str else ''
            partie_descrip = descrip if type(descrip) == str else ''
            s = (partie_design + ' ' + partie_descrip) if len(partie_descrip) > 0 else partie_design
            DESCRIP_test.append(s)
        
      
        X_train = pd.Series(DESCRIP_train)
        X_test = pd.Series(DESCRIP_test)
        #X_train = X_train_avant[['designation', 'PAYS_LANGUE']].apply(self.preprossessing_X, axis=1)
        #X_test = X_test_avant[['designation', 'PAYS_LANGUE']].apply(self.preprossessing_X, axis=1)
        
        X_train_processed_list = X_train_avant[['designation', 'PAYS_LANGUE']].apply(self.preprossessing_X, axis=1).tolist()
        X_train = pd.DataFrame(X_train_processed_list)
        X_test_processed_list = X_test_avant[['designation', 'PAYS_LANGUE']].apply(self.preprossessing_X, axis=1).tolist()
        X_test = pd.DataFrame(X_test_processed_list)
        
        if fic == "Save" :
            print("Sauvegarde de jeu d'entrainement")
            ds.save_ndarray(X_train,self.__nom_modele+'_X_train')
            ds.save_ndarray(X_test,self.__nom_modele+'_X_test')
            ds.save_ndarray(y_train_avant,self.__nom_modele+'_y_train')
            ds.save_ndarray(y_test_avant,self.__nom_modele+'_y_test')
        
        print("save y_train_avant.shape ",y_train_avant.shape)
        return X_train, X_test, y_train_avant, y_test_avant
    
     def fit_modele(self,epochs,savefics=False,freeze=0,Train="None",spacy=False):
    
        X_train,X_test,y_train,y_test = self.Train_Test_Split_(fic=Train)
        
       
        
        embedding_dict = {}
        embedding_matrix = []
        #print(X_train['PAYS_LANGUE'][:5])
        #print(X_train['designation'][:5])
        print("self.EMBEDDING_DIM",self.EMBEDDING_DIM)  
        if spacy:
            print("creation dictionnaire") 
            for lang, sentence in zip(X_train['PAYS_LANGUE'],X_train['designation']):  # Remplacez 'your_dataset' par votre propre ensemble de données
                #print(lang, sentence)
                for word in sentence.split():
                    #print(word)
                    if word not in embedding_dict:
                        embedding_dict[word] = self.get_vector(word, lang)
            print("longueur dictionnaire",len(embedding_dict))            
            print("creation matrice") 
            embedding_matrix = np.zeros((len(embedding_dict)+1, self.EMBEDDING_DIM))  # Remplacez 'EMBEDDING_DIM' par la dimension de votre embedding
            print("embedding_matrix.shape = " ,embedding_matrix.shape)
            for i, word in enumerate(embedding_dict.keys()):
                embedding_vector = embedding_dict.get(word)
                if embedding_vector is not None:
                    embedding_matrix[i] = embedding_vector            
        print("suite")
        #print("embedding_matrix.shape = " ,embedding_matrix.shape)
        X_train = X_train['designation']
        X_test = X_test['designation']
        
        y_train,y_test,label_encoder = self.preprossessing_Y(y_train,y_test)
        
        tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=self.NUM_WORDS)
        # Mettre à jour le dictionnaire du tokenizer
        tokenizer.fit_on_texts(X_train)
        
        word2idx = tokenizer.word_index
        idx2word = tokenizer.index_word
        vocab_size = tokenizer.num_words

        X_train = tokenizer.texts_to_sequences(X_train)
        X_test = tokenizer.texts_to_sequences(X_test)
        
        X_train = tf.keras.preprocessing.sequence.pad_sequences(X_train, maxlen=self.MAXLEN, padding='post', truncating='post')
        X_test = tf.keras.preprocessing.sequence.pad_sequences(X_test, maxlen=self.MAXLEN, padding='post', truncating='post')
               
        print( "input : ",len(embedding_dict)) 
        print( "len(embedding_dict) : ",len(embedding_dict))
        model = self.create_modele(embedding_matrix,len(embedding_dict)+1)    
        
        if Train == "Load" :
            ds.load_model(model,self.__nom_modele+'_weight')
            self.set_model(model)  
        
        lr_plateau = callbacks.ReduceLROnPlateau(monitor = 'val_loss',
                            patience=4,
                            factor=0.5,
                            verbose=1,
                            mode='min')
         
            
        if Train == 'Save' :
            print("Sauvegarde de jeu pour la concatenation")
            ds.save_ndarray(X_train,self.__nom_modele+'_CONCAT_X_train')
            ds.save_ndarray(X_test,self.__nom_modele+'_CONCAT_X_test')
            ds.save_ndarray(y_train,self.__nom_modele+'_CONCAT_y_train')
            ds.save_ndarray(y_test,self.__nom_modele+'_CONCAT_y_test')
        
        training_history = model.fit(X_train, y_train,batch_size = 32, epochs=epochs, validation_data = [X_test, y_test],callbacks=[lr_plateau])
        
        train_acc = training_history.history['accuracy']
        val_acc = training_history.history['val_accuracy']
        tloss = training_history.history['loss']
        tvalloss=training_history.history['val_loss']
          
          
        predictions = model.predict( X_test)
        feature_model_rnn = Model(inputs=model.input, outputs=model.layers[-2].output)
        train_features_rnn = feature_model_rnn.predict(X_train)
        test_features_rnn = feature_model_rnn.predict(X_test)
        
        
        
        if Train == 'Save' or Train == "Weight" :
            print("Sauvegarde de jeu pour la concatenation")
            ds.save_ndarray(train_features_rnn,self.__nom_modele+'_CONCAT2_X_train')
            ds.save_ndarray(test_features_rnn,self.__nom_modele+'_CONCAT2_X_test')
            ds.save_ndarray(y_train,self.__nom_modele+'_CONCAT2_y_train')
            ds.save_ndarray(y_test,self.__nom_modele+'_CONCAT2_y_test')
            
        if Train == "Save" :
            print("Sauvegarde de jeu du label encoder")
            ds.save_ndarray(label_encoder,self.__nom_modele+'_label_encoder')
            print("Sauvegarde du model")
            print(self.__nom_modele+'_weight')
            ds.save_model(model,self.__nom_modele+'_weight')   
            print("Sauvegarde du tokenizer")
            ds.save_tokenizer(tokenizer,self.__nom_modele+'_tokenizer')
        
        
        y_test_original = np.argmax(y_test, axis=1)
        y_test_original2=label_encoder.inverse_transform(y_test_original)
        y_pred = np.argmax(predictions, axis=1)
        test_pred_orinal2=label_encoder.inverse_transform(y_pred)
        
        print("y_test_original2[:5] ", y_test_original2[:5])
        
        
        top5_df = pd.DataFrame({'prdtypecode': y_test_original2,'predict': test_pred_orinal2})

        df_cross=pd.crosstab(top5_df['prdtypecode'], top5_df['predict'],normalize='index')
        
        self.__df_pred = pd.DataFrame()
        for c in self.get_cat():
            s = df_cross.loc[c].sort_values(ascending=False)[:5]
            df_temp = pd.DataFrame([{'Categorie':c,'predict':s.index[0],'pourc':s.values[0],'predict2':s.index[1],'pourc2':s.values[1],'predict3':s.index[2],'pourc3':s.values[2]}])
            self.__df_pred = pd.concat([self.__df_pred, df_temp], ignore_index=True)
        
        self.__y_orig = y_test_original2
        self.__y_pred = test_pred_orinal2
        
        if Train == "Save" :
            print("Sauvegarde de jeu du label encoder")
            ds.save_ndarray(label_encoder,self.__nom_modele+'_label_encoder')
            print(self.__nom_modele+'_weight')
            ds.save_model(model,self.__nom_modele+'_weight')
        
        if savefics :
            ds.save_ndarray(train_acc,self.__nom_modele+'_accuracy')
            ds.save_ndarray(val_acc,self.__nom_modele+'_val_accuracy')
            ds.save_ndarray(tloss,self.__nom_modele+'_loss')
            ds.save_ndarray(tvalloss,self.__nom_modele+'_val_loss')
            ds.save_ndarray(y_test_original2,self.__nom_modele+'_y_orig')
            ds.save_ndarray(test_pred_orinal2,self.__nom_modele+'_y_pred')
            ds.save_dataframe(self.__df_pred,self.__nom_modele+'_df_predict')
            
        
        return train_acc,val_acc,tloss,tvalloss
        
       
class RNN_EMBEDDING(DS_RNN):     

     def __init__(self, nom_modele):
        super().__init__(nom_modele)
            
        self.__nom_modele = nom_modele
        self.set_REPORT_ID("EMBED1")
        self.set_REPORT_MODELE(nom_modele)
        self.set_REPORT_LIBELLE("EMBEDDING")
        
     def create_modele(self,Matrix=None,len_embedding_dict=0):
        model = Sequential()
        model.add(Embedding(self.NUM_WORDS, self.EMBEDDING_DIM))
        model.add(Conv1D(filters=32, kernel_size=8, activation='relu'))
        model.add(GlobalAveragePooling1D())
        model.add(Flatten())
        model.add(Dense(256, activation='relu'))
        #model.add(BatchNormalization())
        model.add(Dropout(0.4))
        model.add(Dense(27, activation='softmax'))
        model.compile(optimizer=Adam(learning_rate=1e-3), loss='categorical_crossentropy', metrics=['accuracy'])    
        
        self.set_model(model)

        return model 
        
class RNN_EMBEDDING2(DS_RNN):     

     def __init__(self, nom_modele):
        super().__init__(nom_modele)
            
        self.__nom_modele = nom_modele
        self.set_REPORT_ID("EMBED1")
        self.set_REPORT_MODELE(nom_modele)
        self.set_REPORT_LIBELLE("EMBEDDING")
        
     def create_modele(self,Matrix=None,len_embedding_dict=0):
        x  = input_layer = Input(shape=(self.MAXLEN,))
        x  = Embedding(self.NUM_WORDS, self.EMBEDDING_DIM)(x)
        x  = Conv1D(filters=32, kernel_size=8, activation='relu')(x)
        x  = GlobalAveragePooling1D()(x)
        x  = Flatten()(x)
        x  = Dense(256, activation='relu')(x)
        x  = Dropout(0.4)(x)
        output_layer = Dense(27, activation='softmax')(x)

        model = Model(inputs=input_layer, outputs=output_layer)
        model.compile(optimizer=Adam(learning_rate=1e-3), loss='categorical_crossentropy', metrics=['accuracy'])    
        
        self.set_model(model)

        return model         
        
class RNN_STEMMER(DS_RNN):     

     def __init__(self, nom_modele):
        super().__init__(nom_modele)
            
        self.__nom_modele = nom_modele
        self.set_REPORT_ID("EMBED2")
        self.set_REPORT_MODELE(nom_modele)
        self.set_REPORT_LIBELLE("EMBEDDING STEMMER")
        
     def add_traitement(self,mots,pays_langue) :
        mots = Stemmer_sentence(mots,pays_langue)
        return mots   
        
     def create_modele(self,Matrix=None,len_embedding_dict=0):
        model = Sequential()
        model.add(Embedding(self.NUM_WORDS, self.EMBEDDING_DIM))
        model.add(Conv1D(filters=32, kernel_size=8, activation='relu'))
        model.add(GlobalAveragePooling1D())
        model.add(Flatten())
        model.add(Dense(256, activation='relu'))
        #model.add(BatchNormalization())
        model.add(Dropout(0.4))
        model.add(Dense(27, activation='softmax'))
        model.compile(optimizer=Adam(learning_rate=1e-3), loss='categorical_crossentropy', metrics=['accuracy'])    
        
        self.set_model(model)

        return model   
        

 
class RNN_GRU(DS_RNN):     

     def __init__(self, nom_modele):
        super().__init__(nom_modele)
            
        self.__nom_modele = nom_modele
        self.set_REPORT_ID("GRU")
        self.set_REPORT_MODELE(nom_modele)
        self.set_REPORT_LIBELLE("RNN_GRU")
        
     def create_modele(self,Matrix=None,len_embedding_dict=0):
        model = Sequential()
        model.add(Embedding(self.NUM_WORDS, self.EMBEDDING_DIM))
        print("NUM_WORDS", self.NUM_WORDS)
        print("EMBEDDING_DIM", self.EMBEDDING_DIM)
        model.add(RNN(GRUCell(128), return_sequences=True))
        model.add(Dropout(0.5))
        model.add(GlobalAveragePooling1D())
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(27, activation='softmax'))
        model.compile(optimizer=Adam(learning_rate=1e-3), loss='categorical_crossentropy', metrics=['accuracy'])    
        
        self.set_model(model)

        return model  
     

class RNN_SPACY(DS_RNN):     

     def __init__(self, nom_modele):
        super().__init__(nom_modele)
            
        self.__nom_modele = nom_modele
        self.set_REPORT_ID("EMBED2")
        self.set_REPORT_MODELE(nom_modele)
        self.set_REPORT_LIBELLE("EMBEDDING STEMMER")
        
     def add_traitement(self,mots,pays_langue) :
        mots = Stemmer_sentence(mots,pays_langue)
        return mots   
        
     def create_modele(self,Matrix=None,len_embedding_dict=0):
    
        print("output : ",self.EMBEDDING_DIM)
        #print("Matrix.shape = " ,Matrix.shape)
        print("len_embedding_dict = " ,len_embedding_dict)
        model = Sequential()
        model.add(Embedding(len_embedding_dict, self.EMBEDDING_DIM, embeddings_initializer=Constant(Matrix), trainable=False))
        #model.add(Bidirectional(LSTM(64, dropout=0.25, recurrent_dropout=0.1)))
        #model.add(LSTM(128))
        #model.add(Conv1D(filters=64, kernel_size=8, activation='relu'))
        model.add(Conv1D(filters=128, kernel_size=8, activation='relu'))
        model.add(GlobalAveragePooling1D())
        model.add(Dropout(0.5))
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(27, activation='softmax'))
        model.compile(optimizer=Adam(learning_rate=1e-3), loss='categorical_crossentropy', metrics=['accuracy'])   
        
        self.set_model(model)

        return model           

"""
class RNN_CAMENBERT(DS_RNN):     

     def __init__(self, nom_modele):
        super().__init__(nom_modele)
            
        self.__nom_modele = nom_modele
        self.set_REPORT_ID("EMBED2")
        self.set_REPORT_MODELE(nom_modele)
        self.set_REPORT_LIBELLE("EMBEDDING STEMMER")
        
        self.tokenizer = CamembertTokenizer.from_pretrained("camembert-base")
        self.camembert_model = TFCamembertForSequenceClassification.from_pretrained("camembert-base")
        
     def add_traitement(self,mots,pays_langue) :
        mots = Stemmer_sentence(mots,pays_langue)
        return mots   
        
     def create_modele(self,Matrix=None,len_embedding_dict=0):
        model = Sequential()
        model.add(self.camembert_model) 
        model.add(LSTM(128))
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.4))
        model.add(Dense(27, activation='softmax'))
        model.compile(optimizer=Adam(learning_rate=1e-3), loss='categorical_crossentropy', metrics=['accuracy'])    
        
        self.set_model(model)

        return model   
  
     def fit_modele(self,epochs,savefics=False,freeze=0,spacy=False):
        
        #X_train_avant, X_test_avant, y_train, y_test = ds.DS_Model.Train_Test_Split_(self)
        X_train_avant, X_test_avant, y_train, y_test = self.Train_Test_Split_()
        
        
        X_train = X_train_avant['designation']
        X_test = X_test_avant['designation']
        
        y_train,y_test,label_encoder = self.preprossessing_Y(y_train,y_test)
        
        print(X_train[:5])
        print(y_train[:5])
        
        
        train_encodings = self.tokenizer(list(X_train), truncation=True, padding=True, max_length=self.MAXLEN)
        test_encodings = self.tokenizer(list(X_test), truncation=True, padding=True, max_length=self.MAXLEN)
        
        model = self.create_modele()    
        
        lr_plateau = callbacks.ReduceLROnPlateau(monitor = 'val_loss',
                            patience=4,
                            factor=0.5,
                            verbose=1,
                            mode='min')
        
        training_history = model.fit(train_encodings, y_train,batch_size = 32, epochs=epochs, validation_data = [test_encodings, y_test],callbacks=[lr_plateau])

        train_acc = training_history.history['accuracy']
        val_acc = training_history.history['val_accuracy']
        tloss = training_history.history['loss']
        tvalloss=training_history.history['val_loss']
        
        return train_acc,val_acc,tloss,tvalloss
"""    

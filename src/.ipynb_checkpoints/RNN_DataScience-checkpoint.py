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
from nltk.stem.snowball import SnowballStemmer
from  nltk.stem.snowball import FrenchStemmer,EnglishStemmer,GermanStemmer,ItalianStemmer,DutchStemmer,SpanishStemmer,ItalianStemmer
from nltk.stem import WordNetLemmatizer


from tensorflow.keras import Sequential,Input, Model
from tensorflow.keras.layers import Embedding, Dense, GlobalAveragePooling1D, RNN, GRUCell,GRU
from tensorflow.keras.layers import  Dropout ,Conv1D,Flatten,Bidirectional,BatchNormalization

from keras.optimizers import Adam
from tensorflow.keras import callbacks
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC,LinearSVC
from sklearn.metrics import f1_score

import string
import re
import nltk
import pickle
import unicodedata
from nltk.corpus import stopwords
import swifter
import spacy
from langdetect import detect



import tensorflow as tf
from tensorflow.keras.utils import to_categorical

import Bibli_DataScience_3_3 as ds

def unicode_to_ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn')



    
class DS_RNN(ds.DS_Model):
    
     def __init__(self, nom_modele):
     
        super().__init__(nom_modele)
            
        
        self.__nom_modele = nom_modele
        self.__model = Sequential()
        
        self.stop_words = self.get_stopwordFR()['MOT'].tolist()
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
        nltk.download('stopwords')
        nltk.download('wordnet')
        
        self.nlp_fr = spacy.load('fr_core_news_sm')
        self.nlp_en = spacy.load('en_core_web_sm')
        self.nlp_de = spacy.load('de_core_news_sm')
        self.nlp_es = spacy.load('es_core_news_sm')
        self.nlp_it = spacy.load('it_core_news_sm')
        self.nlp_nl = spacy.load('nl_core_news_sm')
        self.nlp_ca = spacy.load('ca_core_news_sm')
      
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

     def restore_fit_arrays(self):
        train_acc  = ds.load_ndarray(self.__nom_modele+'_accuracy')
        val_acc    = ds.load_ndarray(self.__nom_modele+'_val_accuracy')
        tloss      = ds.load_ndarray(self.__nom_modele+'_loss')
        tvalloss   = ds.load_ndarray(self.__nom_modele+'_val_loss')
        return train_acc, val_acc, tloss, tvalloss

     def restore_predict_arrays(self):
        self.__y_orig = ds.load_ndarray(self.__nom_modele+'_y_orig')
        self.__y_pred = ds.load_ndarray(self.__nom_modele+'_y_pred')
        return self.__y_orig, self.__y_pred

     def restore_predict_dataframe(self):
        self.__df_pred = ds.load_dataframe(self.__nom_modele+'_df_predict')
        return self.__df_pred

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
     
     
     
     def preprocess_sentence(self,w):
        w = str(w)
    # Remplacer les entités HTML par des caractères spécifiques ou les supprimer
        replacements = {
            '&eacute;': 'e',
            '&amp;': '',    # Esperluette
            '&lt;': '',     # Inférieur à
            '&gt;': '',     # Supérieur à
            '&quot;': '',   # Guillemet double
            '&apos;': '',   # Apostrophe
            '&nbsp;': '',   # Espace insécable
            '&copy;': '',   # Droit d'auteur
            '&reg;': '',    # Marque déposée
            '&euro;': '',   # Symbole de l'euro
            '&rsquo;': '',
            '&agrave;': 'a',
            '&ccedil;': 'c',
            '&egrave;': 'e',
            '&iacute;': 'i',
            '&ntilde;': 'n',
            '&ouml;': 'o',
        }
        for entity, replacement in replacements.items():
            w = w.replace(entity, replacement)
        w = unicode_to_ascii(w.lower().strip())
        # Appliquer les autres règles de nettoyage
        w = w.replace("n°", "??numero??")
        w = re.sub(r"([?.!,¿])", r" \1 ", w)
        w = re.sub(r'[" "]+', " ", w)
        w = re.sub(r"[^a-zA-Z?.!°]+", " ", w)
        w = re.sub(r'\b\w{0,2}\b', '', w)
        w = w.replace("? ? numero ? ?", "n°")


        # Suppression des stopwords
        mots = word_tokenize(w.strip())
        mots = [mot for mot in mots if mot not in self.stop_words]
        return ' '.join(mots).strip()
        
     def preprocess_text(self,text):
        try:
            lang = detect(text)
        except:
            lang = "fr"  # Définit le français comme langue par défaut
        #text = text.lower()
        text = text.translate(str.maketrans('', '', string.punctuation))
        # Utilise le français comme langue de fallback pour la tokenisation
        tokens = word_tokenize(text, language='french' if lang not in ['english', 'spanish', 'dutch', 'german', 'italian'] else lang)
        # Définit le français comme langue de fallback pour les stop words
        stop_words = set(stopwords.words({
            'en': 'english',
            'es': 'spanish',
            'de': 'german',
            'nl': 'dutch',
            'it': 'italian',
            'ca': 'french',  # Utilise explicitement le français pour le catalan
            'fr': 'french'
        }.get(lang, 'french')))  # Fallback sur le français pour toute autre langue non spécifiée

        tokens = [word for word in tokens if word not in stop_words]
        return tokens   

     def preprossessing_X(self,row):
        # au chois on utilise la langue definie par notre soin , soit on laisse décider detect_lang
        pays_langue = row['PAYS_LANGUE']
        
        mots =self.preprocess_sentence(row['phrases'])
        mots = self.preprocess_text(mots)
        
        return {'phrases': mots, 'PAYS_LANGUE': pays_langue}
        
     def preprocess_stemmer(self,tokens):
         # Détection de la langue
        try:
            lang = detect(' '.join(tokens))
        except:
            lang = "fr"  # Langue par défaut
        # Adaptation des ressources linguistiques en fonction de la langue détectée
        if lang == 'en':
            stop_words = set(stopwords.words('english'))
            stemmer = SnowballStemmer("english")
        elif lang == 'es':
            stop_words = set(stopwords.words('spanish'))
            stemmer = SnowballStemmer("spanish")
        elif lang == 'de':
            stop_words = set(stopwords.words('german'))
            stemmer = SnowballStemmer("german")
        elif lang == 'nl':
            stop_words = set(stopwords.words('dutch'))
            stemmer = SnowballStemmer("dutch")
        elif lang == 'it':
            stop_words = set(stopwords.words('italian'))
            stemmer = SnowballStemmer("italian")
        elif lang == 'ca':
            stop_words = set(stopwords.words('french'))
            stemmer = SnowballStemmer("french")
        else:
            stop_words = set(stopwords.words('french'))
            stemmer = SnowballStemmer("french")


        preprocessed_tokens = []
        for token in tokens:
            # Conversion en minuscules et suppression de la ponctuation
            token = token.lower()
            token = token.translate(str.maketrans('', '', string.punctuation))
            # Stemmisation
            stemmed_token = stemmer.stem(token)
            preprocessed_tokens.append(stemmed_token)

        return preprocessed_tokens
        
        
     def preprocess_lemmer(self,tokens):
        lemmatizer = WordNetLemmatizer()

        preprocessed_tokens = []
        for token in tokens:
            # Conversion en minuscules et suppression de la ponctuation
            token = token.lower()
            token = token.translate(str.maketrans('', '', string.punctuation))
            # Lemmatisation
            lemmatized_token = lemmatizer.lemmatize(token)
            preprocessed_tokens.append(lemmatized_token)

        return preprocessed_tokens   
        
     def preprocess_spacy(self,text):
        # Détection de la langue
        try:
            lang = detect(text)
        except:
            lang = "en"  # Langue par défaut

        # Sélection du modèle SpaCy approprié
        nlp = self.get_spacy_model(lang)


        # Nettoyage du texte
        #text = text.lower()
        text = text.translate(str.maketrans('', '', string.punctuation))

        # Lemmatisation avec SpaCy
        doc = nlp(text)
        lemmatized_tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct and not token.is_space]

        return lemmatized_tokens      
    
     def traiter_phrases(self):
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
            print("load y_train_avant.shape ",y_train_avant.shape)
            
            return X_train, X_test, y_train_avant, y_test_avant
        

         
        X_train_avant, X_test_avant, y_train_avant, y_test_avant = super().Train_Test_Split_(train_size, random_state)
        
        
        
        print(X_train_avant.info())
      
        print("etape 1/6")
        
        X_train_processed_list = X_train_avant[['phrases', 'PAYS_LANGUE']].apply(self.preprossessing_X, axis=1).tolist()
        X_train = pd.DataFrame(X_train_processed_list)
        print("etape 2/6")
        X_test_processed_list = X_test_avant[['phrases', 'PAYS_LANGUE']].apply(self.preprossessing_X, axis=1).tolist()
        X_test = pd.DataFrame(X_test_processed_list)
        print("etape 3/6")
        if fic == "Save" :
            print("Sauvegarde de jeu d'entrainement")
            ds.save_ndarray(X_train,self.__nom_modele+'_X_train')
            ds.save_ndarray(X_test,self.__nom_modele+'_X_test')
            ds.save_ndarray(y_train_avant,self.__nom_modele+'_y_train')
            ds.save_ndarray(y_test_avant,self.__nom_modele+'_y_test')
        
        print("save y_train_avant.shape ",y_train_avant.shape)
        return X_train, X_test, y_train_avant, y_test_avant
    
     def fit_modele(self,epochs,savefics=False,freeze=0,Train="None",stemming=False,lemming=False,spacy=False):
    
        X_train,X_test,y_train,y_test = self.Train_Test_Split_(fic=Train)
        
       
        
        embedding_dict = {}
        embedding_matrix = []
      
        print("self.EMBEDDING_DIM",self.EMBEDDING_DIM)  
          
        print("suite")
       
        X_train = X_train['phrases']
        X_test = X_test['phrases']
        
        y_train,y_test,label_encoder = self.preprossessing_Y(y_train,y_test)
        
       
        
        
        
        if stemming:
            X_train = [self.preprocess_stemmer(tokens) for tokens in X_train]
            X_test = [self.preprocess_stemmer(tokens) for tokens in X_test]
        elif lemming:
            X_train = [self.preprocess_lemmer(tokens) for tokens in X_train]
            X_test = [self.preprocess_lemmer(tokens) for tokens in X_test]
        elif spacy:
            print("etape 4/6")
            X_train = [self.preprocess_spacy(tokens) for tokens in X_train]
            print("etape 5/6")
            X_test = [self.preprocess_spacy(tokens) for tokens in X_test]
            print("etape 6/6")
    
        
        
        tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=self.NUM_WORDS)
        # Mettre à jour le dictionnaire du tokenizer
        tokenizer.fit_on_texts(X_train)
        
        word2idx = tokenizer.word_index
        idx2word = tokenizer.index_word
        vocab_size = len(tokenizer.word_index) + 1  # Taille du vocabulaire

        X_train = tokenizer.texts_to_sequences(X_train)
        X_test = tokenizer.texts_to_sequences(X_test)
        
        X_train = tf.keras.preprocessing.sequence.pad_sequences(X_train,  padding='post', truncating='post')
        X_test = tf.keras.preprocessing.sequence.pad_sequences(X_test,  padding='post', truncating='post')
               
        print( "vocab_size : ",vocab_size)
        model = self.create_modele(embedding_matrix,vocab_size)    
        
        if Train == "Load" :
            model = ds.load_model(self.__nom_modele+'_weight')
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
        feature_model_rnn = Model(inputs=model.inputs, outputs=model.layers[-2].output)
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
        
     def create_modele(self,Matrix=None,vocab_size=0):
        model = Sequential()
        model.add(Input(shape=(None,)))
        model.add(Embedding(vocab_size, self.EMBEDDING_DIM))
        model.add(Conv1D(filters=32, kernel_size=8, activation='relu'))
        # Ajout d'une couche GRU
        model.add(GRU(128, return_sequences=True)) 
        model.add(GlobalAveragePooling1D())
        model.add(Dense(256, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))  # Augmentation du taux de dropout pour réduire le surajustement
        model.add(Dense(27, activation='softmax'))
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
        mots = self.preprocess_stemmer(mots,pays_langue)
        return mots   
        
     def create_modele(self,Matrix=None,vocab_size=0):
        model = Sequential()
        model.add(Input(shape=(None,)))
        model.add(Embedding(vocab_size, self.EMBEDDING_DIM))
        model.add(Conv1D(filters=32, kernel_size=8, activation='relu'))
        # Ajout d'une couche GRU
        model.add(GRU(128, return_sequences=True)) 
        model.add(GlobalAveragePooling1D())
        model.add(Dense(256, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))  # Augmentation du taux de dropout pour réduire le surajustement
        model.add(Dense(27, activation='softmax'))
        model.compile(optimizer=Adam(learning_rate=1e-3), loss='categorical_crossentropy', metrics=['accuracy'])    
        
        self.set_model(model)

        return model   
        
class RNN_LEMMER(DS_RNN):     

     def __init__(self, nom_modele):
        super().__init__(nom_modele)
            
        self.__nom_modele = nom_modele
        self.set_REPORT_ID("EMBED3")
        self.set_REPORT_MODELE(nom_modele)
        self.set_REPORT_LIBELLE("EMBEDDING LEMMER")
        
     def add_traitement(self,mots,pays_langue) :
        mots = self.preprocess_lemmer(mots,pays_langue)
        return mots   
        
     def create_modele(self,Matrix=None,vocab_size=0):
        model = Sequential()
        model.add(Input(shape=(None,)))
        model.add(Embedding(vocab_size, self.EMBEDDING_DIM))
        model.add(Conv1D(filters=32, kernel_size=8, activation='relu'))
        # Ajout d'une couche GRU
        model.add(GRU(128, return_sequences=True)) 
        model.add(GlobalAveragePooling1D())
        model.add(Dense(256, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))  # Augmentation du taux de dropout pour réduire le surajustement
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
        
        # Chargement des modèles de langue avec un mappage spécifique
        self.nlp_en = spacy.load('en_core_web_sm')
        self.nlp_fr = spacy.load('fr_core_news_sm')
        self.nlp_es = spacy.load('es_core_news_sm')
        self.nlp_de = spacy.load('de_core_news_sm')
        self.nlp_nl = spacy.load('nl_core_news_sm')
        self.nlp_it = spacy.load('it_core_news_sm')
        
        # Fonction pour sélectionner le modèle SpaCy en fonction de la langue détectée
     def get_spacy_model(self,lang):
        if lang == 'fr':
            return self.nlp_fr
        elif lang == 'es':
            return self.nlp_es
        elif lang == 'en':
            return self.nlp_en
        elif lang == 'de':
            return self.nlp_de
        elif lang == 'nl':
            return self.nlp_nl
        elif lang == 'it':
            return self.nlp_it
        else:  # par défaut à l'anglais
            return self.nlp_fr
        
     def preprocess_text(self,text):
        try:
            lang = detect(text)
        except:
            lang = "fr"  # Définit le français comme langue par défaut
        #text = text.lower()
        text = text.translate(str.maketrans('', '', string.punctuation))
       # Sélection du modèle SpaCy approprié
        nlp = self.get_spacy_model(lang)

        # Traitement du texte avec spaCy
        doc = nlp(text)

        # Filtrage des tokens qui ne sont pas des stopwords et qui ne sont pas des signes de ponctuation
        tokens_sans_stopwords = [token.text for token in doc if not token.is_stop and token.text not in string.punctuation]

        return ' '.join(tokens_sans_stopwords).strip()  
        
     
        
     def create_modele(self,Matrix=None,vocab_size=0):
    
        print("output : ",self.EMBEDDING_DIM)
        print("vocab_size = " ,vocab_size)
        model = Sequential()
        model.add(Input(shape=(None,)))
        model.add(Embedding(vocab_size, self.EMBEDDING_DIM))
        model.add(Conv1D(filters=32, kernel_size=8, activation='relu'))
        # Ajout d'une couche GRU
        model.add(GRU(128, return_sequences=True)) 
        model.add(GlobalAveragePooling1D())
        model.add(Dense(256, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))  # Augmentation du taux de dropout pour réduire le surajustement
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
        model.add(Input(shape=(None,)))
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

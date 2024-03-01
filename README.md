# ***PROJET RAKUTEN***  

## **1) Description du projet**  
**Description du problème**    

L'objectif de ce défi est la classification à grande échelle des données de produits multimodales (textes et images) en type de produits.  
Par exemple, dans le catalogue de Rakuten France, **un produit** avec une désignation "Grand Stylet Ergonomique Bleu Gamepad Nintendo Wii U - Speedlink Pilot Style" **est associé à une image** (\images\ReadMe_ML_filesimage_938777978_product_201115110.jpg) **et
à une description supplémentaire.** Ce produit est catégorisé sous **le code** produit 50.

## **2) Introduction**   

**description des fichiers**

le but du projet est de prédire le code de chaque produit tel que défini dans le catalogue de Rakuten France.  
La catégorisation des annonces de produits se fait par le biais de la désignation, de la description (quand elle est présente) et des images.  
Les fichiers de données sont distribués ainsi :  
***X_train_update.csv*** : fichier d'entrée d'entraînement  
***Y_train_CVw08PX.csv*** : fichier de sortie d'entraînement  
***X_test_update.csv*** : fichier d'entrée de test  
Un fichier images.zip est également fourni, contenant toutes les images.  
La décompression de ce fichier fournira un dossier nommé "images" avec deux sous-dossiers nommés ***"image_train"*** et ***"image_test"***, contenant respectivement les images d'entraînement et de test.  
Pour notre part, ne participant pas au challenge Rakuten, je n'ai pas pas accès au fichier de sortie de test.  
Le fichier d’entrée de test est donc inutilisable.  
**X_train_update.csv** : fichier d'entrée d'entraînement :  
La première ligne des fichiers d'entrée contient l'en-tête et les colonnes sont séparées par des virgules (",").  
Les colonnes sont les suivantes :  


*   **Un identifiant entier pour le produit**. Cet identifiant est utilisé pour associer le produit à son code de produit correspondant.
*   **Désignation** - Le titre du produit, un court texte résumant le produit
*   **Description** - Un texte plus détaillé décrivant le produit. Tous les marchands n'utilisent pas ce champ, il se peut donc que le champ de description contienne la valeur NaN pour de nombreux produits, afin de conserver l'originalité des données.
*   **productid** - Un identifiant unique pour le produit.
*   **imageid** - Un identifiant unique pour l'image associée au produit.
Les champs imageid et productid sont utilisés pour récupérer les images dans le dossier
d'images correspondant. Pour un produit donné, le nom du fichier image est :
image_imageid_product_productid.jpg ex : **image_1263597046_product_3804725264.jpg**  

**Y_train_CVw08PX.csv** : fichier de sortie d'entraînement :  
La première ligne des fichiers d'entrée contient l'en-tête et les colonnes sont séparées par des virgules (",").  
Les colonnes sont les suivantes :  
*  **Un identifiant entier pour le produit**. Cet identifiant est utilisé pour associer le produit à son
code de produit correspondant.
*  **prdtypecode** – Catégorie dans laquelle le produit est classé.

La liaison entre les fichiers se fait par une jointure sur l’identifiant entier présent sur les deux
fichiers.

## **3) Objectif de ce Notebook**  
Ce notebook fait partie d'un ensemble de sous-projets dont le resultat représente le **projet Rakuten** que j'ai réalisé pour mon diplôme de data Scientist chez Datascientest.com.  

Ce repositery est la partie **Réseaux de Neurones Récurrents** et ne traite que de la partie texte.  
Il fait suite à la partie **Machine Learning**    
Il utilise la bibliothèque **Bibli_DataScience** commune à l'ensemble du projet et la bibbliothèque **RNN_DataScience.py** propre à cette partie.  
D'autres dépots viendront, à savoir  :


*   La partie image  traitée par des réseaux convolutifs
*   Une quatrième partie qui est une syntèse par le media Streamlit



Ce notebook traite du Traitement automatique du langage naturel (**natural language processing** ou **NLP**) et teste plusieurs approches.

explication de la bibliothèque **RNN_DataScience.py**  :   

J'ai construit tout le code sur un modèle objet.  
Chaque modèle est une classe et hérite d'une classe générale **DS_RNN**  

*   Une `tokenisation` simple suivie d'une couche d'Embedding de tensorFlow -> classe **RNN_EMBEDDING**  
*   Une `tokenisation` puis une racinisation (`stemming`) suivie d'une couche d'Embedding de tensorFlow -> classe **RNN_STEMMER**
*   Une tokenisation puis une `lemmatisation` en utilisant `NLTK` suivie d'une couche d'Embedding de tensorFlow -> classe **RNN_LEMMER**
*   Une tokenisation puis une `lemmatisation` en utilisant `SPACY` suivie d'une couche d'Embedding de tensorFlow -> classe **RNN_SPACY**

Ces 4 modèles utilisent un réseau de neurones comportant une couche GRU.

Un cinquième modèle fait partie d'un notebook différent car son architecture est un peu différente (Word2VEC) : Modele_RNN_Word2Vec.ipynb  

Pour tous les modèles on utilise le même **préprocessing de base** :  

## ETAPE 1 : Passage en minuscule
Dans un premier temps, nous transformons les majuscules en minuscules car les étapes suivantes sont sensibles à la casse
## ETAPE 2: Tokenisation
 Il s’agit de décomposer une phrase, et donc un document, en tokens. Un token est un élément correspondant à un mot ou une ponctuation, cependant de nombreux cas ne sont pas triviaux à traiter :
Les mots avec un trait d’union, exemple : peut être et peut-être qui ont des significations très différentes ;
Les dates et heures qui peuvent être séparées par des points, des slashs, des deux points ;
Les apostrophes ;
Les caractères spéciaux : émoticônes, formules mathématiques.
## ETAPE 3: Retrait des stopwords
Ensuite, nous retirons les mots appartenant aux stopwords. Il s’agit de listes de mots définies au préalable soit par l’utilisateur soit dans des librairies existantes. Ces listes se composent de mots qui n’apportent aucune information, qui sont en général très courants et donc présents dans la plupart des documents, par exemple : je, nous, avoir (le verbe et ses conjugaisons). La suppression de ces stopwords permet de ne pas polluer les représentations des documents afin qu’elle ne contienne que les mots représentatifs et significatifs. Ce “nettoyage” du texte peut aussi s’accompagner de la suppression d’autres éléments comme les nombres, les dates, la ponctuation etc.


=> **Modèle 1 : classe RNN_EMBEDDING**

## ETAPE 4 : Groupement sémantique
Dès lors, nous disposons pour chaque document d’une liste “nettoyée” de mots porteurs de sens et séparés en tokens. Mais un mot peut être écrit au pluriel, au singulier ou avec différents accords et les verbes peuvent être conjugués à différents temps et personnes.
Nous devons donc réduire les différences grammaticales des mots en trouvant des formes communes. Pour ce faire, nous disposons de deux méthodes distinctes :
La stemmatisation, qui ne prend pas en compte le contexte de la phrase
La lemmatisation, qui prend en compte le contexte
### ETAPE 4.1 : La stemmatisation
La stemmatisation (ou racinisation) réduit les mots à leur radical ou racine.

=> **Modèle 2 : classe RNN_STEMMER**

### ETAPE 4.2 : La Lemmatisation
La lemmatisation, qui prend en considération le contexte dans lequel le mot est écrit, a pour but de trouver la forme canonique du mot, le lemme. Par conséquent, elle doit se faire après la transformation des lettres majuscules en minuscules et avant la tokenisation car les mots présents avant et après sont importants pour déterminer la nature du mot.
Le lemme correspond à l’infinitif des verbes et à la forme au masculin singulier des noms, adjectifs et articles. Par exemple cette méthode est capable de faire la différence entre “nous avions” : verbe avoir et “les avions” : le pluriel d’un avion.

=> **Modèle 3 : classe RNN_LEMMER**  
=> **Modèle 4 : classe RNN_SPACY**  
=> **Modèle 5 : Word2Vec**

# Modèle 1 : classe RNN_EMBEDDING

### Une tokenisation simple suivie d'une couche d'Embedding de tensorFlow


```python
# instanciation du modèle
emb = rnn.RNN_EMBEDDING("EMBEDDING")
```


    Pandas Apply:   0%|          | 0/84916 [00:00<?, ?it/s]


    [nltk_data] Downloading package punkt to /root/nltk_data...
    [nltk_data]   Package punkt is already up-to-date!
    [nltk_data] Downloading package stopwords to /root/nltk_data...
    [nltk_data]   Package stopwords is already up-to-date!
    [nltk_data] Downloading package wordnet to /root/nltk_data...
    [nltk_data]   Package wordnet is already up-to-date!
    


```python
# entrainement du modèle
train_acc,val_acc,tloss,tvalloss = emb.fit_modele(5,True)
```

    <class 'pandas.core.frame.DataFrame'>
    Index: 67932 entries, 83256 to 20596
    Data columns (total 16 columns):
     #   Column          Non-Null Count  Dtype  
    ---  ------          --------------  -----  
     0   Id              67932 non-null  int64  
     1   designation     67932 non-null  object 
     2   description     44084 non-null  object 
     3   productid       67932 non-null  int64  
     4   imageid         67932 non-null  int64  
     5   PAYS_LANGUE     67932 non-null  object 
     6   RATIO_LANGUE    67932 non-null  float64
     7   ORIGINE_LANGUE  67932 non-null  object 
     8   pays_design     67932 non-null  object 
     9   Ratio_design    67932 non-null  float64
     10  pays_descr      44033 non-null  object 
     11  Ratio_descr     44033 non-null  float64
     12  descr_NaN       67932 non-null  bool   
     13  nom_image       67932 non-null  object 
     14  filepath        67932 non-null  object 
     15  phrases         67932 non-null  object 
    dtypes: bool(1), float64(3), int64(3), object(9)
    memory usage: 8.4+ MB
    None
    etape 1/6
    etape 2/6
    etape 3/6
    save y_train_avant.shape  (67932,)
    self.EMBEDDING_DIM 300
    suite
    (67932, 27)
    (16984, 27)
    vocab_size :  112086
    Epoch 1/5
    2123/2123 [==============================] - 1724s 811ms/step - loss: 1.1351 - accuracy: 0.6738 - val_loss: 0.6975 - val_accuracy: 0.8037 - lr: 0.0010
    Epoch 2/5
    2123/2123 [==============================] - 1713s 807ms/step - loss: 0.4697 - accuracy: 0.8641 - val_loss: 0.6949 - val_accuracy: 0.8135 - lr: 0.0010
    Epoch 3/5
    2123/2123 [==============================] - 1708s 804ms/step - loss: 0.2508 - accuracy: 0.9280 - val_loss: 0.7912 - val_accuracy: 0.8065 - lr: 0.0010
    Epoch 4/5
    2123/2123 [==============================] - 1713s 807ms/step - loss: 0.1426 - accuracy: 0.9601 - val_loss: 0.8739 - val_accuracy: 0.8134 - lr: 0.0010
    Epoch 5/5
    2123/2123 [==============================] - 1723s 812ms/step - loss: 0.0983 - accuracy: 0.9723 - val_loss: 1.0212 - val_accuracy: 0.7970 - lr: 0.0010
    531/531 [==============================] - 69s 128ms/step
    2123/2123 [==============================] - 269s 127ms/step
    531/531 [==============================] - 67s 127ms/step
    y_test_original2[:5]  [1301 1140 2583 2280 2403]
    


```python
ds.plot_fit(train_acc,val_acc,tloss,tvalloss)
```


    
![png](/images/ReadMe_ML_files/output_23_0.png)
    


### Tableau des repartitions des **classes prédites** pour chaque **classe réelle**


```python
df_pred = emb.get_df_pred()
df_pred
```





  <div id="df-fdc7ab96-5fa3-40e3-9cc3-080f7a0ace66" class="colab-df-container">
    <div>

</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Categorie</th>
      <th>predict</th>
      <th>pourc</th>
      <th>predict2</th>
      <th>pourc2</th>
      <th>predict3</th>
      <th>pourc3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>10</td>
      <td>2705</td>
      <td>0.335474</td>
      <td>10</td>
      <td>0.327448</td>
      <td>2403</td>
      <td>0.130016</td>
    </tr>
    <tr>
      <th>1</th>
      <td>40</td>
      <td>40</td>
      <td>0.697211</td>
      <td>2705</td>
      <td>0.059761</td>
      <td>2462</td>
      <td>0.043825</td>
    </tr>
    <tr>
      <th>2</th>
      <td>50</td>
      <td>50</td>
      <td>0.761905</td>
      <td>2462</td>
      <td>0.083333</td>
      <td>40</td>
      <td>0.047619</td>
    </tr>
    <tr>
      <th>3</th>
      <td>60</td>
      <td>60</td>
      <td>0.843373</td>
      <td>2462</td>
      <td>0.072289</td>
      <td>50</td>
      <td>0.048193</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1140</td>
      <td>1140</td>
      <td>0.679775</td>
      <td>1280</td>
      <td>0.080524</td>
      <td>40</td>
      <td>0.052434</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1160</td>
      <td>1160</td>
      <td>0.878635</td>
      <td>2705</td>
      <td>0.025284</td>
      <td>40</td>
      <td>0.021492</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1180</td>
      <td>1180</td>
      <td>0.568627</td>
      <td>2705</td>
      <td>0.071895</td>
      <td>1281</td>
      <td>0.071895</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1280</td>
      <td>1280</td>
      <td>0.611910</td>
      <td>1281</td>
      <td>0.174538</td>
      <td>1140</td>
      <td>0.044148</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1281</td>
      <td>1281</td>
      <td>0.659420</td>
      <td>1280</td>
      <td>0.084541</td>
      <td>2705</td>
      <td>0.067633</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1300</td>
      <td>1300</td>
      <td>0.929633</td>
      <td>1280</td>
      <td>0.025768</td>
      <td>10</td>
      <td>0.006938</td>
    </tr>
    <tr>
      <th>10</th>
      <td>1301</td>
      <td>1301</td>
      <td>0.900621</td>
      <td>1281</td>
      <td>0.037267</td>
      <td>1320</td>
      <td>0.012422</td>
    </tr>
    <tr>
      <th>11</th>
      <td>1302</td>
      <td>1302</td>
      <td>0.787149</td>
      <td>1280</td>
      <td>0.046185</td>
      <td>1281</td>
      <td>0.040161</td>
    </tr>
    <tr>
      <th>12</th>
      <td>1320</td>
      <td>1320</td>
      <td>0.774691</td>
      <td>2060</td>
      <td>0.035494</td>
      <td>1560</td>
      <td>0.024691</td>
    </tr>
    <tr>
      <th>13</th>
      <td>1560</td>
      <td>1560</td>
      <td>0.840394</td>
      <td>2060</td>
      <td>0.040394</td>
      <td>2582</td>
      <td>0.035468</td>
    </tr>
    <tr>
      <th>14</th>
      <td>1920</td>
      <td>1920</td>
      <td>0.912892</td>
      <td>1560</td>
      <td>0.030197</td>
      <td>2060</td>
      <td>0.024390</td>
    </tr>
    <tr>
      <th>15</th>
      <td>1940</td>
      <td>1940</td>
      <td>0.776398</td>
      <td>2522</td>
      <td>0.037267</td>
      <td>2705</td>
      <td>0.031056</td>
    </tr>
    <tr>
      <th>16</th>
      <td>2060</td>
      <td>2060</td>
      <td>0.802803</td>
      <td>1560</td>
      <td>0.066066</td>
      <td>1920</td>
      <td>0.022022</td>
    </tr>
    <tr>
      <th>17</th>
      <td>2220</td>
      <td>2220</td>
      <td>0.715152</td>
      <td>1560</td>
      <td>0.054545</td>
      <td>2060</td>
      <td>0.048485</td>
    </tr>
    <tr>
      <th>18</th>
      <td>2280</td>
      <td>2280</td>
      <td>0.751050</td>
      <td>2403</td>
      <td>0.154412</td>
      <td>2705</td>
      <td>0.044118</td>
    </tr>
    <tr>
      <th>19</th>
      <td>2403</td>
      <td>2403</td>
      <td>0.758115</td>
      <td>2705</td>
      <td>0.078534</td>
      <td>2280</td>
      <td>0.062827</td>
    </tr>
    <tr>
      <th>20</th>
      <td>2462</td>
      <td>2462</td>
      <td>0.721831</td>
      <td>40</td>
      <td>0.112676</td>
      <td>50</td>
      <td>0.035211</td>
    </tr>
    <tr>
      <th>21</th>
      <td>2522</td>
      <td>2522</td>
      <td>0.917836</td>
      <td>1560</td>
      <td>0.019038</td>
      <td>2403</td>
      <td>0.013026</td>
    </tr>
    <tr>
      <th>22</th>
      <td>2582</td>
      <td>2582</td>
      <td>0.758687</td>
      <td>2060</td>
      <td>0.067568</td>
      <td>1560</td>
      <td>0.057915</td>
    </tr>
    <tr>
      <th>23</th>
      <td>2583</td>
      <td>2583</td>
      <td>0.979432</td>
      <td>2705</td>
      <td>0.004407</td>
      <td>2060</td>
      <td>0.003428</td>
    </tr>
    <tr>
      <th>24</th>
      <td>2585</td>
      <td>2585</td>
      <td>0.683367</td>
      <td>2582</td>
      <td>0.086172</td>
      <td>1560</td>
      <td>0.062124</td>
    </tr>
    <tr>
      <th>25</th>
      <td>2705</td>
      <td>2705</td>
      <td>0.806159</td>
      <td>2403</td>
      <td>0.061594</td>
      <td>10</td>
      <td>0.054348</td>
    </tr>
    <tr>
      <th>26</th>
      <td>2905</td>
      <td>2905</td>
      <td>0.994253</td>
      <td>1281</td>
      <td>0.005747</td>
      <td>1920</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
</div>
    


### Rapport de classification TOKENISATION + EMBEDDING


```python
y_orig = emb.get_y_orig()
y_pred = emb.get_y_pred()
_,_ = ds.get_classification_report(y_orig, y_pred)
```

    Précision de la prédiction: 79.70442769665567 %
    Evaluation détaillée de la Classification par RDF :
     
                   precision    recall  f1-score   support
    
              10       0.53      0.33      0.41       623
              40       0.61      0.70      0.65       502
              50       0.83      0.76      0.79       336
              60       0.92      0.84      0.88       166
            1140       0.80      0.68      0.74       534
            1160       0.85      0.88      0.87       791
            1180       0.69      0.57      0.62       153
            1280       0.75      0.61      0.67       974
            1281       0.45      0.66      0.54       414
            1300       0.96      0.93      0.95      1009
            1301       0.89      0.90      0.90       161
            1302       0.88      0.79      0.83       498
            1320       0.82      0.77      0.80       648
            1560       0.79      0.84      0.82      1015
            1920       0.91      0.91      0.91       861
            1940       0.89      0.78      0.83       161
            2060       0.81      0.80      0.80       999
            2220       0.76      0.72      0.74       165
            2280       0.84      0.75      0.79       952
            2403       0.66      0.76      0.71       955
            2462       0.68      0.72      0.70       284
            2522       0.90      0.92      0.91       998
            2582       0.76      0.76      0.76       518
            2583       0.96      0.98      0.97      2042
            2585       0.88      0.68      0.77       499
            2705       0.47      0.81      0.59       552
            2905       0.98      0.99      0.99       174
    
        accuracy                           0.80     16984
       macro avg       0.79      0.77      0.78     16984
    weighted avg       0.81      0.80      0.80     16984
    
    

### Matrice de confusion TOKENISATION + EMBEDDING


```python
ds.show_confusion_matrix(y_orig, y_pred)
```


    
![png](/images/ReadMe_ML_files/output_29_0.png)
    


# Modèle 2 : classe RNN_STEMMER

### Une tokenisation puis une racinisation (stemming) suivie d'une couche d'Embedding de tensorFlow


```python
import Bibli_DataScience_3_3 as ds
import RNN_DataScience as rnn
```


```python
# instanciation du modèle
stem = rnn.RNN_STEMMER("EMBEDDING STEMMER")

```


    Pandas Apply:   0%|          | 0/84916 [00:00<?, ?it/s]


    [nltk_data] Downloading package punkt to /root/nltk_data...
    [nltk_data]   Package punkt is already up-to-date!
    [nltk_data] Downloading package stopwords to /root/nltk_data...
    [nltk_data]   Package stopwords is already up-to-date!
    [nltk_data] Downloading package wordnet to /root/nltk_data...
    [nltk_data]   Package wordnet is already up-to-date!
    


```python
# entrainement du modèle
train_acc,val_acc,tloss,tvalloss = stem.fit_modele(5,True,stemming=True)
```

    <class 'pandas.core.frame.DataFrame'>
    Index: 67932 entries, 83256 to 20596
    Data columns (total 16 columns):
     #   Column          Non-Null Count  Dtype  
    ---  ------          --------------  -----  
     0   Id              67932 non-null  int64  
     1   designation     67932 non-null  object 
     2   description     44084 non-null  object 
     3   productid       67932 non-null  int64  
     4   imageid         67932 non-null  int64  
     5   PAYS_LANGUE     67932 non-null  object 
     6   RATIO_LANGUE    67932 non-null  float64
     7   ORIGINE_LANGUE  67932 non-null  object 
     8   pays_design     67932 non-null  object 
     9   Ratio_design    67932 non-null  float64
     10  pays_descr      44033 non-null  object 
     11  Ratio_descr     44033 non-null  float64
     12  descr_NaN       67932 non-null  bool   
     13  nom_image       67932 non-null  object 
     14  filepath        67932 non-null  object 
     15  phrases         67932 non-null  object 
    dtypes: bool(1), float64(3), int64(3), object(9)
    memory usage: 8.4+ MB
    None
    etape 1/6
    etape 2/6
    etape 3/6
    save y_train_avant.shape  (67932,)
    self.EMBEDDING_DIM 300
    suite
    (67932, 27)
    (16984, 27)
    vocab_size :  112086
    Epoch 1/5
    2123/2123 [==============================] - 1680s 790ms/step - loss: 1.1424 - accuracy: 0.6714 - val_loss: 0.7287 - val_accuracy: 0.7922 - lr: 0.0010
    Epoch 2/5
    2123/2123 [==============================] - 1607s 757ms/step - loss: 0.4548 - accuracy: 0.8675 - val_loss: 0.6938 - val_accuracy: 0.8139 - lr: 0.0010
    Epoch 3/5
    2123/2123 [==============================] - 1615s 761ms/step - loss: 0.2382 - accuracy: 0.9311 - val_loss: 0.7906 - val_accuracy: 0.8068 - lr: 0.0010
    Epoch 4/5
    2123/2123 [==============================] - 1609s 758ms/step - loss: 0.1332 - accuracy: 0.9608 - val_loss: 0.9150 - val_accuracy: 0.8037 - lr: 0.0010
    Epoch 5/5
    2123/2123 [==============================] - 1608s 758ms/step - loss: 0.0955 - accuracy: 0.9725 - val_loss: 1.0060 - val_accuracy: 0.8059 - lr: 0.0010
    531/531 [==============================] - 63s 117ms/step
    2123/2123 [==============================] - 253s 119ms/step
    531/531 [==============================] - 65s 122ms/step
    y_test_original2[:5]  [1301 1140 2583 2280 2403]
    


```python
ds.plot_fit(train_acc,val_acc,tloss,tvalloss)
```


    
![png](/images/ReadMe_ML_files/output_35_0.png)
    


### Tableau des repartitions des **classes prédites** pour chaque **classe réelle**


```python
df_pred = stem.get_df_pred()
df_pred
```





  <div id="df-54f2bc82-6c63-40fb-9a78-bd19b8d93e4a" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Categorie</th>
      <th>predict</th>
      <th>pourc</th>
      <th>predict2</th>
      <th>pourc2</th>
      <th>predict3</th>
      <th>pourc3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>10</td>
      <td>10</td>
      <td>0.306581</td>
      <td>2403</td>
      <td>0.189406</td>
      <td>2705</td>
      <td>0.166934</td>
    </tr>
    <tr>
      <th>1</th>
      <td>40</td>
      <td>40</td>
      <td>0.745020</td>
      <td>50</td>
      <td>0.039841</td>
      <td>1280</td>
      <td>0.033865</td>
    </tr>
    <tr>
      <th>2</th>
      <td>50</td>
      <td>50</td>
      <td>0.818452</td>
      <td>40</td>
      <td>0.050595</td>
      <td>2462</td>
      <td>0.050595</td>
    </tr>
    <tr>
      <th>3</th>
      <td>60</td>
      <td>60</td>
      <td>0.903614</td>
      <td>50</td>
      <td>0.042169</td>
      <td>2462</td>
      <td>0.024096</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1140</td>
      <td>1140</td>
      <td>0.676030</td>
      <td>1280</td>
      <td>0.108614</td>
      <td>40</td>
      <td>0.035581</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1160</td>
      <td>1160</td>
      <td>0.858407</td>
      <td>40</td>
      <td>0.036662</td>
      <td>2705</td>
      <td>0.022756</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1180</td>
      <td>1180</td>
      <td>0.627451</td>
      <td>40</td>
      <td>0.071895</td>
      <td>1280</td>
      <td>0.058824</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1280</td>
      <td>1280</td>
      <td>0.757700</td>
      <td>1281</td>
      <td>0.047228</td>
      <td>1140</td>
      <td>0.041068</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1281</td>
      <td>1281</td>
      <td>0.466184</td>
      <td>1280</td>
      <td>0.212560</td>
      <td>40</td>
      <td>0.053140</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1300</td>
      <td>1300</td>
      <td>0.941526</td>
      <td>1280</td>
      <td>0.020813</td>
      <td>2280</td>
      <td>0.005946</td>
    </tr>
    <tr>
      <th>10</th>
      <td>1301</td>
      <td>1301</td>
      <td>0.906832</td>
      <td>1280</td>
      <td>0.018634</td>
      <td>1302</td>
      <td>0.018634</td>
    </tr>
    <tr>
      <th>11</th>
      <td>1302</td>
      <td>1302</td>
      <td>0.799197</td>
      <td>1280</td>
      <td>0.058233</td>
      <td>2583</td>
      <td>0.032129</td>
    </tr>
    <tr>
      <th>12</th>
      <td>1320</td>
      <td>1320</td>
      <td>0.756173</td>
      <td>1280</td>
      <td>0.050926</td>
      <td>1920</td>
      <td>0.032407</td>
    </tr>
    <tr>
      <th>13</th>
      <td>1560</td>
      <td>1560</td>
      <td>0.830542</td>
      <td>2582</td>
      <td>0.048276</td>
      <td>2060</td>
      <td>0.036453</td>
    </tr>
    <tr>
      <th>14</th>
      <td>1920</td>
      <td>1920</td>
      <td>0.921022</td>
      <td>2060</td>
      <td>0.025552</td>
      <td>1560</td>
      <td>0.024390</td>
    </tr>
    <tr>
      <th>15</th>
      <td>1940</td>
      <td>1940</td>
      <td>0.813665</td>
      <td>2060</td>
      <td>0.024845</td>
      <td>2280</td>
      <td>0.024845</td>
    </tr>
    <tr>
      <th>16</th>
      <td>2060</td>
      <td>2060</td>
      <td>0.797798</td>
      <td>1560</td>
      <td>0.056056</td>
      <td>1920</td>
      <td>0.029029</td>
    </tr>
    <tr>
      <th>17</th>
      <td>2220</td>
      <td>2220</td>
      <td>0.751515</td>
      <td>1320</td>
      <td>0.048485</td>
      <td>50</td>
      <td>0.024242</td>
    </tr>
    <tr>
      <th>18</th>
      <td>2280</td>
      <td>2280</td>
      <td>0.822479</td>
      <td>2403</td>
      <td>0.097689</td>
      <td>10</td>
      <td>0.023109</td>
    </tr>
    <tr>
      <th>19</th>
      <td>2403</td>
      <td>2403</td>
      <td>0.773822</td>
      <td>2280</td>
      <td>0.074346</td>
      <td>10</td>
      <td>0.048168</td>
    </tr>
    <tr>
      <th>20</th>
      <td>2462</td>
      <td>2462</td>
      <td>0.700704</td>
      <td>40</td>
      <td>0.105634</td>
      <td>60</td>
      <td>0.070423</td>
    </tr>
    <tr>
      <th>21</th>
      <td>2522</td>
      <td>2522</td>
      <td>0.928858</td>
      <td>2403</td>
      <td>0.015030</td>
      <td>1560</td>
      <td>0.010020</td>
    </tr>
    <tr>
      <th>22</th>
      <td>2582</td>
      <td>2582</td>
      <td>0.754826</td>
      <td>2060</td>
      <td>0.067568</td>
      <td>1560</td>
      <td>0.044402</td>
    </tr>
    <tr>
      <th>23</th>
      <td>2583</td>
      <td>2583</td>
      <td>0.980901</td>
      <td>1302</td>
      <td>0.005387</td>
      <td>2060</td>
      <td>0.003918</td>
    </tr>
    <tr>
      <th>24</th>
      <td>2585</td>
      <td>2585</td>
      <td>0.739479</td>
      <td>2583</td>
      <td>0.056112</td>
      <td>2582</td>
      <td>0.052104</td>
    </tr>
    <tr>
      <th>25</th>
      <td>2705</td>
      <td>2705</td>
      <td>0.679348</td>
      <td>2403</td>
      <td>0.096014</td>
      <td>10</td>
      <td>0.090580</td>
    </tr>
    <tr>
      <th>26</th>
      <td>2905</td>
      <td>2905</td>
      <td>0.994253</td>
      <td>2583</td>
      <td>0.005747</td>
      <td>1920</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
</div>
    
### Rapport de classification TOKENISATION + STEMMISATION + EMBEDDING


```python
#y_orig = ds.load_ndarray('EMBEDDING STEMMER_y_orig')
#y_pred = ds.load_ndarray('EMBEDDING STEMMER_y_pred')
y_orig = stem.get_y_orig()
y_pred = stem.get_y_pred()
_,_ = ds.get_classification_report(y_orig, y_pred)
```

    Précision de la prédiction: 80.59349976448422 %
    Evaluation détaillée de la Classification par RDF :
     
                   precision    recall  f1-score   support
    
              10       0.55      0.31      0.39       623
              40       0.55      0.75      0.63       502
              50       0.76      0.82      0.79       336
              60       0.74      0.90      0.81       166
            1140       0.80      0.68      0.73       534
            1160       0.90      0.86      0.88       791
            1180       0.55      0.63      0.59       153
            1280       0.68      0.76      0.72       974
            1281       0.65      0.47      0.54       414
            1300       0.94      0.94      0.94      1009
            1301       0.90      0.91      0.90       161
            1302       0.85      0.80      0.82       498
            1320       0.87      0.76      0.81       648
            1560       0.84      0.83      0.84      1015
            1920       0.90      0.92      0.91       861
            1940       0.83      0.81      0.82       161
            2060       0.82      0.80      0.81       999
            2220       0.77      0.75      0.76       165
            2280       0.80      0.82      0.81       952
            2403       0.68      0.77      0.72       955
            2462       0.79      0.70      0.74       284
            2522       0.86      0.93      0.89       998
            2582       0.75      0.75      0.75       518
            2583       0.95      0.98      0.97      2042
            2585       0.87      0.74      0.80       499
            2705       0.61      0.68      0.64       552
            2905       0.98      0.99      0.99       174
    
        accuracy                           0.81     16984
       macro avg       0.78      0.78      0.78     16984
    weighted avg       0.81      0.81      0.80     16984
    
    

### Matrice de confusion TOKENISATION + STEMMISATION + EMBEDDING


```python
ds.show_confusion_matrix(y_orig, y_pred)
```


    
![png](/images/ReadMe_ML_files/output_41_0.png)
    


# Modèle 3 : classe RNN_LEMMER

### Une tokenisation puis une lemmatisation en utilisant NLTK suivie d'une couche d'Embedding de tensorFlow


```python
# instanciation du modèle
lem = rnn.RNN_LEMMER("EMBEDDING LEMMER")
```


    Pandas Apply:   0%|          | 0/84916 [00:00<?, ?it/s]


    [nltk_data] Downloading package punkt to /root/nltk_data...
    [nltk_data]   Unzipping tokenizers/punkt.zip.
    [nltk_data] Downloading package stopwords to /root/nltk_data...
    [nltk_data]   Unzipping corpora/stopwords.zip.
    [nltk_data] Downloading package wordnet to /root/nltk_data...
    


```python
# entrainement du modèle
train_acc,val_acc,tloss,tvalloss = lem.fit_modele(5,True,lemming=True)
```

    <class 'pandas.core.frame.DataFrame'>
    Index: 67932 entries, 83256 to 20596
    Data columns (total 16 columns):
     #   Column          Non-Null Count  Dtype  
    ---  ------          --------------  -----  
     0   Id              67932 non-null  int64  
     1   designation     67932 non-null  object 
     2   description     44084 non-null  object 
     3   productid       67932 non-null  int64  
     4   imageid         67932 non-null  int64  
     5   PAYS_LANGUE     67932 non-null  object 
     6   RATIO_LANGUE    67932 non-null  float64
     7   ORIGINE_LANGUE  67932 non-null  object 
     8   pays_design     67932 non-null  object 
     9   Ratio_design    67932 non-null  float64
     10  pays_descr      44033 non-null  object 
     11  Ratio_descr     44033 non-null  float64
     12  descr_NaN       67932 non-null  bool   
     13  nom_image       67932 non-null  object 
     14  filepath        67932 non-null  object 
     15  phrases         67932 non-null  object 
    dtypes: bool(1), float64(3), int64(3), object(9)
    memory usage: 8.4+ MB
    None
    etape 1/6
    etape 2/6
    etape 3/6
    save y_train_avant.shape  (67932,)
    self.EMBEDDING_DIM 300
    suite
    (67932, 27)
    (16984, 27)
    vocab_size :  112086
    Epoch 1/5
    2123/2123 [==============================] - 1802s 848ms/step - loss: 1.1214 - accuracy: 0.6749 - val_loss: 0.7337 - val_accuracy: 0.7908 - lr: 0.0010
    Epoch 2/5
    2123/2123 [==============================] - 1785s 841ms/step - loss: 0.4579 - accuracy: 0.8673 - val_loss: 0.7026 - val_accuracy: 0.8066 - lr: 0.0010
    Epoch 3/5
    2123/2123 [==============================] - 1792s 844ms/step - loss: 0.2357 - accuracy: 0.9307 - val_loss: 0.8267 - val_accuracy: 0.8041 - lr: 0.0010
    Epoch 4/5
    2123/2123 [==============================] - 1795s 846ms/step - loss: 0.1378 - accuracy: 0.9606 - val_loss: 0.8843 - val_accuracy: 0.8115 - lr: 0.0010
    Epoch 5/5
    2123/2123 [==============================] - 1811s 853ms/step - loss: 0.0930 - accuracy: 0.9732 - val_loss: 1.0278 - val_accuracy: 0.8060 - lr: 0.0010
    531/531 [==============================] - 78s 145ms/step
    2123/2123 [==============================] - 311s 146ms/step
    531/531 [==============================] - 76s 144ms/step
    y_test_original2[:5]  [1301 1140 2583 2280 2403]
    


```python
# RECUPERATION
"""
train_acc,val_acc,tloss,tvalloss = lem.restore_fit_arrays()
y_orig,y_pred = lem.restore_predict_arrays()
df_pred = lem.restore_predict_dataframe()

"""
```


```python
ds.plot_fit(train_acc,val_acc,tloss,tvalloss)
```


    
![png](/images/ReadMe_ML_files/output_46_0.png)
    


### Tableau des repartitions des **classes prédites** pour chaque **classe réelle**


```python
df_pred = lem.get_df_pred()
df_pred
```





  <div id="df-aafd30d1-28dd-44de-b472-fab2f8940eff" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Categorie</th>
      <th>predict</th>
      <th>pourc</th>
      <th>predict2</th>
      <th>pourc2</th>
      <th>predict3</th>
      <th>pourc3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>10</td>
      <td>10</td>
      <td>0.627608</td>
      <td>2705</td>
      <td>0.113965</td>
      <td>2280</td>
      <td>0.060995</td>
    </tr>
    <tr>
      <th>1</th>
      <td>40</td>
      <td>40</td>
      <td>0.701195</td>
      <td>10</td>
      <td>0.073705</td>
      <td>2462</td>
      <td>0.043825</td>
    </tr>
    <tr>
      <th>2</th>
      <td>50</td>
      <td>50</td>
      <td>0.770833</td>
      <td>2462</td>
      <td>0.065476</td>
      <td>40</td>
      <td>0.050595</td>
    </tr>
    <tr>
      <th>3</th>
      <td>60</td>
      <td>60</td>
      <td>0.855422</td>
      <td>2462</td>
      <td>0.072289</td>
      <td>50</td>
      <td>0.030120</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1140</td>
      <td>1140</td>
      <td>0.732210</td>
      <td>1280</td>
      <td>0.069288</td>
      <td>40</td>
      <td>0.037453</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1160</td>
      <td>1160</td>
      <td>0.859671</td>
      <td>10</td>
      <td>0.058154</td>
      <td>40</td>
      <td>0.020228</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1180</td>
      <td>1180</td>
      <td>0.529412</td>
      <td>10</td>
      <td>0.065359</td>
      <td>1140</td>
      <td>0.065359</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1280</td>
      <td>1280</td>
      <td>0.642710</td>
      <td>1281</td>
      <td>0.124230</td>
      <td>1140</td>
      <td>0.065708</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1281</td>
      <td>1281</td>
      <td>0.606280</td>
      <td>1280</td>
      <td>0.082126</td>
      <td>40</td>
      <td>0.045894</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1300</td>
      <td>1300</td>
      <td>0.903865</td>
      <td>1280</td>
      <td>0.029732</td>
      <td>1140</td>
      <td>0.009911</td>
    </tr>
    <tr>
      <th>10</th>
      <td>1301</td>
      <td>1301</td>
      <td>0.900621</td>
      <td>1320</td>
      <td>0.024845</td>
      <td>1281</td>
      <td>0.018634</td>
    </tr>
    <tr>
      <th>11</th>
      <td>1302</td>
      <td>1302</td>
      <td>0.831325</td>
      <td>1281</td>
      <td>0.030120</td>
      <td>2583</td>
      <td>0.024096</td>
    </tr>
    <tr>
      <th>12</th>
      <td>1320</td>
      <td>1320</td>
      <td>0.820988</td>
      <td>1560</td>
      <td>0.024691</td>
      <td>2060</td>
      <td>0.024691</td>
    </tr>
    <tr>
      <th>13</th>
      <td>1560</td>
      <td>1560</td>
      <td>0.838424</td>
      <td>2582</td>
      <td>0.044335</td>
      <td>2060</td>
      <td>0.041379</td>
    </tr>
    <tr>
      <th>14</th>
      <td>1920</td>
      <td>1920</td>
      <td>0.911731</td>
      <td>2060</td>
      <td>0.031359</td>
      <td>1560</td>
      <td>0.020906</td>
    </tr>
    <tr>
      <th>15</th>
      <td>1940</td>
      <td>1940</td>
      <td>0.782609</td>
      <td>10</td>
      <td>0.043478</td>
      <td>1320</td>
      <td>0.024845</td>
    </tr>
    <tr>
      <th>16</th>
      <td>2060</td>
      <td>2060</td>
      <td>0.799800</td>
      <td>1560</td>
      <td>0.070070</td>
      <td>1920</td>
      <td>0.024024</td>
    </tr>
    <tr>
      <th>17</th>
      <td>2220</td>
      <td>2220</td>
      <td>0.678788</td>
      <td>2060</td>
      <td>0.078788</td>
      <td>2585</td>
      <td>0.048485</td>
    </tr>
    <tr>
      <th>18</th>
      <td>2280</td>
      <td>2280</td>
      <td>0.789916</td>
      <td>10</td>
      <td>0.117647</td>
      <td>2403</td>
      <td>0.052521</td>
    </tr>
    <tr>
      <th>19</th>
      <td>2403</td>
      <td>2403</td>
      <td>0.678534</td>
      <td>10</td>
      <td>0.160209</td>
      <td>2280</td>
      <td>0.072251</td>
    </tr>
    <tr>
      <th>20</th>
      <td>2462</td>
      <td>2462</td>
      <td>0.707746</td>
      <td>40</td>
      <td>0.102113</td>
      <td>50</td>
      <td>0.049296</td>
    </tr>
    <tr>
      <th>21</th>
      <td>2522</td>
      <td>2522</td>
      <td>0.898798</td>
      <td>1560</td>
      <td>0.020040</td>
      <td>2403</td>
      <td>0.013026</td>
    </tr>
    <tr>
      <th>22</th>
      <td>2582</td>
      <td>2582</td>
      <td>0.743243</td>
      <td>2060</td>
      <td>0.067568</td>
      <td>2585</td>
      <td>0.063707</td>
    </tr>
    <tr>
      <th>23</th>
      <td>2583</td>
      <td>2583</td>
      <td>0.977473</td>
      <td>1302</td>
      <td>0.004897</td>
      <td>2585</td>
      <td>0.003428</td>
    </tr>
    <tr>
      <th>24</th>
      <td>2585</td>
      <td>2585</td>
      <td>0.807615</td>
      <td>2582</td>
      <td>0.048096</td>
      <td>1560</td>
      <td>0.028056</td>
    </tr>
    <tr>
      <th>25</th>
      <td>2705</td>
      <td>2705</td>
      <td>0.695652</td>
      <td>10</td>
      <td>0.215580</td>
      <td>2403</td>
      <td>0.025362</td>
    </tr>
    <tr>
      <th>26</th>
      <td>2905</td>
      <td>2905</td>
      <td>1.000000</td>
      <td>1920</td>
      <td>0.000000</td>
      <td>2705</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
</div>
    


### Rapport de classification EMBEDDING + LEMMISATION


```python

y_orig = lem.get_y_orig()
y_pred = lem.get_y_pred()
_,_ = ds.get_classification_report(y_orig, y_pred)
```

    Précision de la prédiction: 80.59938765897316 %
    Evaluation détaillée de la Classification par RDF :
     
                   precision    recall  f1-score   support
    
              10       0.41      0.63      0.49       623
              40       0.64      0.70      0.67       502
              50       0.82      0.77      0.79       336
              60       0.87      0.86      0.86       166
            1140       0.69      0.73      0.71       534
            1160       0.89      0.86      0.87       791
            1180       0.65      0.53      0.58       153
            1280       0.76      0.64      0.70       974
            1281       0.51      0.61      0.55       414
            1300       0.99      0.90      0.94      1009
            1301       0.96      0.90      0.93       161
            1302       0.76      0.83      0.79       498
            1320       0.80      0.82      0.81       648
            1560       0.82      0.84      0.83      1015
            1920       0.91      0.91      0.91       861
            1940       0.85      0.78      0.82       161
            2060       0.81      0.80      0.81       999
            2220       0.88      0.68      0.77       165
            2280       0.84      0.79      0.82       952
            2403       0.82      0.68      0.74       955
            2462       0.71      0.71      0.71       284
            2522       0.94      0.90      0.92       998
            2582       0.75      0.74      0.75       518
            2583       0.96      0.98      0.97      2042
            2585       0.77      0.81      0.79       499
            2705       0.69      0.70      0.69       552
            2905       0.93      1.00      0.96       174
    
        accuracy                           0.81     16984
       macro avg       0.79      0.78      0.78     16984
    weighted avg       0.82      0.81      0.81     16984
    
    

### Matrice de confusion  EMBEDDING + LEMMISATION


```python
ds.show_confusion_matrix(y_orig, y_pred)
```


    
![png](/images/ReadMe_ML_files/output_52_0.png)
    


# Modèle 4 : classe RNN_SPACY

### Une tokenisation puis une lemmatisation en utilisant SPACY suivie d'une couche d'Embedding de tensorFlow


```python
import Bibli_DataScience_3_3 as ds
import RNN_DataScience as rnn
```

 

```python
# instanciation du modèle
spacy = rnn.RNN_SPACY("EMBEDDING SPACY")
```


    Pandas Apply:   0%|          | 0/84916 [00:00<?, ?it/s]


    [nltk_data] Downloading package punkt to /root/nltk_data...
    [nltk_data]   Package punkt is already up-to-date!
    [nltk_data] Downloading package stopwords to /root/nltk_data...
    [nltk_data]   Package stopwords is already up-to-date!
    [nltk_data] Downloading package wordnet to /root/nltk_data...
    [nltk_data]   Package wordnet is already up-to-date!
    


```python
# entrainement du modèle
train_acc,val_acc,tloss,tvalloss = spacy.fit_modele(5,True,spacy=True)
```

    <class 'pandas.core.frame.DataFrame'>
    Index: 67932 entries, 83256 to 20596
    Data columns (total 16 columns):
     #   Column          Non-Null Count  Dtype  
    ---  ------          --------------  -----  
     0   Id              67932 non-null  int64  
     1   designation     67932 non-null  object 
     2   description     44084 non-null  object 
     3   productid       67932 non-null  int64  
     4   imageid         67932 non-null  int64  
     5   PAYS_LANGUE     67932 non-null  object 
     6   RATIO_LANGUE    67932 non-null  float64
     7   ORIGINE_LANGUE  67932 non-null  object 
     8   pays_design     67932 non-null  object 
     9   Ratio_design    67932 non-null  float64
     10  pays_descr      44033 non-null  object 
     11  Ratio_descr     44033 non-null  float64
     12  descr_NaN       67932 non-null  bool   
     13  nom_image       67932 non-null  object 
     14  filepath        67932 non-null  object 
     15  phrases         67932 non-null  object 
    dtypes: bool(1), float64(3), int64(3), object(9)
    memory usage: 8.4+ MB
    None
    etape 1/6
    etape 2/6
    etape 3/6
    save y_train_avant.shape  (67932,)
    self.EMBEDDING_DIM 300
    suite
    (67932, 27)
    (16984, 27)
    etape 4/6
    etape 5/6
    etape 6/6
    vocab_size :  111846
    output :  300
    vocab_size =  111846
    Epoch 1/5
    2123/2123 [==============================] - 1848s 869ms/step - loss: 1.1225 - accuracy: 0.6769 - val_loss: 0.7559 - val_accuracy: 0.7773 - lr: 0.0010
    Epoch 2/5
    2123/2123 [==============================] - 1850s 872ms/step - loss: 0.4418 - accuracy: 0.8719 - val_loss: 0.6675 - val_accuracy: 0.8104 - lr: 0.0010
    Epoch 3/5
    2123/2123 [==============================] - 1852s 872ms/step - loss: 0.2220 - accuracy: 0.9363 - val_loss: 0.8601 - val_accuracy: 0.7979 - lr: 0.0010
    Epoch 4/5
    2123/2123 [==============================] - 1845s 869ms/step - loss: 0.1344 - accuracy: 0.9624 - val_loss: 0.9159 - val_accuracy: 0.8035 - lr: 0.0010
    Epoch 5/5
    2123/2123 [==============================] - 1847s 870ms/step - loss: 0.0886 - accuracy: 0.9749 - val_loss: 1.0275 - val_accuracy: 0.8028 - lr: 0.0010
    531/531 [==============================] - 78s 146ms/step
    2123/2123 [==============================] - 309s 146ms/step
    531/531 [==============================] - 77s 145ms/step
    y_test_original2[:5]  [1301 1140 2583 2280 2403]
    


```python
# RECUPERATION
"""
train_acc,val_acc,tloss,tvalloss = spacy.restore_fit_arrays()
y_orig,y_pred = spacy.restore_predict_arrays()
df_pred = spacy.restore_predict_dataframe()

"""

```


```python
ds.plot_fit(train_acc,val_acc,tloss,tvalloss)
```


    
![png](/images/ReadMe_ML_files/output_58_0.png)
    


### Tableau des repartitions des **classes prédites** pour chaque **classe réelle**


```python
df_pred = spacy.get_df_pred()
df_pred
```





  <div id="df-dff2f148-1ed2-4fb9-ac57-d04876197cab" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Categorie</th>
      <th>predict</th>
      <th>pourc</th>
      <th>predict2</th>
      <th>pourc2</th>
      <th>predict3</th>
      <th>pourc3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>10</td>
      <td>10</td>
      <td>0.600321</td>
      <td>2705</td>
      <td>0.110754</td>
      <td>2403</td>
      <td>0.072231</td>
    </tr>
    <tr>
      <th>1</th>
      <td>40</td>
      <td>40</td>
      <td>0.645418</td>
      <td>1140</td>
      <td>0.057769</td>
      <td>2462</td>
      <td>0.051793</td>
    </tr>
    <tr>
      <th>2</th>
      <td>50</td>
      <td>50</td>
      <td>0.747024</td>
      <td>2462</td>
      <td>0.068452</td>
      <td>1140</td>
      <td>0.050595</td>
    </tr>
    <tr>
      <th>3</th>
      <td>60</td>
      <td>60</td>
      <td>0.789157</td>
      <td>2462</td>
      <td>0.090361</td>
      <td>50</td>
      <td>0.072289</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1140</td>
      <td>1140</td>
      <td>0.784644</td>
      <td>1280</td>
      <td>0.073034</td>
      <td>1180</td>
      <td>0.028090</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1160</td>
      <td>1160</td>
      <td>0.867257</td>
      <td>10</td>
      <td>0.032870</td>
      <td>1140</td>
      <td>0.030341</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1180</td>
      <td>1180</td>
      <td>0.594771</td>
      <td>1140</td>
      <td>0.117647</td>
      <td>10</td>
      <td>0.065359</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1280</td>
      <td>1280</td>
      <td>0.723819</td>
      <td>1281</td>
      <td>0.074949</td>
      <td>1140</td>
      <td>0.070842</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1281</td>
      <td>1281</td>
      <td>0.550725</td>
      <td>1280</td>
      <td>0.135266</td>
      <td>1302</td>
      <td>0.043478</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1300</td>
      <td>1300</td>
      <td>0.917740</td>
      <td>1280</td>
      <td>0.031715</td>
      <td>1140</td>
      <td>0.011893</td>
    </tr>
    <tr>
      <th>10</th>
      <td>1301</td>
      <td>1301</td>
      <td>0.869565</td>
      <td>1280</td>
      <td>0.024845</td>
      <td>1320</td>
      <td>0.024845</td>
    </tr>
    <tr>
      <th>11</th>
      <td>1302</td>
      <td>1302</td>
      <td>0.825301</td>
      <td>1280</td>
      <td>0.036145</td>
      <td>2583</td>
      <td>0.028112</td>
    </tr>
    <tr>
      <th>12</th>
      <td>1320</td>
      <td>1320</td>
      <td>0.816358</td>
      <td>1280</td>
      <td>0.037037</td>
      <td>1560</td>
      <td>0.020062</td>
    </tr>
    <tr>
      <th>13</th>
      <td>1560</td>
      <td>1560</td>
      <td>0.855172</td>
      <td>2582</td>
      <td>0.039409</td>
      <td>2060</td>
      <td>0.020690</td>
    </tr>
    <tr>
      <th>14</th>
      <td>1920</td>
      <td>1920</td>
      <td>0.867596</td>
      <td>1560</td>
      <td>0.058072</td>
      <td>1320</td>
      <td>0.023229</td>
    </tr>
    <tr>
      <th>15</th>
      <td>1940</td>
      <td>1940</td>
      <td>0.801242</td>
      <td>10</td>
      <td>0.037267</td>
      <td>1320</td>
      <td>0.031056</td>
    </tr>
    <tr>
      <th>16</th>
      <td>2060</td>
      <td>2060</td>
      <td>0.694695</td>
      <td>1560</td>
      <td>0.114114</td>
      <td>1320</td>
      <td>0.037037</td>
    </tr>
    <tr>
      <th>17</th>
      <td>2220</td>
      <td>2220</td>
      <td>0.812121</td>
      <td>1320</td>
      <td>0.042424</td>
      <td>2583</td>
      <td>0.024242</td>
    </tr>
    <tr>
      <th>18</th>
      <td>2280</td>
      <td>2280</td>
      <td>0.817227</td>
      <td>10</td>
      <td>0.090336</td>
      <td>2403</td>
      <td>0.055672</td>
    </tr>
    <tr>
      <th>19</th>
      <td>2403</td>
      <td>2403</td>
      <td>0.712042</td>
      <td>10</td>
      <td>0.114136</td>
      <td>2280</td>
      <td>0.074346</td>
    </tr>
    <tr>
      <th>20</th>
      <td>2462</td>
      <td>2462</td>
      <td>0.735915</td>
      <td>40</td>
      <td>0.070423</td>
      <td>50</td>
      <td>0.049296</td>
    </tr>
    <tr>
      <th>21</th>
      <td>2522</td>
      <td>2522</td>
      <td>0.892786</td>
      <td>1560</td>
      <td>0.026052</td>
      <td>2403</td>
      <td>0.025050</td>
    </tr>
    <tr>
      <th>22</th>
      <td>2582</td>
      <td>2582</td>
      <td>0.725869</td>
      <td>1560</td>
      <td>0.098456</td>
      <td>2585</td>
      <td>0.044402</td>
    </tr>
    <tr>
      <th>23</th>
      <td>2583</td>
      <td>2583</td>
      <td>0.978942</td>
      <td>1302</td>
      <td>0.007346</td>
      <td>1320</td>
      <td>0.002938</td>
    </tr>
    <tr>
      <th>24</th>
      <td>2585</td>
      <td>2585</td>
      <td>0.753507</td>
      <td>2583</td>
      <td>0.050100</td>
      <td>2582</td>
      <td>0.046092</td>
    </tr>
    <tr>
      <th>25</th>
      <td>2705</td>
      <td>2705</td>
      <td>0.666667</td>
      <td>10</td>
      <td>0.215580</td>
      <td>2403</td>
      <td>0.030797</td>
    </tr>
    <tr>
      <th>26</th>
      <td>2905</td>
      <td>2905</td>
      <td>0.982759</td>
      <td>1180</td>
      <td>0.005747</td>
      <td>1281</td>
      <td>0.005747</td>
    </tr>
  </tbody>
</table>
</div>
    


### Rapport de classification EMBEDDING + SPACY


```python

y_orig = spacy.get_y_orig()
y_pred = spacy.get_y_pred()
_,_ = ds.get_classification_report(y_orig, y_pred)
```

    Précision de la prédiction: 80.2814413565709 %
    Evaluation détaillée de la Classification par RDF :
     
                   precision    recall  f1-score   support
    
              10       0.45      0.60      0.51       623
              40       0.72      0.65      0.68       502
              50       0.81      0.75      0.78       336
              60       0.96      0.79      0.86       166
            1140       0.58      0.78      0.67       534
            1160       0.88      0.87      0.87       791
            1180       0.51      0.59      0.55       153
            1280       0.71      0.72      0.72       974
            1281       0.60      0.55      0.57       414
            1300       0.98      0.92      0.95      1009
            1301       0.95      0.87      0.91       161
            1302       0.74      0.83      0.78       498
            1320       0.77      0.82      0.79       648
            1560       0.74      0.86      0.80      1015
            1920       0.94      0.87      0.90       861
            1940       0.87      0.80      0.83       161
            2060       0.89      0.69      0.78       999
            2220       0.69      0.81      0.75       165
            2280       0.84      0.82      0.83       952
            2403       0.78      0.71      0.74       955
            2462       0.71      0.74      0.72       284
            2522       0.93      0.89      0.91       998
            2582       0.79      0.73      0.76       518
            2583       0.96      0.98      0.97      2042
            2585       0.82      0.75      0.78       499
            2705       0.68      0.67      0.67       552
            2905       0.99      0.98      0.99       174
    
        accuracy                           0.80     16984
       macro avg       0.79      0.78      0.78     16984
    weighted avg       0.81      0.80      0.81     16984
    
    

### Matrice de confusion  EMBEDDING + SPACY


```python
ds.show_confusion_matrix(y_orig, y_pred)
```


    
![png](/images/ReadMe_ML_files/output_64_0.png)
    



```python

```


```python

```


```python

```

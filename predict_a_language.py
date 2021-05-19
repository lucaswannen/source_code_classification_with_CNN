# Ceci est un script permettant de prédire la langage d'un script contenu dans un fichier.


path_model = 'models/model_final.h5' #Chemin du modèle sauvegardé

#Imports



import numpy as np
import matplotlib.pyplot as plt
import cv2
from os import listdir
from os.path import isfile, join
import random
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import MultiLabelBinarizer
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from sklearn.preprocessing import MultiLabelBinarizer
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf
import string
import warnings
import sys, getopt
from scipy.special import softmax

#Fonction utiles

def keep_n_first_caracters(scripts,n): #garde les n premiers caractères de chaque script
    res = []
    for script in scripts:
        res.append(script[:n])
    return res

def oneHotEncoding(samples): #Fonction permettant le oneHotEncoding adéquat
    characters = string.printable  # All printable ASCII characters.
    token_index = dict(zip(characters, range(1, len(characters) + 1)))
    max_length = 1024
    results = np.zeros((len(samples), max_length, max(token_index.values()) + 1))
    for i, sample in enumerate(samples):
        for j, character in enumerate(sample[:max_length]):
            index = token_index.get(character)
            results[i, j, index] = 1.
    return results

def preprocessing_script(script): #Fonction permettant d'effectuer le preprocessing d'un script. Entrée de la forme : ['le_script']
    n = 1024
    if len(script[0])<n: #Retourne une erreur si le script est de taille inférieure à 1024
        warnings.warn("Le script fait moins de 1024 caractères : il faut essayer avec un script plus long !")
    else:
        script_cut = keep_n_first_caracters(script, n)
        script_encoded = oneHotEncoding(script_cut)
        res = script_encoded.reshape(script_encoded.shape[0],script_encoded.shape[1],script_encoded.shape[2],1)
        print()
        return res

def decoder_sortie(output): #Interprète la sortie de manière à fournir une chaine de caractère
    max_index = np.argmax(output[0], axis=0)
    if(max_index==0):
        return "C"
    if(max_index==1):
        return "html"
    if(max_index==2):
        return "java"
    if(max_index==3):
        return "python"


def predire(script, retourner_probas = False):  # Utilise les fonctions précédentes pour effectuer une prédiction de A à Z.
                                                # Entrée de la forme ['un_script']
                                                # mettre retourner_probas à True si on veut avoir la distribution de probabilités
    script_endoded = preprocessing_script(script)
    output =model.predict(script_endoded)
    res = decoder_sortie(output)
    if retourner_probas==True:
        return output
    return res


def lire_fichier(path): #Lit et stocke le contenu d'un fichier dans une variable
    file = open(path, "r")
    script = file.read()
    file.close()
    return script


## Execution


model = keras.models.load_model(path_model)

fichier_a_predire = sys.argv[1]
script = lire_fichier(fichier_a_predire)
prediction = predire([script])
sortie_brut = predire([script], retourner_probas=True)
probabilités = softmax(sortie_brut) 

#Affichage
print('\n\n')
print("/////////////////// Bienvenue dans notre prédicteur de langage de programmation ! //////////////\n")

if len(script)<1024: # Traite le cas d'un script trop court
    print("Votre script ne contient que",len(script),"caractères. Notre prédicteur ne prend que en entrée des scripts de taille supérieure ou égale à 1024 caractères. Veuillez réessayer avec un script plus long")

else :
    print("Pour le fichier",sys.argv[1],'le langage de programmation prédit est :',prediction, '\n')
    languages = ["C","html","java","python"]
    for i in range(len(probabilités[0])):
        print("La probabilité pour que le code soit du",languages[i],"est de",probabilités[0][i]*100,"%")

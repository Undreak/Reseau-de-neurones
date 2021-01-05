import numpy as np
import matplotlib.pyplot as plt
from emnist import extract_training_samples
from emnist import extract_test_samples
import time

images_balanced, labels_balanced = extract_training_samples('balanced')

start = time.time()
def learn(mat):
    # cette fonction permet d'apprendre une lettre
    # plus un pixel sera present et plus sa valeur sera elevée
    # à l'inverse moins un pixel sera present et plus sa valeur sera faible
    for i in range(im_size):
        for j in range(im_size):
            if mat[i,j] == 1:
                # on regarde si un pixel est present et si oui alors on augmente sa valeur de 1
                mat[i,j] += 1
            else:
                # sinon on diminue sa valeur de 1
                mat[i,j] += -1
    return mat

def readim(im):
    # cette fonction permet de lire une image de taille im_size x im_size et de la transformer en image binaire
    th = 128    # valeur seuil
    im_bool = (im > th)     # test booléen pour ne garder que les valeurs au dessus d'une valeur seuil 
    mat = np.zeros((im_size, im_size))  # on definit la matrice binaire de sortie
    for i in range(im_size):
        for j in range(im_size):
            # on scanne toute l'image pour regarder si un élément de matrice est True ou False 
            # puis on lui attribut une valeur: True => 1 et False => 0
            if im_bool[i,j]:
                mat[i,j] = 1
            else:
                mat[i,j] = 0
    return mat

def Q(image):
    # cette fonction permet de calculer le quotient de reconnaissance d'une image 
    # et de determiner de quelle lettre il s'agit
    I = image
    phik = np.zeros(k)  # score candidat
    muk = np.zeros(k)   # score idéal du modèle de poids
    Qk = np.zeros(k)    # quotient de reconnaissance

    for i in range(k):
        # on calcule ce quotient pour chacune des lettres à reconnaître
        for j in range(im_size):
            for l in range(im_size):
                # on calcule le produit matriciel du poid Wk et de l'image I
                phik[i] += Wk[i][j][l]*I[j][l]
                if Wk[i][j][l] > 0:
                    # on calcule la somme de tous les êlêments positif de Wk
                    muk[i] += Wk[i][j][l]
        # on calcule le quotient de reconnaissance pour une lettre k
        Qk[i] = phik[i]/muk[i]
    return np.argmax(Qk)    # renvoie la lettre avec le plus haut quotient de reconnaissance


# alphabet utilisé, les lettres trop similaire entre majuscule/minuscule sont consideré comme etant la même lettre
alphabet = np.array(['A','B','C','D','E','F','G','H','I','J','K','L','M',
                    'N','O','P','Q','R','S','T','U','V','W','X','Y','Z',
                    'a','b','d','e','f','g','h','n','q','r','t'])

num_images = len(labels_balanced)   # nombre d'images utilisé pour l'apprentissage, on utilise toutes les images disponible
im_size = 28    # taille des images
k = 37  # nombres de caractères  reconnaitre
Wk = np.zeros((k,im_size,im_size))  # initialisation de la matrice de poids
for i in range(num_images):
    # ici on calcule la matrice de poids pour chacune des lettres en utilisant le dataset balanced 
    # on ignore les labels < 9 car se sont des chiffres dans ce dataset
    if labels_balanced[i] >= 9:
        Wk[int(labels_balanced[i]) - 10] += learn(readim(images_balanced[i]))
    
end = time.time()
print(f"Temps d'execution du programme: {end - start} s\n")

images_balanced_test, labels_balanced_test = extract_test_samples('balanced')

start = time.time()

# Test verifiant les performances du réseau de neurones
num_images_test = len(labels_balanced_test)  # nombre d'images utilisé pour le test, on utilise toutes les images disponible
Qstat = 0   # statistique nous donnant le pourcentage de reussite du modèle
Qkstat = np.zeros(k)    # la même mais pour chacune des lettres individuelle
kk = np.zeros(k)    # nombre permettant de normaliser Qkstat
Qkreal = np.zeros((k,num_images_test)) - 1  # statistique mémorisant les choix de Q(images) pour voir la répartition
Nlettres = 0    # on compte le nombre de lettres testé

for i in range(int(num_images_test - 1)):
    if labels_balanced_test[i] >= 9:   
    # on ignore les valeurs < 9 car se sont des chiffres dans ce dataset
        if Q(images_balanced_test[i]) == int(labels_balanced_test[i] - 10): 
            # on regarde si le resultat de Q(image) est correct et correspond au bon label
            # on incrémente Qstat et Qkstat a chaque fois que Q(images) est correct
            Qstat += 1
            Qkstat[int(labels_balanced_test[i] - 10)] += 1
        kk[int(labels_balanced_test[i] - 10)] += 1  # on calcule la norme de Qkstat
        Qkreal[int(labels_balanced_test[i] - 10)][i] = Q(images_balanced_test[i])
        # on mémorise le label renvoyé par Q(image) dans le bon label de Qkreal
        Nlettres += 1
        
end = time.time()
print(f"Temps d'execution du programme: {end - start} s\n")

# AFFICHAGE ET ÉCRITURE DES RESULTATS
print(Qstat/Nlettres)
plt.bar(alphabet,Qkstat/kk)
plt.ylabel('Q')
plt.title('Quotient de reconnaissance par lettre')
plt.savefig('images/Qkstat.png')
plt.clf()

for i in range(k):
    y = np.zeros(k)
    for j in range(len(Qkreal[i])):
        if Qkreal[i][j] >= 0:
            y[int(Qkreal[i][j])] += 1
    plt.bar(alphabet,y)
    plt.title(alphabet[i])
    plt.savefig('images/' + alphabet[i] + '.png')
    plt.close()
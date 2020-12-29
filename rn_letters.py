import numpy as np
import matplotlib.pyplot as plt
from emnist import extract_training_samples
from emnist import extract_test_samples
import time

images_letters, labels_letters = extract_training_samples('letters')

start = time.time()
def learn(mat):
    # cette fonction permet d'apprendre une lettre
    # plus un pixel sera present et plus sa valeur sera elevee
    # a l'inverse moins un pixel sera present et plus sa valeur sera faibles voir meme negative
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
    # cette fonction permet de lire une image de taille im_size x im_size et de la binarizer
    th = 128    #valeur seuil
    im_bool = (im > th)     # test boolean pour ne garder que les valeurs au dessus d'une valeur seuil 
    mat = np.zeros((im_size, im_size))  # on definit la matrice binaire de sortie
    for i in range(im_size):
        for j in range(im_size):
            # on scan toute l'image pour regarder si un element de matrice est True ou False 
            # puis on lui attribut une valeur: True => 1 et False => 0
            if im_bool[i,j]:
                mat[i,j] = 1
            else:
                mat[i,j] = 0
    return mat

def Q(image):
    # cette fonction permet de calculer le score d'une image et de determiner de quelle lettre il s'agit
    I = image
    phik = np.zeros(k)  # score candidat
    muk = np.zeros(k)   # score idéal du modèle de poids
    Qk = np.zeros(k)    # quotient de reconnaissance

    for i in range(k):
        # on calcul ce quotient pour chacune des lettre a reconnaitre
        for j in range(im_size):
            for l in range(im_size):
                # on scan l'image pour calculer le produit matricielle du poid Wk et de l'image I
                phik[i] += Wk[i][j][l]*I[j][l]
                if Wk[i][j][l] > 0:
                    # on calcule la somme de tous les elements positif de Wk
                    muk[i] += Wk[i][j][l]
        # on calcul le quotient de reconnaire pour une lettre k
        Qk[i] = phik[i]/muk[i]
    return np.argmax(Qk)    # renvoie la lettre avec le plus haut quotient de reconnaissance


# alphabet utilise, les lettres trop similaire entre majuscule/minuscule sont considere comme etant une seule lettre
alphabet = np.array(['A','B','C','D','E','F','G','H','I','J','K','L','M',
                    'N','O','P','Q','R','S','T','U','V','W','X','Y','Z'])

num_images = 100000
im_size = 28
k = 26
Qkreal = np.zeros((k,num_images))  # statistique memorisant les choix de Q(images) pour identifier les erreurs

Wk = np.zeros((k,im_size,im_size))
for i in range(num_images):
    # ici on calcul la matrice de poid pour chacune des lettres en utilisant le dataset balanced 
    # on ignore les valeurs < 9 car ceux sont des chiffres dans ce dataset
    Wk[int(labels_letters[i]) - 1] += learn(readim(images_letters[i]))
    
end = time.time()
print(f"Temps d'execution du programme: {end - start} s\n")

images_letters_test, labels_letters_test = extract_test_samples('letters')

num_images_test = 20800  # Test verifiant si le reseau de neuronnes est performant
Qstat = 0   # statistique nous donnant le pourcentage de reussite du modele
Qkstat = np.zeros(k)    # la meme mais pour chacune des lettres individuelle
kk = np.zeros(k)    # nombre permettant de normaliser Qkstat
Qkreal = np.zeros((k,num_images_test))  # statistique memorisant les choix de Q(images) pour identifier les erreurs

for i in range(int(num_images_test - 1)):
    if Q(images_letters_test[i]) == int(labels_letters_test[i] - 1): 
        # on regarde si le resultat de Q(image) est correct et correspond au bon label
        # on incremente Qstat/Qkstat a chaque fois que Q(images) est correct
        Qstat += 1
        Qkstat[int(labels_letters_test[i] - 1)] += 1
    kk[int(labels_letters_test[i] - 1)] += 1  # on calcule la norme de Qkstat
    Qkreal[Q(images_letters_test[i])][i] = labels_letters_test[i] 
        # on memorise le bon label de la lettre que Q(image) pense etre le bon resultat

end = time.time()
print(f"Temps d'execution du programme: {end - start} s")

# AFFICHAGE DES RESULTATS
print(Qstat/num_images_test)
plt.bar(alphabet,Qkstat/kk)
plt.ylabel('Q')
plt.title('taux de reconnaissance par lettre')
plt.savefig('images/Qkstat_letters.png')
plt.clf()

xk = np.zeros(k)
for i in range(len(Qkreal)):
    for j in range(len((Qkreal[i]))):      
        if Qkreal[i][j] > 0:
            xk[i] += 1

for i in range(k):
    n = 0
    y = np.zeros(int(xk[i]))
    for j in range(len(Qkreal[i])):
        if Qkreal[i][j] > 0:
            y[n] = Qkreal[i][j]
            n += 1
    N, bins, pathces = plt.hist(y,bins=k)
    plt.xticks(np.arange(1,1+k),alphabet)
    plt.title(alphabet[i])
    plt.savefig('images/' + alphabet[i] + '_letters.png')
    plt.close()
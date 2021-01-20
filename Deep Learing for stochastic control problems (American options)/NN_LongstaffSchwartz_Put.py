from numpy import *
import scipy.interpolate as scint
import matplotlib.pyplot as plt
from copy import deepcopy
from numpy.polynomial.polynomial import *
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
n = 10000#nombre d'iter
T = 1#Maturité
d = 10#Nb de divisions
s0 = 1#Prix initial
r = 0.1#Taux d'intérêt continu
#Pour les simulations avec Premia, remplacer r par r'=100*(exp(r)-1)
vol = 0.2#volatilite
K = 1#strike

#---Partie création de prix aléatoires---

def genPrice(n, T, d, s0, r, vol):
    p = [s0 for i in range(n)]
    P = deepcopy([p])
    dt = T/(d-1)
    for i in range(1, d):
        for j in range(n):
            p[j] = p[j] * exp( (r-0.5*vol**2)*dt + vol*sqrt(dt)*random.normal() )
        P.append(deepcopy(p))
    return array(P)

#---Partie LS avec réseau de neurones---

def build_model( loss_fun  = "mse", learning_rate = 0.01):
    model = Sequential()
    model.add(Dense(units=100, activation='relu',input_shape=[1]))
    model.add(Dense(units=60, activation='softmax'))
    model.add(Dense(units=1, activation='elu'))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    model.compile(loss=loss_fun,
              optimizer=optimizer,
              metrics=['mae'])
    
    """
    model = keras.Sequential([
    keras.layers.Dense(60, activation=tf.nn.tanh,input_shape=[1]),
    keras.layers.Dense(60, activation=tf.nn.tanh),
    keras.layers.Dense(1, activation=tf.nn.elu)
    ])

    optimizer = tf.train.GradientDescentOptimizer(learning_rate)

    model.compile(loss=loss_fun,
                  optimizer=optimizer,
                  metrics=['mae'])
    """
    return model

def NNLSput(n, T, d, s0, r, vol, K, S = []):
    """Longstaff-Schwartz avec réseaux de neurones pour un put"""
    dt = T/(d-1)
    if S == []:
        S = genPrice(n, T, d, s0, r, vol) #Creation de prix random
    P = [ max(K-S[d-1][i], 0) for i in range(n)]#Stockage du meilleur gain
    tho1=[d-1 for i in range(n)] #stocker les temps optimaux d'exercice pour chaque trajectoire selon NNLS
    NNList=[] #Neural Network List
    Train_datas=[[] for i in range(d-2)] 
    Train_labels=[[] for i in range(d-2)]
    Predictions=[[] for i in range(d-2)]
    for t in range(d-2, 0, -1):
        toLook = [i for i in range(n) if (K - S[t][i]) > 0]#qui fait du profit?#Seulement pour 1/10 des trajectoires
        if toLook!=[]:
            train_data = [S[t][i] for i in toLook[:int(n/10)]]
            train_label = [exp(-r*dt)*P[i] for i in toLook[:int(n/10)]]
            model = build_model( "mse", 0.01)
            history = model.fit(train_data, train_label, epochs=200, validation_split = 0.1)
            NNList.append(history)
            for i in toLook:#Correction du tableau P
                Train_datas[t-1].append(S[t][i])
                Train_labels[t-1].append(exp(-r*dt)*P[i])
                Predictions[t-1].append(float(model.predict([S[t][i]])))
                if [K-S[t][i]] > model.predict([S[t][i]]):
                    P[i] = K-S[t][i]#On a trouve un nouveau payoff optimal
                    tho1[i]=t
                else:
                    P[i] = P[i]*exp(-r*dt)
                    #On retire taux d'intérêt pour la comparaison suivante
    return (1/n*exp(-r*dt)*sum(P) , tho1, Train_datas, Train_labels, Predictions, NNList)


A=NNLSput(n, T, d, s0, r, vol, K, S = [])
print(A[0])
#---Plotting the Neural Network Errors as a function of the epoch:

# for i in range(d-2):
#     V = A[-1][i].history["val_loss"]
#     L = A[-1][i].history["loss"]    
#     E=list(range(len(V)))
#     plt.subplot(4,2,i+1)
#     plt.plot(E, V, label = "Perte sur val ")
#     plt.plot(E, L, label = "Perte sur train ")
#     plt.xlabel("Nb d'epochs")
#     plt.legend()
for i in range(d-2):
    plt.subplot(4,2,i+1)
    plt.plot(A[2][i],A[3][i],label="Vraie valeur")
    plt.plot(A[2][i],A[4][i],label="Valeur prédite")
    plt.legend()
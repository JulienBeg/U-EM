import numpy as np
from numpy.linalg import norm

import scipy
from scipy.optimize import minimize
import scipy.io

import matplotlib.pyplot as plt
from matplotlib import cm

from time import time
import os

binary_repr_vec = np.vectorize(np.binary_repr)
hw = np.char.count

sigma_a = float(input("Quel bruit epistemique ?"))
print("Bruit de : ",sigma_a)

"""
This computes the 4-bits PRESENT Substitution Box.
Input:
    b    : array of uint8 between 0 and 15
Output:
    S(b) : array of corresponding image by the PRESENT Subs
"""
def Sbox_(b):

    # PRESENT SBox
    S = np.array([0xC, 0x5, 0x6, 0xB, 0x9, 0x0, 0xA, 0xD, 0x3, 0xE, 0xF, 0x8, 0x4, 0x7, 0x1, 0x2],dtype=np.uint8)
    return S[b]
Sbox = np.vectorize(Sbox_)

#List of possible masks
M = np.arange(2**4,dtype = np.uint8)

# Number of bits targeted
N = 4

"""
This implements the so called second order corelation power analysis (20-CPA)
    - It considers the centered product of the traces a combination function
    - The model used asumme that the two shares leaks the HW of the mask and of the Sbox output.

Input:
    - T       : the plaintexts used for the considered Traces
    - Y       : the two shares of the Leakage
Output:
    - k_guess : key hypothesis ranked in deacreasing pearson correlation (from most likely to least likely)
"""
def HO_CPA(T,Y):

    Q = len(T)
    Text,Mask = np.meshgrid(T, M, copy=False, sparse=True)

    ybar = np.reshape(np.mean(Y,axis=1),(2,1))
    Yp = np.prod(Y - ybar,axis=0) #Belong to R^Q

    Pearsons = np.zeros(2 ** N)

    mu_y = np.mean(Yp) #Precompute the mean
    for k in range(2**N):

        X = (hw(binary_repr_vec(Sbox(k^Text)^Mask),'1')-N/2)*(hw(binary_repr_vec(Mask),'1')-N/2)
        mu_x = np.mean(X)
        std_x = np.std(X)

        if std_x == 0:
            print("Warn on cpa with 0 variance")
            return np.arange(16)

        mu_y = np.mean(Yp)
        cov = np.mean((Yp-mu_y )* (X-mu_x))
        Pearsons[k] = abs(cov) / std_x

    k_guess = np.argsort(-Pearsons)
    return k_guess

"""
Subfonction used in the maximization step of the U-EM with HW leakage model.

Input:
    beta : posterior probabilities for each masks and keys
    X    : sensitive varioables of the model
    Y    : leakage
    Q    : number of traces

Output;
    na,b : parameters that maximizes the goodness of fit
"""
def minimize_u(beta,X,Y,Q):

    x_tilde = np.sum(beta * X, axis = 0).T #Should belong to R^Q
    x_bar = np.mean(x_tilde,axis = 0) #Should belong to R

    var_x = (np.sum(beta * (X-x_bar) ** 2) / Q)
    cov = np.sum(beta * (X-x_bar) *Y) / Q

    na = cov / var_x
    b = - na * x_bar

    return na,b

"""
This implements the U-EM-HAM algorithm described in the article.

Input:
    T         : the plaintext used for the considered Traces
    Y         : the two shares of the leakage
    sigma     : the noise level
    tolerance : a threshold to decide when to stopp the while loop

Output:
    k_guess   : the key hypothesis ranked by deacreasing goodness of fit with the model
"""
def EM_bivarie_2(T,Y,sigma=np.reshape(np.array([1,1]),(2,1)),N=4,tolerance=10**-10):

    Q = len(T)

    #Normalisation and centering
    Y/=(2 ** .5 * sigma)

    #Normalize and center the traces
    sigma_y = np.std(Y,axis=1)
    y_bar = np.reshape(np.mean(Y,axis=1),(2,1))
    Y-=y_bar
    Y = np.reshape(Y,(2,1,Q))

    #To store the log-likelihood of each key hypothesis
    LL = np.zeros(2**N)

    Text,Mask = np.meshgrid(T, M, copy=False, sparse=True)

    X1 = np.reshape(np.char.count(binary_repr_vec(Mask), '1'),(len(Mask),1))

    for k in range(2**N):

        a1,b1 = np.abs( sigma_y[0] ** 2 - sigma[0] **2) ** .5 * np.sqrt(4/N),0
        a2,b2 = np.abs( sigma_y[1] ** 2 - sigma[1] **2) ** .5 * np.sqrt(4/N),0
        X0 = np.char.count(binary_repr_vec(Sbox(k^Text)^Mask), '1')

        stop = False
        compteur = 0
        while not stop:

            compteur += 1

            #The E Step
            Norm =(Y[0]-a1*X0-b1)**2 + (Y[1]-a2*X1-b2)**2

            C = np.min(Norm,axis=0) #For numerical stabiblity of the exponnetial
            beta = np.exp(C-Norm)
            S  = np.sum(beta,axis=0) #Normalisation coefficient
            beta = beta / S #Normalised p.m.f

            #The M Step
            na1,b1 = minimize_u(beta,X0,Y[0],Q)
            na2,b2 = minimize_u(beta,X1,Y[1],Q)

            #Does the new values of a and b changed up to a certain tolerance level ?
            stop = (abs(na1-a1) + abs(na2-a2) < tolerance) or (compteur > 25)

            a1,a2 = na1,na2

        LL[k] = np.sum(np.log(S)) - np.sum(C) #Store the log-likelihood of the key

    k_guess = np.argsort(-LL)
    return k_guess


"""
Subfonction used in the maximization step of the U-EM with Linear leakage model.

Input:
    beta : posterior probabilities for each masks and keys
    X    : sensitive varioables of the model
    Y    : leakage
    Q    : number of traces
    Mask : masks for sensitive  variable
    N    : Number of bits targeted

Output;
    na,nb: parameters that maximizes the goodness of fit
"""
def minimize(beta,X,Y,Mask,Q,N):

    x_tilde = np.sum(np.transpose(X,axes=(2,0,1)) * beta, axis = 1).T #Should belong to R^(QxN)
    x_bar = np.mean(x_tilde,axis = 0) #Should belong to R^N

    X_cent = np.reshape(X - x_bar,(len(Mask),Q,N,1))
    #X_cent = X - x_bar

    AutoCorr = np.einsum('mq,mqtp->tp' , beta,np.einsum('ijkl,ijhl->ijkh', X_cent,X_cent))
    RelCorr = np.einsum('qm,q->m' ,  x_tilde ,Y) #Since Y is centered no need to sub x_bar to x_tilde

    #na = np.linalg.solve(AutoCorr,RelCorr)
    na = np.linalg.lstsq(AutoCorr,RelCorr,rcond=0)[0]
    nb = - np.dot(x_bar,na)

    return na,nb

"""
This implements the U-EM-LIN algorithm described in the article.

Input:
    T         : the plaintext used for the considered Traces
    Y         : the two shares of the leakage
    sigma     : the noise level
    tolerance : a threshold to decide when to stopp the while loop
    N         : number of bits targeted

Output:
    k_guess   : the key hypothesis ranked by deacreasing goodness of fit with the model
"""
def EM_BIV_LIN2(T,Y,sigma=np.reshape(np.array([1,1]),(2,1)),N=4,tolerance=10**-8):

    Q = len(T)

    Y/=(2 ** .5 * sigma)
    y_bar = np.reshape(np.mean(Y,axis=1),(2,1)) #Precompute the average of the traces
    Y-=y_bar

    LL = np.zeros(2**N) #To store the log-likelihood of each key hypothesis

    Text,Mask = np.uint8(np.meshgrid(T, M))

    X2 = np.reshape(np.unpackbits(Mask),(2**N,Q,8))[:,:,8-N:8]
    sigma_y = np.std(Y,axis=1)

    for k in range(2**N):

        a_1,b_1 = np.ones(N) * np.abs( (np.std(Y,axis=1)[0]/Q) ** 2 - sigma[0] **2) ** .5 * np.sqrt(N/4),0
        a_2,b_2 = np.ones(N) * np.abs( (np.std(Y,axis=1)[1]/Q) ** 2 - sigma[1] **2) ** .5 * np.sqrt(N/4),0

        a_1+= np.linalg.norm(a_1)  * np.random.randn(N)
        a_2+= np.linalg.norm(a_2)  * np.random.randn(N)

        X1 = np.reshape(np.unpackbits(Sbox(k^Text)^Mask),(2**N,Q,8))[:,:,8-N:8]

        stop = False
        compteur = 0
        while not stop:

            #Count the number of iterartions
            compteur += 1

            #The E Step
            Norm = (Y[0,:]-np.dot(X1,a_1)-b_1)**2+(Y[1,:]-np.dot(X2,a_2)-b_2)**2
            C = np.min(Norm,axis=0) #For numerical stabiblity of the exponnetial
            beta = np.exp(C-Norm)
            S  = np.sum(beta,axis=0) #Normalisation coefficient
            beta = beta / S #Normalised p.m.f

            #The M Step
            na1,b_1 = minimize(beta,X1,Y[0,:],Mask,Q,N)
            na2,b_2 = minimize(beta,X2,Y[1,:],Mask,Q,N)

            #Have a and b changed up to a certain tolerance level ?
            diff = np.linalg.norm(na1-a_1)**2  + np.linalg.norm(na2-a_2)**2
            stop = (diff < tolerance) or (compteur > 25)

            a_1,a_2 = na1,na2

        LL[k] = np.sum(np.log(S)) - np.sum(C) #Store the log-likelihood of the key

    k_guess = np.argsort(-LL)
    #print("EM-LIN",LL[k_guess[0]])
    return k_guess

"""
This implements the U-EM-HYB algorithm described in the article.
 ====> hamming model for the masks
 ====> linera model for the sbox output

Input:
    T         : the plaintext used for the considered Traces
    Y         : the two shares of the leakage
    sigma     : the noise level
    tolerance : a threshold to decide when to stopp the while loop
    N         : number of bits targeted

Output:
    k_guess   : the key hypothesis ranked by deacreasing goodness of fit with the model
"""
def EM_Hybride(T,Y,sigma=np.reshape(np.array([1,1]),(2,1)),N=8,tolerance=10**-8):

    Q = len(T)

    # Normalize and center the traces
    Y/=(2 ** .5 * sigma)
    y_bar = np.reshape(np.mean(Y,axis=1),(2,1))
    Y-=y_bar

    #To store the log-likelihood of each key hypothesis
    LL = np.zeros(2**N)

    Text,Mask = np.uint8(np.meshgrid(T, M))

    X2 = np.reshape(np.unpackbits(Mask),(2**N,Q,8))[:,:,8-N:8]

    sigma_y = np.std(Y,axis=1)

    for k in range(2**N):

        #a_1,b_1 = 1,0
        #a_2,b_2 = np.ones(N),0
        a_1,b_1 = np.abs( sigma_y[0] ** 2 - sigma[0] **2) ** .5 * np.sqrt(4/N),0
        a_2,b_2 = np.ones(N) * np.abs( sigma_y[0] ** 2 - sigma[0] **2) ** .5 * np.sqrt(4/N),0

        X1 = np.char.count(binary_repr_vec(Mask), '1').reshape((len(Mask),1))
        #X1 = np.reshape(np.unpackbits(Sbox(k^Text)^Mask),(2**N,Q,8))[:,:,8-N:8]

        stop = False
        while not stop:

            #The E Step
            Norm = (Y[0,:]-a_1 * X1-b_1)**2+(Y[1,:]-np.dot(X2,a_2)-b_2)**2
            C = np.min(Norm,axis=0) #For numerical stabiblity of the exponnetial
            beta = np.exp(C-Norm)
            S  = np.sum(beta,axis=0) #Normalisation coefficient
            beta = beta / S #Normalised p.m.f

            #The M Step
            na1,b_1 = minimize_u(beta,X1,Y[0,:],Mask,Q,N)
            na2,b_2 = minimize(beta,X2,Y[1,:],Mask,Q,N)

            #Have a and b changed up to a certain tolerance level ?
            diff = np.linalg.norm(na1-a_1)**2  + np.linalg.norm(na2-a_2)**2
            stop = (diff < tolerance)

            a_1,a_2 = na1,na2

        #Store the log-likelihood of the key
        LL[k] = np.sum(np.log(S)) - np.sum(C)


    k_guess = np.argsort(-LL)

    return k_guess

"""
This implements the  template attck with linear leakage model.

Input:
    T         : the plaintext used for the considered Traces
    Y         : the two shares of the leakage
    sigma     : the noise level
    tolerance : a threshold to decide when to stopp the while loop
    N         : number of bits targeted
    a_1,...   : parameters of the template to use

Output:
    k_guess   : the key hypothesis ranked by deacreasing goodness of fit with the model
"""
def template2(T,Y,sigma=np.reshape(np.array([1,1]),(2,1)),a_1=np.ones(4),b_1=0,a_2=np.ones(4),b_2=0,N=4):

    a1=a_1/(2 ** .5 * sigma[0])
    a2=a_2/(2 ** .5 * sigma[1])
    b1=b_1/(2 ** .5 * sigma[0])
    b2=b_2/(2 ** .5 * sigma[1])

    Q = len(T)
    Y/=(2 ** .5 * sigma)

    y_bar = np.mean(Y,axis=1) #Precompute the average of the traces
    LL = np.zeros(2**N) #To store the log-likelihood of each key hypothesis

    Text,Mask = np.uint8(np.meshgrid(T, M))

    X2 = np.reshape(np.unpackbits(Mask),(2**N,Q,8))[:,:,8-N:8]

    for k in range(2**N):

        X1 = np.reshape(np.unpackbits(Sbox(k^Text)^Mask),(2**N,Q,8))[:,:,8-N:8]
        Norm = (Y[0,:]-np.dot(X1,a1)-b1)**2+(Y[1,:]-np.dot(X2,a2)-b2)**2
        C = np.min(Norm,axis=0) #For numerical stabiblity of the exponnetial
        beta = np.exp(C-Norm)
        S  = np.sum(beta,axis=0) #Normalisation coefficient

        LL[k] = np.sum(np.log(S)) - np.sum(C) #Store the log-likelihood of the key


    k_guess = np.argsort(-LL)
    return k_guess

"""
This implements the  template attack with Hamming model.

Input:
    T         : the plaintext used for the considered Traces
    Y         : the two shares of the leakage
    sigma     : the noise level
    tolerance : a threshold to decide when to stopp the while loop
    N         : number of bits targeted
    a_1,...   : parameters of the template to use

Output:
    k_guess   : the key hypothesis ranked by deacreasing goodness of fit with the model
"""
def template_ham(T,Y,sigma=np.reshape(np.array([1,1]),(2,1)),a_1=1,b_1=0,a_2=1,b_2=0,N=4):

    t1 = time()

    a1=a_1/(2 ** .5 * sigma[0])
    a2=a_2/(2 ** .5 * sigma[1])
    b1=b_1/(2 ** .5 * sigma[0])
    b2=b_2/(2 ** .5 * sigma[1])

    Q = len(T)

    Y/=(2 ** .5 * sigma)
    Y = np.reshape(Y,(2,1,Q))


    LL = np.zeros(2**N) #To store the log-likelihood of each key hypothesis

    Text,Mask = np.meshgrid(T, M, copy=False, sparse=True)

    X1 = np.reshape(np.char.count(binary_repr_vec(Mask), '1'),(len(Mask),1))

    for k in range(2**N):

        X0 = np.char.count(binary_repr_vec(Sbox(k^Text)^Mask), '1')

        Norm =(Y[0]-a1*X0-b1)**2 + (Y[1]-a2*X1-b2)**2

        C = np.min(Norm,axis=0) #For numerical stabiblity of the exponnetial
        beta = np.exp(C-Norm)
        S  = np.sum(beta,axis=0) #Normalisation coefficient
        beta/=S #Normalised p.m.f

        LL[k] = np.sum(np.log(S)) - np.sum(C) #Store the log-likelihood of the key

    k_guess = np.argsort(-LL)

    t2 = time()

    return k_guess


"""
Subfonction used in the maximization step of the U-EM with regression ridge.

Input:
    beta : posterior probabilities for each masks and keys
    X    : sensitive varioables of the model
    Y    : leakage
    Q    : number of traces
    Mask : masks for sensitive  variable
    N    : Number of bits targeted
    lamb : The regularization parameter.

Output;
    na,nb: parameters that maximizes the goodness of fit
"""
def minimize_ridge(beta,X,Y,Mask,Q,N,lamb =10**2):

    XX = np.sum(X,axis=2)
    x_tilde = np.sum(beta * XX, axis = 0).T #Should belong to R^Q
    x_bar = np.mean(x_tilde,axis = 0) #Should belong to R

    var_x = np.sum(beta * (XX-x_bar) ** 2)
    cov = np.sum(beta * (XX-x_bar) *Y)

    na = cov / var_x * np.ones(N)

    x_tilde = np.sum(np.transpose(X,axes=(2,0,1)) * beta, axis = 1).T #Should belong to R^(QxN)
    x_bar = np.mean(x_tilde,axis = 0) #Should belong to R^N

    X_cent = np.reshape(X - x_bar,(len(Mask),Q,N,1))

    AutoCorr = np.einsum('mq,mqtp->tp' , beta,np.einsum('ijkl,ijhl->ijkh', X_cent,X_cent)) + lamb * np.eye(N)
    X_cent = np.reshape(X_cent,(len(Mask),Q,N))

    RelCorr = np.einsum('qm,q->m' ,  x_tilde ,np.reshape(Y - np.dot(x_tilde-x_bar,na),(Q)))  #Since Y is centered no need to sub x_bar to x_tilde

    na += np.linalg.solve(AutoCorr,RelCorr)
    #na += np.linalg.lstsq(AutoCorr,RelCorr,rcond=None)[0]
    nb = - np.dot(x_bar,na)

    return na,nb

"""
This implements the U-EM-LIN algorithm described in the article with ridge regression.

Input:
    T         : the plaintext used for the considered Traces
    Y         : the two shares of the leakage
    sigma     : the noise level
    tolerance : a threshold to decide when to stopp the while loop
    N         : number of bits targeted

Output:
    k_guess   : the key hypothesis ranked by deacreasing goodness of fit with the model
"""
def EM_BIV_LIN_RIDGE(T,Y,N=4,tolerance=10**-8,sigma=np.reshape(np.array([1,1]),(2,1))):

        t1 = time()
        lamb = 1/(2 * sigma_a ** 2)
        Q = len(T)

        Y/=(2 ** .5 * sigma)
        y_bar = np.reshape(np.mean(Y,axis=1),(2,1)) #Precompute the average of the traces
        Y-=y_bar
        sigma_y=np.std(Y,axis=1)

        LL = np.zeros(2**N) #To store the log-likelihood of each key hypothesis

        Text,Mask = np.uint8(np.meshgrid(T, M))

        X2 = np.reshape(np.unpackbits(Mask),(2**N,Q,8))[:,:,8-N:8]

        for k in range(2**N):

            a_1,b_1 = np.ones(N) * np.abs( sigma_y[0] ** 2 - sigma[0] **2) ** .5 * np.sqrt(4/N),0
            a_2,b_2 = np.ones(N) * np.abs( sigma_y[1] ** 2 - sigma[1] **2) ** .5 * np.sqrt(4/N),0

            X1 = np.reshape(np.unpackbits(Sbox(k^Text)^Mask),(2**N,Q,8))[:,:,8-N:8]

            stop = False
            while not stop:
                #The E Step
                Norm = (Y[0,:]-np.dot(X1,a_1)-b_1)**2+(Y[1,:]-np.dot(X2,a_2)-b_2)**2
                C = np.min(Norm,axis=0) #For numerical stabiblity of the exponnetial
                beta = np.exp(C-Norm)
                S  = np.sum(beta,axis=0) #Normalisation coefficient
                beta = beta / S #Normalised p.m.f

                #The M Step
                na1,b_1 = minimize_ridge(beta,X1,Y[0,:],Mask,Q,N,lamb=lamb)
                na2,b_2 = minimize_ridge(beta,X2,Y[1,:],Mask,Q,N,lamb=lamb)

                #Have a and b changed up to a certain tolerance level ?
                diff = np.linalg.norm(na1-a_1)**2  + np.linalg.norm(na2-a_2)**2
                stop = (diff < tolerance)

                a_1,a_2 = na1,na2
                #print(a_1)

            LL[k] = np.sum(np.log(S)) - np.sum(C) #Store the log-likelihood of the key

        k_guess = np.argsort(-LL)

        t2 = time()

        return k_guess

"""
Subfonction that instantiates the sensitive variable matrices.
"""
def x_synth_biv(k,Text,Mask,N=4):
    Q  = len(Text)
    k = np.uint8(k)
    l = np.reshape(np.unpackbits(Sbox(k^Text)^Mask),(Q,8))
    l1 = np.reshape(np.unpackbits(Mask),(Q,8))
    return l[:,8-N:8],l1[:,8-N:8]

"""
Procedure that computes the metrics (Succes Rate and Guessing Entropy) with numerical simulations.

Input:
    ListQ   : the list of number of traces to be tested.
    File    : prefix of the file name to be sed to store the results
    sigma   : noise level
    sigma_a : epistemic noise level
    NbIter  : number of repetitions used to estimate the metrics
"""
def metrics(ListQ=[10,20,30],File="test.txt",sigma=1,sigma_a=.4,NbIter=1000):

    N=4
    Sigma = np.reshape(np.array([sigma,sigma]),(2,1))

    convergence_EM_LIN,convergence_EM_HAM,convergence_ML_HAM,convergence_ML_LIN,convergence_2OCPA,convergence_ridge = False,False,False,False,False,True

    for Q in ListQ:

        SR_EM_LIN,SR_EM_HAM,SR_ML_LIN,SR_ML_HAM,SR_2OCPA,SR_EM_RIDGE = [],[],[],[],[],[]
        GE_EM_LIN,GE_EM_HAM,GE_ML_LIN,GE_ML_HAM,GE_2OCPA,GE_EM_RIDGE = [],[],[],[],[],[]
        t_EM_LIN,t_EM_HAM,t_ML_HAM,t_ML_LIN,t_2OCPA,t_EM_RIDGE = 0,0,0,0,0,0

        for i in range(NbIter):

            #The leakage Coefficient (we set b to 0)
            secret_a1 = np.ones(N) + sigma_a * np.random.randn(N)
            secret_a1 = np.sqrt(N) *  secret_a1 / norm(secret_a1)

            secret_a2 = np.ones(N) + sigma_a * np.random.randn(N)
            secret_a2 = np.sqrt(N) *  secret_a2 / norm(secret_a2)

            secret_k = np.random.randint(0,2**N) #The secret key
            T = np.random.randint(0,2**N,Q ,dtype=np.uint8) #The plaintext
            Mask = np.random.randint(0,2**N,Q ,dtype=np.uint8) #The Mask

            Noise1 = sigma * np.random.randn(Q) #The Noise
            Noise2 = sigma * np.random.randn(Q) #The Noise

            X2,X1 = x_synth_biv(secret_k,T,Mask,N=N)
            Y1 = np.dot(X1,secret_a1) + Noise1 #The synthetic leaks
            Y2 = np.dot(X2,secret_a2) + Noise2 #The synthetic leaks
            Y = np.array([Y1,Y2])
            
            """
            if not convergence_EM_HAM:
                YY = np.copy(Y)
                t1=time()
                k_g = EM_bivarie_2(T,YY,sigma=Sigma,N=4,tolerance=10**-8)
                t2=time()
                GE_EM_HAM.append(np.argmax(k_g == secret_k))
                SR_EM_HAM.append(k_g[0] == secret_k)
                t_EM_HAM += t2-t1

            if not convergence_EM_LIN:
                YY = np.copy(Y)
                t1=time()
                k_g = EM_BIV_LIN2(T,YY,sigma=Sigma,N=4,tolerance=10**-8)
                t2=time()
                GE_EM_LIN.append(np.argmax(k_g == secret_k))
                SR_EM_LIN.append(k_g[0] == secret_k)
                t_EM_LIN += t2-t1

            if not convergence_ML_HAM:
                YY = np.copy(Y)
                t1=time()
                k_g = template_ham(T,YY,sigma=Sigma,a_1=np.mean(secret_a1),b_1=0,a_2=np.mean(secret_a2),b_2=0,N=4)
                t2=time()
                GE_ML_HAM.append(np.argmax(k_g == secret_k))
                SR_ML_HAM.append(k_g[0] == secret_k)
                t_ML_HAM += t2-t1

            if not convergence_ML_LIN:
                YY = np.copy(Y)
                t1=time()
                k_g = template2(T,YY,sigma=Sigma,a_1=secret_a1,b_1=0,a_2=secret_a2,b_2=0,N=4)
                t2 = time()
                GE_ML_LIN.append(np.argmax(k_g == secret_k))
                SR_ML_LIN.append(k_g[0] == secret_k)
                t_ML_LIN += t2 - t1
            """

            if not convergence_2OCPA:
                YY = np.copy(Y)
                t1=time()
                k_g = HO_CPA(T,YY)
                t2 = time()
                GE_2OCPA.append(np.argmax(k_g == secret_k))
                SR_2OCPA.append(k_g[0] == secret_k)
                t_2OCPA += t2 - t1

            """
            if not convergence_ridge:
                YY = np.copy(Y)
                t1=time()
                k_g= EM_BIV_LIN_RIDGE(T,YY,N=4)
                t2 =time()
                GE_EM_RIDGE.append(np.argmax(k_g == secret_k))
                SR_EM_RIDGE.append(k_g[0] == secret_k)
                t_EM_RIDGE += t2 - t1
            """

        print("For Q = ",Q," we have :")
        
        """
        if not convergence_EM_HAM:
            print("SR_EM_HAM : ",np.mean(SR_EM_HAM)," GE_EM_HAM : ",np.mean(GE_EM_HAM)," t = ",t_EM_HAM/NbIter)
            convergence_EM_HAM = (np.mean(SR_EM_HAM) > 0.995)
            with open(File+"_EM_HAM", "a") as f:
                f.write(str(NbIter) + " " + str(Q) + " " +  str(sigma) + " " + str(t_EM_HAM/NbIter) + " " +str(np.mean(SR_EM_HAM)) + " " + str(np.std(SR_EM_HAM)) + " " + str(np.mean(GE_EM_HAM)) + " " + str(np.std(GE_EM_HAM)) + "\n")
            f.close()

        if not convergence_EM_LIN:
            print("SR_EM_LIN : ",np.mean(SR_EM_LIN)," GE_EM_LIN : ",np.mean(GE_EM_LIN)," t = ",t_EM_LIN/NbIter)
            convergence_EM_LIN = (np.mean(SR_EM_LIN) > 0.995)
            with open(File+"_EM_LIN", "a") as f:
                f.write(str(NbIter) + " " + str(Q) + " " +  str(sigma) + " " + str(t_EM_LIN/NbIter) + " " +  str(np.mean(SR_EM_LIN)) + " " + str(np.std(SR_EM_LIN)) + " " + str(np.mean(GE_EM_LIN)) + " " + str(np.std(GE_EM_LIN)) + "\n")
            f.close()

        if not convergence_ML_HAM:
            print("SR_ML_HAM : ",np.mean(SR_ML_HAM)," GE_ML_HAM : ",np.mean(GE_ML_HAM)," t = ",t_ML_HAM/NbIter)
            convergence_ML_HAM = (np.mean(SR_ML_HAM) > 0.995)
            with open(File+"_ML_HAM", "a") as f:
                f.write(str(NbIter) + " " + str(Q) + " " +  str(sigma) + " " + str(t_ML_HAM/NbIter) + " " +  str(np.mean(SR_ML_HAM)) + " " + str(np.std(SR_ML_HAM)) + " " + str(np.mean(GE_ML_HAM)) + " " + str(np.std(GE_ML_HAM)) + "\n")
            f.close()

        if not convergence_ML_LIN:
            print("SR_ML_LIN : ",np.mean(SR_ML_LIN)," GE_ML_LIN : ",np.mean(GE_ML_LIN)," t = ",t_ML_LIN/NbIter)
            convergence_ML_LIN = (np.mean(SR_ML_LIN) > 0.995)
            with open(File+"_ML_LIN", "a") as f:
                f.write(str(NbIter) + " " + str(Q) + " " +  str(sigma) + " " + str(t_ML_LIN/NbIter) + " " +  str(np.mean(SR_ML_LIN)) + " " + str(np.std(SR_ML_LIN)) + " " + str(np.mean(GE_ML_LIN)) + " " + str(np.std(GE_ML_LIN)) + "\n")
            f.close()
        """
        if not convergence_2OCPA:
            print("SR_2OCPA : ",np.mean(SR_2OCPA)," GE_2OCPA : ",np.mean(GE_2OCPA)," t = ",t_2OCPA/NbIter)
            convergence_2OCPA = (np.mean(SR_2OCPA) > 0.995)
            with open(File+"_2OCPA", "a") as f:
                f.write(str(NbIter) + " " + str(Q) + " " +  str(sigma) + " " + str(t_2OCPA/NbIter) + " " +  str(np.mean(SR_2OCPA)) + " " + str(np.std(SR_2OCPA)) + " " + str(np.mean(GE_2OCPA)) + " " + str(np.std(GE_2OCPA)) + "\n")
            f.close()
        """
        if not convergence_ridge:
            print("SR_ridge : ",np.mean(SR_EM_RIDGE)," GE_ridge : ",np.mean(GE_EM_RIDGE)," t = ",t_EM_RIDGE/NbIter)
            convergence_ridge = (np.mean(SR_EM_RIDGE) > 0.995)
            with open(File+"_ridge","a") as f:
                f.write(str(NbIter) + " " + str(Q) + " " + str(sigma) + " " + str(t_EM_RIDGE/NbIter) + " " + str(np.mean(SR_EM_RIDGE)) + " " + str(np.std(SR_EM_RIDGE)) + " " + str(np.mean(GE_EM_RIDGE)) + " " + str(np.std(GE_EM_RIDGE))+ "\n")
            f.close()
        """
# GARBAGE CODE
    """
    def y(k,t,m): #Leakage model
        return hw(binary_repr_vec(Sbox(k^t)^m),'1')+hw(binary_repr_vec(m),'1')

    def CPA(T,X,N=4,sigma=1):

        Q = len(T)
        Text,Mask = np.meshgrid(T, M, copy=False, sparse=True)

        Pearsons = np.zeros(2 ** N)

        mu_x = np.mean(X) #Precompute the mean

        for k in range(2**N):
            Y = np.mean((y(k,Text,Mask) - N) ** 2,axis=0)
            std_y = np.std(Y)

            if std_y == 0:
                return np.arange(16)

            mu_y = np.mean(Y)
            cov = np.mean((Y-mu_y) * (X-mu_x))
            Pearsons[k] = abs(cov) / std_y

        k_guess = np.argsort(-Pearsons)

        return k_guess



    We use the definition of:
    Taylor Expansion of ML Attacks for Masked and Shuffled Implementations

    def CenteredProduct(T,Y,N=4,sigma=.5):

        ybar = np.reshape(np.mean(Y,axis=1),(2,1))
        Yp = np.prod(Y - ybar,axis=0) #Belong to R^Q
        k_guess = CPA(T,Yp,N=N,sigma=2*sigma)
        return k_guess
    """

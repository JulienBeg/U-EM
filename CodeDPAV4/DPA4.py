import numpy as np
from numpy.linalg import norm

import scipy
from scipy.optimize import minimize
import scipy.io


import matplotlib.pyplot as plt
from matplotlib import cm


from time import time

import os

from sklearn.linear_model import LinearRegression
from scipy.stats import linregress


binary_repr_vec = np.vectorize(np.binary_repr)
hw = np.char.count

template_extraction = scipy.io.loadmat('template_extraction.mat')

a1_lin = template_extraction['a1_lin']
a2_lin = template_extraction['a2_lin']
b1_lin = template_extraction['b1_lin'][0]
b2_lin = template_extraction['b2_lin'][0]

a1_ham = template_extraction['a1_ham'][0]
a2_ham = template_extraction['a2_ham'][0]
b1_ham = template_extraction['b1_ham'][0]
b2_ham = template_extraction['b2_ham'][0]

K_s = template_extraction['K_s'][0]
sigma_reg = template_extraction['sigma_reg']
sigma_reg_a = template_extraction['sigma_reg_a']

def Sbox_(b):

    #S = np.array([2,12,4,1,7,10,11,6,8,5,3,15,13,0,14,9],dtype=np.uint8)
    # 1st line of 1st DES Sbox (can be critized as an NSA cipher, which non-formalized security proof)
    # https://en.wikipedia.org/wiki/S-box
    #S = np.array([0xC, 0x5, 0x6, 0xB, 0x9, 0x0, 0xA, 0xD, 0x3, 0xE, 0xF, 0x8, 0x4, 0x7, 0x1, 0x2],dtype=np.uint8) # PRESENT

    # SubBytes look-up-table

    S = np.array([
    0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5, 0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76,
    0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0, 0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0,
    0xb7, 0xfd, 0x93, 0x26, 0x36, 0x3f, 0xf7, 0xcc, 0x34, 0xa5, 0xe5, 0xf1, 0x71, 0xd8, 0x31, 0x15,
    0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a, 0x07, 0x12, 0x80, 0xe2, 0xeb, 0x27, 0xb2, 0x75,
    0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0, 0x52, 0x3b, 0xd6, 0xb3, 0x29, 0xe3, 0x2f, 0x84,
    0x53, 0xd1, 0x00, 0xed, 0x20, 0xfc, 0xb1, 0x5b, 0x6a, 0xcb, 0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf,
    0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85, 0x45, 0xf9, 0x02, 0x7f, 0x50, 0x3c, 0x9f, 0xa8,
    0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5, 0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2,
    0xcd, 0x0c, 0x13, 0xec, 0x5f, 0x97, 0x44, 0x17, 0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73,
    0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a, 0x90, 0x88, 0x46, 0xee, 0xb8, 0x14, 0xde, 0x5e, 0x0b, 0xdb,
    0xe0, 0x32, 0x3a, 0x0a, 0x49, 0x06, 0x24, 0x5c, 0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79,
    0xe7, 0xc8, 0x37, 0x6d, 0x8d, 0xd5, 0x4e, 0xa9, 0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08,
    0xba, 0x78, 0x25, 0x2e, 0x1c, 0xa6, 0xb4, 0xc6, 0xe8, 0xdd, 0x74, 0x1f, 0x4b, 0xbd, 0x8b, 0x8a,
    0x70, 0x3e, 0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e, 0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e,
    0xe1, 0xf8, 0x98, 0x11, 0x69, 0xd9, 0x8e, 0x94, 0x9b, 0x1e, 0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf,
    0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68, 0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb, 0x16 ],dtype=np.uint8)


    return S[b]

Sbox = np.vectorize(Sbox_)

mask = np.array([3,12,53,58,80,95,102,105,150,153,160,175,197,202,243,252],dtype= np.uint8)


def Mask_(b):
    return mask[b]

Mask = np.vectorize(Mask_)

def sr_ge(algo,Y,T,batch,k_s=K_s[0],file="test",folder=0):

    i = 0
    ge = 0
    sr = 0

    while (i+1) * batch <= len(T):

        Y_batch = np.copy(Y[:,i*batch:(i+1)*batch])
        T_batch = T[i*batch:(i+1)*batch]

        guess = algo(T_batch,Y_batch,N=8,folder=folder)

        print("En position ",np.argmax(guess == k_s))

        ge = ge + (np.argmax(guess == k_s))
        sr = sr + (guess[0] == k_s)
        i = i+1

    sr =sr/i
    ge = ge/i

    if file != "test":
        with open(file,'a') as f:
            f.write(str(batch) + " " + str(ge) + " " + str(sr) + "\n")
        f.close()

    print("Pour Q = " + str(batch) + " on a GE = " + str(ge) + " et SR = " + str(sr))

    return sr,ge


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

def minimize_u(beta,X,Y):

    x_tilde = np.sum(beta * X, axis = 0).T #Should belong to R^Q
    x_bar = np.mean(x_tilde,axis = 0) #Should belong to R

    var_x = np.sum(beta * (X-x_bar) ** 2)
    cov = np.sum(beta * (X-x_bar) *Y)

    na = cov / var_x
    b = - na * x_bar

    return na,b

def EM_bivarie_2(T,Y,N=8,tolerance=10**-8,folder=0):

    t1 = time()

    sigma=np.copy(sigma_reg[folder])
    Q = len(T)

    sigma = np.reshape(np.array(sigma),(2,1))
    #Normalisation and centering
    Y/=(2 ** .5 * sigma)
    y_bar = np.reshape(np.mean(Y,axis=1),(2,1)) #Precompute the average of the traces
    Y-=y_bar
    sigma_y = np.std(Y,axis=1)/Q

    LL = np.zeros(2**N) #To store the log-likelihood of each key hypothesis

    M = np.array([3,12,53,58,80,95,102,105,150,153,160,175,197,202,243,252],dtype = np.uint8)
    Text,Mask = np.meshgrid(T, M, copy=False, sparse=True)

    X1 = np.reshape(np.char.count(binary_repr_vec(Mask), '1'),(len(Mask),1))

    for k in range(2**N):

        a1,b1 = - np.abs( sigma_y[0] ** 2 - sigma[0] **2) ** .5 * np.sqrt(4/N),0
        a2,b2 = - np.abs( sigma_y[1] ** 2 - sigma[1] **2) ** .5 * np.sqrt(4/N),0

        X0 = np.char.count(binary_repr_vec(Sbox(k^Text)^Mask), '1')

        stop = False
        while not stop:

            #The E Step
            Norm =(Y[0]-a1*X0-b1)**2 + (Y[1]-a2*X1-b2)**2

            C = np.min(Norm,axis=0) #For numerical stabiblity of the exponnetial
            beta = np.exp(C-Norm)
            S  = np.sum(beta,axis=0) #Normalisation coefficient
            beta = beta / S #Normalised p.m.f

            #The M Step
            na1,b1 = minimize_u(beta,X0,Y[0])
            na2,b2 = minimize_u(beta,X1,Y[1])

            #Does the new values of a and b changed up to a certain tolerance level ?
            stop = abs(na1-a1) + abs(na2-a2) < tolerance

            a1,a2 = na1,na2

        LL[k] = np.sum(np.log(S)) - np.sum(C) #Store the log-likelihood of the key

    k_guess = np.argsort(-LL)

    t2 = time()
    print("Execution de EM bivarié en " + str(t2-t1))

    return k_guess

def minimize(beta,X,Y,Mask,Q,N):

    x_tilde = np.sum(np.transpose(X,axes=(2,0,1)) * beta, axis = 1).T #Should belong to R^(QxN)
    x_bar = np.mean(x_tilde,axis = 0) #Should belong to R^N

    X_cent = np.reshape(X - x_bar,(len(Mask),Q,N,1))

    AutoCorr = np.einsum('mq,mqtp->tp' , beta,np.einsum('ijkl,ijhl->ijkh', X_cent,X_cent))
    RelCorr = np.einsum('qm,q->m' ,  x_tilde ,Y) #Since Y is centered no need to sub x_bar to x_tilde

    na = np.linalg.solve(AutoCorr,RelCorr)
    #na = np.linalg.lstsq(AutoCorr,RelCorr,rcond=-1)[0]
    nb = - np.dot(x_bar,na)

    return na,nb

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

    AutoCorr = np.einsum('mq,mqtp->tp' , beta,np.einsum('ijkl,ijhl->ijkh', X_cent,X_cent)) + Q * lamb * np.eye(N)
    X_cent = np.reshape(X_cent,(len(Mask),Q,N))

    RelCorr = np.einsum('qm,q->m' ,  x_tilde ,np.reshape(Y - np.dot(x_tilde-x_bar,na),(Q)))  #Since Y is centered no need to sub x_bar to x_tilde

    na += np.linalg.solve(AutoCorr,RelCorr)
    #na += np.linalg.lstsq(AutoCorr,RelCorr,rcond=None)[0]
    nb = - np.dot(x_bar,na)

    return na,nb


def EM_BIV_LIN_RIDGE(T,Y,N=8,tolerance=10**-8,folder=0):
    
    t1 = time()
    lamb = 0.1
    Q = len(T)

    sigma=np.copy(sigma_reg[folder])
    sigma = np.reshape(np.array(sigma),(2,1))
    Y/=(2 ** .5 * sigma)
    y_bar = np.reshape(np.mean(Y,axis=1),(2,1)) #Precompute the average of the traces
    Y-=y_bar
    sigma_y=np.std(Y,axis=1)

    LL = np.zeros(2**N) #To store the log-likelihood of each key hypothesis

    M = np.array([3,12,53,58,80,95,102,105,150,153,160,175,197,202,243,252],dtype = np.uint8)
    Text,Mask = np.uint8(np.meshgrid(T, M))

    X2 = np.reshape(np.unpackbits(Mask),(len(Mask),Q,N))

    for k in range(2**N):

        a_1,b_1 = -np.ones(N) * np.abs( sigma_y[0] ** 2 - sigma[0] **2) ** .5 * np.sqrt(4/N),0
        a_2,b_2 = -np.ones(N) * np.abs( sigma_y[1] ** 2 - sigma[1] **2) ** .5 * np.sqrt(4/N),0

        X1 = np.reshape(np.unpackbits(Sbox(k^Text)^Mask),(len(Mask),Q,N))

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
    print("Execution de EM_BIV_LIN en " + str(t2-t1))
    
    return k_guess

def EM_Hybride_ridge(T,Y,N=8,tolerance=10**-8,folder=0):
    
    lamb = 5
    t1 = time()
    Q = len(T)

    sigma = sigma_reg[folder]
    sigma = np.reshape(np.array(sigma),(2,1))

    Y/=(2 ** .5)
    y_bar = np.reshape(np.mean(Y,axis=1),(2,1)) #Precompute the average of the traces
    Y-=y_bar
    sigma_y=np.std(Y,axis=1)/Q

    LL = np.zeros(2**N) #To store the log-likelihood of each key hypothesis
    M = np.array([3,12,53,58,80,95,102,105,150,153,160,175,197,202,243,252],dtype = np.uint8)
    Text,Mask = np.uint8(np.meshgrid(T, M))

    X2 = np.reshape(np.unpackbits(Mask),(len(Mask),Q,N))

    for k in range(2**N):
        
        a_1,b_1 = - np.abs( sigma_y[0] ** 2 - sigma[0] **2) ** .5 * np.sqrt(4/N),0
        a_2,b_2 = - np.ones(N) * np.abs( sigma_y[1] ** 2 - sigma[1] **2) ** .5 * np.sqrt(4/N),0

        X1 = np.char.count(binary_repr_vec(Sbox(k^Text)^Mask), '1')

        stop = False
        while not stop:
            
            #The E Step
            Norm = (Y[0,:]-a_1 * X1-b_1)**2/(sigma[0]**2)+(Y[1,:]-np.dot(X2,a_2)-b_2)**2/(sigma[1] ** 2) 
            C = np.min(Norm,axis=0) #For numerical stabiblity of the exponnetial
            beta = np.exp(C-Norm)
            S  = np.sum(beta,axis=0) #Normalisation coefficient
            beta = beta / S #Normalised p.m.f
            
            #The M Step
            na1,b_1 = minimize_u(beta,X1,Y[0,:])
            na2,b_2 = minimize_ridge(beta,X2,Y[1,:],Mask,Q,N,lamb=lamb)
            
            #print(sigma)
            sigma = np.array([np.sum(beta*(Y[0,:]-a_1*X1-b_1)**2),np.sum(beta*(Y[1,:]-np.dot(X2,a_2)-b_2)**2)]) 
            sigma/=Q
            sigma = sigma ** .5 

            #Have a and b changed up to a certain tolerance level ?
            diff = np.linalg.norm(na1-a_1)**2  + np.linalg.norm(na2-a_2)**2
            stop = (diff < tolerance)
            
            a_1,a_2 = na1,na2
            
        LL[k] = np.sum(np.log(S)) - np.sum(C) #Store the log-likelihood of the key

    k_guess = np.argsort(-LL)
    
    t2 =time()
    print("Execution en " + str(t2-t1))
    
    return k_guess



def EM_BIV_LIN2(T,Y,N=8,tolerance=10**-8,folder=0):
    t1 = time()

    Q = len(T)

    sigma=np.copy(sigma_reg[folder])

    sigma = np.reshape(np.array(sigma),(2,1))
    Y/=(2 ** .5 * sigma)
    y_bar = np.reshape(np.mean(Y,axis=1),(2,1)) #Precompute the average of the traces
    Y-=y_bar
    sigma_y=np.std(Y,axis=1)

    LL = np.zeros(2**N) #To store the log-likelihood of each key hypothesis

    M = np.array([3,12,53,58,80,95,102,105,150,153,160,175,197,202,243,252],dtype = np.uint8)
    Text,Mask = np.uint8(np.meshgrid(T, M))

    X2 = np.reshape(np.unpackbits(Mask),(len(Mask),Q,N))

    for k in range(2**N):

        a_1,b_1 = -np.ones(N) * np.abs( sigma_y[0] ** 2 - sigma[0] **2) ** .5 * np.sqrt(4/N),0
        a_2,b_2 = -np.ones(N) * np.abs( sigma_y[1] ** 2 - sigma[1] **2) ** .5 * np.sqrt(4/N),0

        X1 = np.reshape(np.unpackbits(Sbox(k^Text)^Mask),(len(Mask),Q,N))

        stop = False
        while not stop:

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
            stop = (diff < tolerance)

            a_1,a_2 = na1,na2

        LL[k] = np.sum(np.log(S)) - np.sum(C) #Store the log-likelihood of the key


    k_guess = np.argsort(-LL)

    t2 = time()
    print("Execution de EM_BIV_LIN en " + str(t2-t1))

    return k_guess

def EM_Hybride(T,Y,N=8,tolerance=10**-8,folder=0):

    t1 = time()
    Q = len(T)

    sigma = sigma_reg[folder]
    sigma = np.reshape(np.array(sigma),(2,1))

    Y/=(2 ** .5 * sigma)
    y_bar = np.reshape(np.mean(Y,axis=1),(2,1)) #Precompute the average of the traces
    Y-=y_bar
    sigma_y=np.std(Y,axis=1)/Q

    LL = np.zeros(2**N) #To store the log-likelihood of each key hypothesis

    M = np.array([3,12,53,58,80,95,102,105,150,153,160,175,197,202,243,252],dtype = np.uint8)
    Text,Mask = np.uint8(np.meshgrid(T, M))

    X2 = np.reshape(np.unpackbits(Mask),(len(Mask),Q,N))

    for k in range(2**N):

        a_1,b_1 = - np.abs( sigma_y[0] ** 2 - sigma[0] **2) ** .5 * np.sqrt(4/N),0
        a_2,b_2 = - np.ones(N) * np.abs( sigma_y[1] ** 2 - sigma[1] **2) ** .5 * np.sqrt(4/N),0

        X1 = np.char.count(binary_repr_vec(Sbox(k^Text)^Mask), '1')

        stop = False
        while not stop:

            #The E Step
            Norm = (Y[0,:]-a_1 * X1-b_1)**2+(Y[1,:]-np.dot(X2,a_2)-b_2)**2
            C = np.min(Norm,axis=0) #For numerical stabiblity of the exponnetial
            beta = np.exp(C-Norm)
            S  = np.sum(beta,axis=0) #Normalisation coefficient
            beta = beta / S #Normalised p.m.f

            #The M Step
            na1,b_1 = minimize_u(beta,X1,Y[0,:])
            na2,b_2 = minimize(beta,X2,Y[1,:],Mask,Q,N)

            #Have a and b changed up to a certain tolerance level ?

            diff = np.linalg.norm(na1-a_1)**2  + np.linalg.norm(na2-a_2)**2
            stop = (diff < tolerance)

            a_1,a_2 = na1,na2

        LL[k] = np.sum(np.log(S)) - np.sum(C) #Store the log-likelihood of the key

    k_guess = np.argsort(-LL)
    t2 =time()

    print("Execution en " + str(t2-t1))
    return k_guess

def template2(T,Y,N=8,folder=0):

    t1 = time()

    a1 = np.copy(a1_lin[folder])
    a2 = np.copy(a2_lin[folder])
    b1 = np.copy(b1_lin[folder])
    b2 = np.copy(b2_lin[folder])
    sigma = np.copy(sigma_reg[folder])

    a1/=(2 ** .5 * sigma[0])
    a2/=(2 ** .5 * sigma[1])
    b1/=(2 ** .5 * sigma[0])
    b2/=(2 ** .5 * sigma[1])
    Q = len(T)

    sigma = np.reshape(np.array(sigma),(2,1))
    Y/=(2 ** .5 * sigma)

    y_bar = np.mean(Y,axis=1) #Precompute the average of the traces
    LL = np.zeros(2**N) #To store the log-likelihood of each key hypothesis

    Y = np.reshape(Y,(2,1,Q))

    M = np.array([3,12,53,58,80,95,102,105,150,153,160,175,197,202,243,252],dtype = np.uint8)

    Text,Mask = np.uint8(np.meshgrid(T, M))

    X2 = np.reshape(np.unpackbits(Mask),(len(Mask),Q,N))

    for k in range(2**N):

        X1 = np.reshape(np.unpackbits(Sbox(k^Text)^Mask),(len(Mask),Q,N))

        Norm = (Y[0,:]-np.dot(X1,a1)-b1)**2+(Y[1,:]-np.dot(X2,a2)-b2)**2
        C = np.min(Norm,axis=0) #For numerical stabiblity of the exponnetial
        beta = np.exp(C-Norm)
        S  = np.sum(beta,axis=0) #Normalisation coefficient

        LL[k] = np.sum(np.log(S)) - np.sum(C) #Store the log-likelihood of the key


    k_guess = np.argsort(-LL)

    t2 = time()
    #print("Execution de template en " + str(t2-t1))

    return k_guess

def template_ham(T,Y,N=8,folder=0):

    t1 = time()

    sigma=np.copy(sigma_reg[folder])
    a1=np.copy(a1_ham[folder])
    b1=np.copy(b1_ham[folder])
    a2=np.copy(a2_ham[folder])
    b2=np.copy(b2_ham[folder])

    a1/=(2 ** .5 * sigma[0])
    a2/=(2 ** .5 * sigma[1])
    b1/=(2 ** .5 * sigma[0])
    b2/=(2 ** .5 * sigma[1])

    Q = len(T)

    sigma = np.reshape(np.array(sigma),(2,1))
    Y/=(2 ** .5 * sigma)

    LL = np.zeros(2**N) #To store the log-likelihood of each key hypothesis

    M = np.array([3,12,53,58,80,95,102,105,150,153,160,175,197,202,243,252],dtype = np.uint8)
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
    #print("Execution de template en " + str(t2-t1))

    return k_guess

def read_trace(trace,Q=5000):

    T = np.zeros(Q,dtype=np.uint8)
    Y = np.zeros((Q,2),dtype=np.float32)

    with open(trace,'r') as f:
        ligne = f.readline()

        for i in range(Q):
            ligne = f.readline()
            [Plain,Y1,Y2] = ligne.split(" ")
            T[i] = np.uint8(Plain)
            Y[i] = np.array([np.float(Y1),np.float32(Y2)])
    return Y.T,T

"""
sigma_reg = [
    [0.002146930604462152,  0.0038926152600440417],
    [0.002146930604462152,  0.0038926152600440417],
    [0.0022124021189260166, 0.004488499352663603],
    [0.0021257884864935954, 0.0045181655310092645],
    [0.002091923976776936, 0.004475695416467315],
    [0.0020812447768082587,0.004445649039268404],
    [0.0020437358553688383,0.004722585802476086],
    [0.0019648144704376425,0.004080696825757869],
    [0.0018518886749266511,0.004079834623682795],
    [0.002056695455600077, 0.0045988908408684005],
    [0.0019214036565788862, 0.004209190233502404],
    [0.0018837530318619353,0.004018815585011199],
    [0.0019452933808886202, 0.006625427655089334],
    [0.0021412520776138855,0.004997030655142686],
    [0.0021497190370067107,0.004617404945519296],
    [0.0021336660261363546,0.004587851296374271 ]
]

a1_ham = [
    -0.0018920969036581592,-0.0019026283681700432,-0.0018701289033154324,-0.0017979451116750707,
    -0.001775644107515437,-0.0017456772245749781,-0.0016774217576311053,-0.00165151873593629,
    -0.001629563315715378,-0.0017199455044341,-0.0017630546019408745,-0.0017017442531719084,
    -0.001719040505985539 ,-0.0016935940702841631,-0.0016237631209947408,-0.0015909232333311514
    ]

b1_ham = [
    0.03629889037897192,0.036353473542006944,0.03635669334665605,0.03709192717546261,
    0.03723580288968541,0.037190104767246004,0.03674636599607699, 0.03621521910154526,
    0.03506896636560759,0.036939252859045986, 0.036314813780034726, 0.03610716439149118,
    0.03627603823307338,0.03697299940614879, 0.03708373174759961, 0.03701967018182819
    ]

a2_ham = [
    -0.0031265738379141214,-0.0026283289021268915,-0.002744444693043594, -0.002748035918456314,
    -0.0027464795670550084,-0.002749555607608848,-0.0026598979810894792,-0.0026514491008283315,
    -0.0025088994314698236,-0.002612320829732172,-0.002619764580003875,-0.002582703232662246 ,
    -0.0027817162249929206,-0.0028034416453664936,-0.0028902208449608437,-0.0028500913380074214
    ]


b2_ham = [
    0.018416932679172965,-0.001884512591843487,0.0005090333913082539,0.007888908551949602,
    0.009441089316745309,0.010590612918499276,0.01288180668525271,0.01516767960394155,
    0.013981531177701307,0.013428011795908329,0.013068855932994191,0.013322062125691961,
    0.0124194395578714966, 0.006783479546658998,0.009829249632689238,0.009893467762324271
    ]

a1_lin = [
    np.array([-0.00190943, -0.00191019, -0.00210784, -0.00204082, -0.00172717, -0.00194921,-0.00190403, -0.00157196]),
    np.array([-0.00203394, -0.00198705, -0.00219777, -0.00200749, -0.00170913, -0.00186192,-0.00189162, -0.00153149]),
    np.array([-0.00193277, -0.00197393, -0.00216179, -0.00184839, -0.0017035,-0.00198147,-0.00189572, -0.00146338]),
    np.array([-0.00182131,-0.00190581,-0.00206084,-0.00194002,-0.00166516,-0.00190154,-0.00164367, -0.00144123]),
    np.array([-0.00185676, -0.00187299, -0.00193095, -0.00186147, -0.00166437, -0.00191625,-0.00164492, -0.00144408]),
    np.array([-0.00181135, -0.00181332, -0.00185848, -0.00190787, -0.0016072,  -0.00194043,-0.00166981, -0.00135567]),
    np.array([-0.00178455, -0.00180862, -0.00191624, -0.001737, -0.00144567, -0.00193845,-0.00154296, -0.00123622]),
    np.array([-0.00166478, -0.00188072, -0.00180414, -0.00170803, -0.0014581,  -0.00174861, -0.00157071, -0.00136703]),
    np.array([-0.00178847, -0.00169761, -0.00181732, -0.0018189,  -0.00141778, -0.00168175,-0.00154224, -0.00129138]),
    np.array([-0.00177393, -0.00174763, -0.00188144, -0.00177275, -0.00161968, -0.00189941,-0.00169348, -0.00137046]),
    np.array([-0.00176458, -0.00181597, -0.00204029, -0.00183127, -0.00164496, -0.00181658,-0.0017516, -0.00143084]),
    np.array([-0.00172238, -0.00166345, -0.00192328, -0.0018268,  -0.00153997, -0.0018388, -0.00166685, -0.00142171]),
    np.array([-0.00172833, -0.00172255, -0.00184583, -0.00191832, -0.00168872, -0.00191433,-0.00159469, -0.00133231]),
    np.array([-0.00169825, -0.00197432, -0.00184983, -0.00176745, -0.00156619, -0.00179908,-0.00157991, -0.00134159]),
    np.array([-0.00168494, -0.00188059, -0.00163265, -0.00168064, -0.00152291, -0.00180447,-0.00162485, -0.001146 ]),
    np.array([-0.00152326, -0.00179007, -0.00173819, -0.0016329,  -0.00150584, -0.00168436,-0.00149187, -0.00136619])
]

a2_lin = [
    np.array([-0.00371293, -0.0022243,  -0.00635163, -0.0016884,  -0.00685319, -0.00152264,-0.00150035, -0.00129781]),
    np.array([-0.00504764, -0.00316664, -0.00422107, -0.00129882, -0.0033019,  -0.00141764, -0.00115853, -0.00156999]),
    np.array([-0.00490919,-0.00309786,-0.00484119,-0.001326,-0.00366893,-0.00153317,-0.00137669,-0.00138296]),
    np.array([-0.00436183,-0.00266647,-0.00615409,-0.0013147,-0.00382606,-0.00110381,-0.0010939 ,-0.00120248]),
    np.array([-0.00407916, -0.00273035, -0.00624452, -0.00125753, -0.00403567, -0.00114535,-0.00099752, -0.001356]),
    np.array([-0.00427868, -0.00286277, -0.00617421, -0.0012152, -0.00399345, -0.00119463, -0.00111185, -0.00128084]),
    np.array([-0.00399502, -0.00264796, -0.00616204, -0.00119096, -0.00425068, -0.00131162,-0.00094715, -0.0011259 ]),
    np.array([-0.00406801, -0.00240589, -0.00601464, -0.00128793, -0.00447499, -0.00114916, -0.00078053, -0.00100684]),
    np.array([-0.00378824, -0.00251766, -0.00565825, -0.00115141, -0.00384835, -0.00127433,-0.00101448, -0.00106989]),
    np.array([-0.00412824, -0.00258914, -0.00582852, -0.00120306, -0.00378025, -0.00113442,-0.00121947, -0.00100003]),
    np.array([-0.00401611, -0.00237782, -0.0059097,  -0.00107482, -0.00404167, -0.00121923,-0.00104262, -0.00117328]),
    np.array([-0.00397569, -0.00253154, -0.00590249, -0.00108463, -0.00378773, -0.00118877, -0.00102284, -0.0011278 ]),
    np.array([-0.00414406, -0.00262449, -0.00644423, -0.00120103, -0.00383996, -0.00114248,-0.00108304, -0.00138397]),
    np.array([-0.00463403, -0.0029234,  -0.00664346, -0.001229,   -0.00430671, -0.00120822,-0.00099118, -0.00103476]),
    np.array([-0.00417286, -0.00264312, -0.00657102, -0.00127029, -0.00468196, -0.00119667,-0.00114871, -0.00099807]),
    np.array([-0.00419463, -0.00264668, -0.00662119, -0.00110131, -0.00466834, -0.00106452,-0.00116147, -0.00125887])
]

b1_lin = [
    0.03629269585401671,0.036351682475452,0.036362589886256805,0.03708753768722482,
    0.03723174746138423,0.03718951870740546,0.036738576200648194,0.03621164955897839,
    0.035084836743360596,0.03693877337891938,0.03630762673735857,0.03610771552529783,
    0.03627004188647614,0.036990582212581585,0.03708448358580673,0.03702179670369548
]

b2_lin = [
    0.018486490797636673,-0.001804398276041088,0.0006014501846784218,0.007741467638722593,
    0.009380633872389289, 0.010608230350388208, 0.013032821719323563,0.015143381675450977,
    0.014097354103505589,0.013417472203888354,0.012992372968844307,0.01336821855114482,
    0.012262444355906833,0.007028554466824926,0.009562519095466933,0.009862261040301194
]

K_s = [130,239,121,239,106,192,56,47,243,212,195,128,5,81,83,179]
"""

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

def Sbox_(b):

    # PRESENT SBox
    S = np.array([0xC, 0x5, 0x6, 0xB, 0x9, 0x0, 0xA, 0xD, 0x3, 0xE, 0xF, 0x8, 0x4, 0x7, 0x1, 0x2],dtype=np.uint8) # PRESENT
    return S[b]

Sbox = np.vectorize(Sbox_)

M = np.arange(2**4,dtype = np.uint8)
N = 4
Nparams = int(N*(N+1)/2) + 1

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
def EM_bivarie_2(T,Y,sigma=np.reshape(np.array([1,1]),(2,1)),tolerance=10**-10):

    Q = len(T)

    #Normalisation and centering
    Y/=(2 ** .5 * sigma)
    y_bar = np.reshape(np.mean(Y,axis=1),(2,1)) #Precompute the average of the traces
    Y-=y_bar
    Y = np.reshape(Y,(2,1,Q))

    LL = np.zeros(2**N) #To store the log-likelihood of each key hypothesis

    Text,Mask = np.meshgrid(T, M, copy=False, sparse=True)

    X0 = np.reshape(np.char.count(binary_repr_vec(Mask), '1'),(len(Mask),1))

    for k in range(2**N):
        a1,a2 = 1,1
        b1,b2 = 0,0

        X1 = np.char.count(binary_repr_vec(Sbox(k^Text)^Mask), '1')

        stop = False
        while not stop:

            #The E Step
            Norm =(Y[0]-a1*X0-b1)**2 + (Y[1]-a2*X1-b2)**2
            C = np.min(Norm,axis=0) #For numerical stabiblity of the exponnetia
            beta = np.exp(C-Norm)
            S  = np.sum(beta,axis=0) #Normalisation coefficient
            beta = beta / S #Normalised p.m.f

            #The M Step
            na1,b1 = minimize_u(beta,X0,Y[0],Q)
            na2,b2 = minimize_u(beta,X1,Y[1],Q)

            #Does the new values of a and b changed up to a certain tolerance level ?
            stop = abs(na1-a1) + abs(na2-a2) < tolerance

            a1,a2 = na1,na2

        LL[k] = np.sum(np.log(S)) - np.sum(C) #Store the log-likelihood of the key
    k_guess = np.argsort(-LL)
    #print("EM-HAM",LL[k_guess[0]])
    return k_guess

"""
Subfonction used for the U-EM-QUAD
Input:
    Cvec : a list of Nparams - 1 parameters
Output:
    sym  : an upper trangular matrix where the parameters are layed out
"""
def vec_to_sym(Cvec):
    pos = 0
    sym = np.zeros((N,N))
    for i in range(N):
        sym[i,i:] =  Cvec[pos:pos+N-i]
        pos+=N-i
    return sym

"""
Another subfonction that does the inverse of the previous one.
(with some extra dimension at the begining)
"""
def matrix_to_vec(to_be_vec,m,q):
    pos = 0
    vec = np.zeros((m,q,Nparams-1))
    for i in range(N):
        vec[:,:,pos:pos+N-i] = to_be_vec[:,:,i,i:N]
        pos+=N-i
    return vec

"""
Subfonction that evaluates a quadratic form with design matrix C at the vector X.
"""
def evaluate(C,X):
    return np.einsum("mqj,ji,mqi->qm",X,vec_to_sym(C[0:Nparams-1]),X) + C[Nparams-1]

"""
Subfonction that compute the euclidean distance between a a model and some leakages.
"""
def objective(Cvec,bias,X,Y):
    return norm(Y-evaluate(vec_to_sym(Cvec),X)-bias)

"""
This implements the U-EM-QUAD as described in the article.

Input :
    T          :  the plaintext used with the considered traces
    Y          :  the leakages
    sigma      :  the noise level
    regu       :  the constant used for regression ridge
    tolerance  :  threshold to decide when to stop the while loop

Output:
    k_guess    : the key hypothesis ranked by deacreasing goodness of fit with the model.
"""
def EM_quadratic(T,Y,sigma=np.reshape(np.array([1,1]),(2,1)),regu = 1,tolerance=5*10**-3):

    Q = len(T)

    #Normalisation and centering
    Y/=(2 ** .5 * sigma)

    ones = np.ones(Nparams) / (2 ** .5 * sigma[0])

    sigma_y = np.std(Y,axis=1)
    y_bar = np.reshape(np.mean(Y,axis=1),(2,1)) #Precompute the average of the traces
    Y-=y_bar
    Y = Y.reshape((2,Q))

    LL = np.zeros(2**N) #To store the log-likelihood of each key hypothesis

    #Text,Mask = np.meshgrid(T, M, copy=False, sparse=True)
    Text,Mask = np.meshgrid(T,M, copy = False)

    X0 = np.unpackbits(Mask).reshape((2**N,Q,8))[:,:,8-N:8]
    X0vec = np.ones((2**N,Q,Nparams))
    X0vec[:,:,0:Nparams-1] = matrix_to_vec(np.einsum("abn,abm->abnm",X0,X0),2**N,Q)
    X0vec = X0vec.transpose((1,0,2))

    for k in range(2**N):

        Cvec0 = np.copy(ones)
        Cvec1 = np.copy(ones)

        X1 = np.unpackbits(Sbox(k^Text)^Mask).reshape((2**N,Q,8))[:,:,8-N:8]
        X1vec = np.ones((2**N,Q,Nparams))
        X1vec[:,:,0:Nparams-1] = matrix_to_vec(np.einsum("abn,abm->abnm",X1,X1),2**N,Q)
        X1vec = X1vec.transpose((1,0,2))

        last_ll = 10**4 #dummy init

        stop = False
        maxiter = 15
        niter = 0
        while (not stop) and (niter < maxiter):

            niter += 1

            # Precompute the Quadratic Form
            f0 = evaluate(Cvec0,X0).reshape((Q,2**N))
            f1 = evaluate(Cvec1,X1).reshape((Q,2**N))

            #The E Step
            Norm =  (Y[0].reshape((Q,1)) - f0)**2
            Norm += (Y[1].reshape((Q,1)) - f1)**2 #Shape (Q,2**N)

            #For numerical stabiblity of the exponential
            Cc = np.min(Norm,axis=1).reshape((Q,1))#Most likely mask for each trace

            #Compute pdf with Bayes
            beta = np.exp(Cc-Norm) #Gaussian Kernel with shape (Q,2**N)
            S  = np.sum(beta,axis=1) #Normalisation coefficient in Bayes
            beta = beta / S.reshape((Q,1))#Normalised p.m.f

            #The M Step
            def objective0(Cvec):
                return np.sum(beta * (Y[0].reshape((Q,1)) - evaluate(Cvec,X0).reshape((Q,2**N))) ** 2) + regu * Q * norm(Cvec-ones)

            def jac0(Cvec):
                return - 2 * np.sum(beta.reshape((Q,2**N,1)) * (Y[0].reshape((Q,1,1)) - evaluate(Cvec,X0).reshape((Q,2**N,1))) * X0vec,axis=(0,1)) + regu * 2 * Q * (Cvec-ones)

            def objective1(Cvec):
                return np.sum(beta * (Y[1].reshape((Q,1)) - evaluate(Cvec,X1).reshape((Q,2**N))) ** 2) + regu *  Q * norm(Cvec-ones)
            def jac1(Cvec):
                return - 2 * np.sum(beta.reshape((Q,2**N,1)) * (Y[1].reshape((Q,1,1)) - evaluate(Cvec,X1).reshape((Q,2**N,1))) * X1vec,axis=(0,1)) + regu * 2 * Q * (Cvec-ones)

            #Update parameters
            Cvec0 = minimize(objective0,Cvec0,method="BFGS",jac=jac0,options={'gtol': 1e-04, 'maxiter': 10}).x
            Cvec1 = minimize(objective1,Cvec1,method="BFGS",jac=jac1,options={'gtol': 1e-04, 'maxiter': 10}).x

            #Does the new values of a and b changed up to a certain tolerance level ?
            ll = ((np.sum(np.log(S)) - np.sum(Cc))/Q)
            stop = np.abs(ll-last_ll) < tolerance

            #Store obtained goodness of fit
            last_ll = ll

        LL[k] = np.sum(np.log(S)) - np.sum(Cc) #Store the log-likelihood of the key

    k_guess = np.argsort(-LL)

    return k_guess


"""
This implements a template attack with quadratic leakage model.
Input:
    T      : the plaintext used for the considered traces.
    Y      : the traces
    sigma  : the noise level
    Cvec0  : model parameters (Quadratic form of the first share)
    Cvec1  : model parameters (Quadratic form of the second share)
"""
def template_quadratic(T,Y,sigma,Cvec0,Cvec1):

    Q = len(T)

    Y/=(2 ** .5 * sigma)
    Cvec0/=(2 ** .5 * sigma[0])
    Cvec1/=(2 ** .5 * sigma[1])

    LL = np.zeros(2**N) #To store the log-likelihood of each key hypothesis
    Text,Mask = np.meshgrid(T,M, copy = False)
    X0 = np.unpackbits(Mask).reshape((2**N,Q,8))[:,:,8-N:8]

    for k in range(2**N):
        X1 = np.unpackbits(Sbox(k^Text)^Mask).reshape((2**N,Q,8))[:,:,8-N:8]

        # Precompute the Quadratic Form
        f0 = evaluate(Cvec0,X0).reshape((Q,2**N))
        f1 = evaluate(Cvec1,X1).reshape((Q,2**N))

        Norm =  (Y[0].reshape((Q,1)) - f0)**2
        Norm += (Y[1].reshape((Q,1)) - f1)**2 #Shape (Q,2**N)

        Cc = np.min(Norm,axis=1).reshape((Q,1))#Most likely mask for each trace
        beta = np.exp(Cc-Norm) #Gaussian Kernel with shape (Q,2**N)
        S  = np.sum(beta,axis=1) #Normalisation coefficient in Bayes

        LL[k] = np.sum(np.log(S)) - np.sum(Cc) #Store the log-likelihood of the key
    k_guess = np.argsort(-LL)

    return k_guess

"""
Subfonction that instantiates the sensitive variable matrices.
"""
def x_synth_biv(k,Text,Mask):
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
def metrics(ListQ=[5,10,50,100],File='test.txt',sigma=1,sigma_a=0.4,NbIter=5):

    Sigma = np.array([sigma,sigma]).reshape((2,1))
    convergence_QUAD_LAM1,convergence_QUAD_LAM2,convergence_HAM,convergence_TEMP,convergence_CPA = False,False,False,False,False

    for Q in ListQ:

        print("I start Q = ",Q)

        SR_QUAD_LAM2, SR_QUAD_LAM1 ,SR_HAM, SR_TEMP, SR_CPA = [],[],[],[],[]
        GE_QUAD_LAM2, GE_QUAD_LAM1 ,GE_HAM, GE_TEMP, GE_CPA = [],[],[],[],[]
        t_QUAD_LAM2,  t_QUAD_LAM1  ,t_HAM , t_TEMP , t_CPA  = 0,0,0,0,0

        for i in range(NbIter):

            secretC0 =  np.ones(Nparams) + sigma_a * np.random.randn(Nparams)
            secretC0 *= np.sqrt(Nparams) / norm(secretC0)

            secretC1 = np.ones(Nparams) + sigma_a * np.random.randn(Nparams)
            secretC1 *= np.sqrt(Nparams) / norm(secretC1)

            secret_k = np.random.randint(0,2**N) #The secret key
            T = np.random.randint(0,2**N,Q ,dtype=np.uint8) #The plaintext
            Mask = np.random.randint(0,2**N,Q ,dtype=np.uint8) #The Mask

            Noise0 = sigma * np.random.randn(Q) #The Noise
            Noise1 = sigma * np.random.randn(Q) #The Noise

            X1 = np.unpackbits(Sbox(np.uint8(secret_k)^T)^Mask).reshape((Q,8))[:,8-N:8]
            X0 = np.unpackbits(Mask).reshape((Q,8))[:,8-N:8]

            Y0 = np.einsum("qj,ji,qi->q",X0,vec_to_sym(secretC0[0:Nparams-1]),X0) + secretC0[Nparams-1]
            Y0 += Noise0

            Y1 = np.einsum("qj,ji,qi->q",X1,vec_to_sym(secretC1[0:Nparams-1]),X1) + secretC1[Nparams-1]
            Y1 += Noise1

            Y = np.array([Y0,Y1])

            print("\n")
            print("=============================================")
            print("============== Q : ",Q, "  ==================")

            if not convergence_CPA:

                YY = np.copy(Y)
                t1 = time()
                k_g = HO_CPA(T,YY)
                t2 = time()

                GE_CPA.append(np.argmax(k_g == secret_k))
                SR_CPA.append(k_g[0] == secret_k)
                print(" 2O CPA Rank : ",np.argmax(k_g == secret_k))
                t_CPA += t2-t1

            if not convergence_QUAD_LAM1:

                YY = np.copy(Y)
                t1 = time()
                k_g = EM_quadratic(T,YY,Sigma,regu=1)
                t2 = time()

                GE_QUAD_LAM1.append(np.argmax(k_g == secret_k))
                SR_QUAD_LAM1.append(k_g[0] == secret_k)

                print("EM QUAD LAM1 Rank : ",np.argmax(k_g == secret_k))
                t_QUAD_LAM1 += t2-t1

            if not convergence_QUAD_LAM2:

                YY = np.copy(Y)
                t1 = time()
                k_g = EM_quadratic(T,YY,Sigma,regu=2)
                t2 = time()

                GE_QUAD_LAM2.append(np.argmax(k_g == secret_k))
                SR_QUAD_LAM2.append(k_g[0] == secret_k)

                print("EM QUAD LAM2 Rank : ",np.argmax(k_g == secret_k))
                t_QUAD_LAM2 += t2-t1

            if not convergence_HAM:

                YY = np.copy(Y)
                t1 = time()
                k_g = EM_bivarie_2(T,YY,Sigma)
                t2 = time()

                GE_HAM.append(np.argmax(k_g == secret_k))
                SR_HAM.append(k_g[0] == secret_k)

                print("EM HAM Rank : ",np.argmax(k_g == secret_k))
                t_HAM += t2-t1

            if not convergence_TEMP:

                YY = np.copy(Y)
                t1 = time()
                k_g = template_quadratic(T,YY,Sigma,secretC0,secretC1)
                t2 = time()

                GE_TEMP.append(np.argmax(k_g == secret_k))
                SR_TEMP.append(k_g[0] == secret_k)

                print("ML TEMP Rank : ",np.argmax(k_g == secret_k))

                t_TEMP += t2-t1

            print("=============================================")

        if not convergence_CPA:
            convergence_CPA = (np.mean(SR_CPA) > 0.995)
            with open(File+"_20CPA", "a") as f:
                f.write(str(NbIter) + " " + str(Q) + " " +  str(sigma) + " " + str(t_CPA/NbIter) + " " +str(np.mean(SR_CPA)) + " " + str(np.std(SR_CPA)) + " " + str(np.mean(GE_CPA)) + " " + str(np.std(GE_CPA)) + "\n")
            f.close()

        if not convergence_QUAD_LAM1:
            convergence_QUAD_LAM1 = (np.mean(SR_QUAD_LAM1) > 0.995)
            with open(File+"_EM_QUAD_LAM1", "a") as f:
                f.write(str(NbIter) + " " + str(Q) + " " +  str(sigma) + " " + str(t_QUAD_LAM1/NbIter) + " " +str(np.mean(SR_QUAD_LAM1)) + " " + str(np.std(SR_QUAD_LAM1)) + " " + str(np.mean(GE_QUAD_LAM1)) + " " + str(np.std(GE_QUAD_LAM1)) + "\n")
            f.close()

        if not convergence_QUAD_LAM2:
            convergence_QUAD_LAM2 = (np.mean(SR_QUAD_LAM2) > 0.995)
            with open(File+"_EM_QUAD_LAM2", "a") as f:
                f.write(str(NbIter) + " " + str(Q) + " " +  str(sigma) + " " + str(t_QUAD_LAM2/NbIter) + " " +str(np.mean(SR_QUAD_LAM2)) + " " + str(np.std(SR_QUAD_LAM2)) + " " + str(np.mean(GE_QUAD_LAM2)) + " " + str(np.std(GE_QUAD_LAM2)) + "\n")
            f.close()

        if not convergence_HAM:
            convergence_HAM = (np.mean(SR_HAM) > 0.995)
            with open(File+"_EM_HAM", "a") as f:
                f.write(str(NbIter) + " " + str(Q) + " " +  str(sigma) + " " + str(t_HAM/NbIter) + " " +str(np.mean(SR_HAM)) + " " + str(np.std(SR_HAM)) + " " + str(np.mean(GE_HAM)) + " " + str(np.std(GE_HAM)) + "\n")
            f.close()

        if not convergence_TEMP:
            convergence_TEMP = (np.mean(SR_TEMP) > 0.995)
            with open(File+"_ML_QUAD", "a") as f:
                f.write(str(NbIter) + " " + str(Q) + " " +  str(sigma) + " " + str(t_TEMP/NbIter) + " " +str(np.mean(SR_TEMP)) + " " + str(np.std(SR_TEMP)) + " " + str(np.mean(GE_TEMP)) + " " + str(np.std(GE_TEMP)) + "\n")
            f.close()

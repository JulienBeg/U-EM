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


"""
We use the definition of:
Taylor Expansion of ML Attacks for Masked and Shuffled Implementations
"""
def CenteredProduct(T,Y,N=4,sigma=.5):

    ybar = np.reshape(np.mean(Y,axis=1),(2,1))
    Yp = np.prod(Y - ybar,axis=0) #Belong to R^Q
    k_guess = CPA(T,Yp,N=N,sigma=2*sigma)
    return k_guess


def minimize_u(beta,X,Y,Q):

    x_tilde = np.sum(beta * X, axis = 0).T #Should belong to R^Q
    x_bar = np.mean(x_tilde,axis = 0) #Should belong to R

    var_x = (np.sum(beta * (X-x_bar) ** 2) / Q)
    cov = np.sum(beta * (X-x_bar) *Y) / Q

    na = cov / var_x
    b = - na * x_bar

    return na,b

def EM_bivarie_2(T,Y,sigma=np.reshape(np.array([1,1]),(2,1)),N=4,tolerance=10**-10):

    Q = len(T)

    #Normalisation and centering
    Y/=(2 ** .5 * sigma)
    y_bar = np.reshape(np.mean(Y,axis=1),(2,1)) #Precompute the average of the traces
    Y-=y_bar
    Y = np.reshape(Y,(2,1,Q))


    LL = np.zeros(2**N) #To store the log-likelihood of each key hypothesis

    Text,Mask = np.meshgrid(T, M, copy=False, sparse=True)

    X1 = np.reshape(np.char.count(binary_repr_vec(Mask), '1'),(len(Mask),1))

    for k in range(2**N):

        a1,a2 = 1,1
        b1,b2 = 0,0

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
            na1,b1 = minimize_u(beta,X0,Y[0],Q)
            na2,b2 = minimize_u(beta,X1,Y[1],Q)

            #Does the new values of a and b changed up to a certain tolerance level ?
            stop = abs(na1-a1) + abs(na2-a2) < tolerance

            a1,a2 = na1,na2

        LL[k] = np.sum(np.log(S)) - np.sum(C) #Store the log-likelihood of the key

    k_guess = np.argsort(-LL)
    print("EM-HAM",LL[k_guess[0]])
    return k_guess


def minimize(beta,X,Y,Mask,Q,N):

    x_tilde = np.sum(np.transpose(X,axes=(2,0,1)) * beta, axis = 1).T #Should belong to R^(QxN)
    x_bar = np.mean(x_tilde,axis = 0) #Should belong to R^N

    X_cent = np.reshape(X - x_bar,(len(Mask),Q,N,1))
    #X_cent = X - x_bar

    AutoCorr = np.einsum('mq,mqtp->tp' , beta,np.einsum('ijkl,ijhl->ijkh', X_cent,X_cent))
    RelCorr = np.einsum('qm,q->m' ,  x_tilde ,Y) #Since Y is centered no need to sub x_bar to x_tilde

    #na = np.linalg.solve(AutoCorr,RelCorr)
    na = np.linalg.lstsq(AutoCorr,RelCorr,rcond=None)[0]
    nb = - np.dot(x_bar,na)

    return na,nb

def EM_BIV_LIN2(T,Y,sigma=np.reshape(np.array([1,1]),(2,1)),N=4,tolerance=10**-8):

    Q = len(T)

    #Y/=(2 ** .5 * sigma)
    y_bar = np.reshape(np.mean(Y,axis=1),(2,1)) #Precompute the average of the traces
    Y-=y_bar

    LL = np.zeros(2**N) #To store the log-likelihood of each key hypothesis

    Text,Mask = np.uint8(np.meshgrid(T, M))

    X2 = np.reshape(np.unpackbits(Mask),(2**N,Q,8))[:,:,8-N:8]

    for k in range(2**N):

        for i in range(10):
            a_1,b_1 = -np.ones(N) * np.abs( (np.std(Y,axis=1)[0]/Q) ** 2 - sigma[0] **2) ** .5 * np.sqrt(N/4),0
            a_2,b_2 = -np.ones(N) * np.abs( (np.std(Y,axis=1)[1]/Q) ** 2 - sigma[1] **2) ** .5 * np.sqrt(N/4),0

            a_1+= np.linalg.norm(a_1)  * np.random.randn(N)
            a_2+= np.linalg.norm(a_2)  * np.random.randn(N)

            X1 = np.reshape(np.unpackbits(Sbox(k^Text)^Mask),(2**N,Q,8))[:,:,8-N:8]

            #LLk = 0

            stop = False
            while not stop:

                #The E Step
                Norm = ((Y[0,:]-np.dot(X1,a_1)-b_1)**2/(sigma[0]**2)+(Y[1,:]-np.dot(X2,a_2)-b_2)**2/(sigma[1] ** 2))/2
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

                #LLk1 = np.sum(np.log(S)) - np.sum(C)
                #stop = np.abs(LLk-LLk1) < tolerance
                #LLk = LLk1

                a_1,a_2 = na1,na2

            LL[k] = np.max([np.sum(np.log(S)) - np.sum(C),LL[k]]) #Store the log-likelihood of the key


    k_guess = np.argsort(-LL)
    print("EM-LIN",LL[k_guess[0]])
    return k_guess

def EM_Hybride(T,Y,sigma=np.reshape(np.array([1,1]),(2,1)),N=4,tolerance=10**-8):

    Q = len(T)

    Y/=(2 ** .5 * sigma)
    y_bar = np.reshape(np.mean(Y,axis=1),(2,1)) #Precompute the average of the traces
    Y-=y_bar

    LL = np.zeros(2**N) #To store the log-likelihood of each key hypothesis

    Text,Mask = np.uint8(np.meshgrid(T, M))

    X2 = np.reshape(np.unpackbits(Mask),(2**N,Q,8))[:,:,8-N:8]

    for k in range(2**N):

        a_1,b_1 = 1,0
        a_2,b_2 = np.ones(N),0

        X1 = np.reshape(np.char.count(binary_repr_vec(Mask), '1'),(len(Mask),1))
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

        LL[k] = np.sum(np.log(S)) - np.sum(C) #Store the log-likelihood of the key


    k_guess = np.argsort(-LL)

    return k_guess

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
    print("ML-LIN",LL[k_guess[0]])
    return k_guess

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
    #print("Execution de template en " + str(t2-t1))

    return k_guess

def x_synth_biv(k,Text,Mask,N=4):
    Q  = len(Text)
    k = np.uint8(k)
    l = np.reshape(np.unpackbits(Sbox(k^Text)^Mask),(Q,8))
    l1 = np.reshape(np.unpackbits(Mask),(Q,8))
    return l[:,8-N:8],l1[:,8-N:8]

def metrics(ListQ=[10,20,30],File="test.txt",sigma=1,NbIter=1000):

    sigma_a = .8
    N=4
    Sigma = np.reshape(np.array([sigma,sigma]),(2,1))

    convergence_EM_LIN,convergence_EM_HAM,convergence_ML_HAM,convergence_ML_LIN,convergence_2OCPA = False,False,True,False,True

    for Q in ListQ:

        SR_EM_LIN,SR_EM_HAM,SR_ML_LIN,SR_ML_HAM,SR_2OCPA = [],[],[],[],[]
        GE_EM_LIN,GE_EM_HAM,GE_ML_LIN,GE_ML_HAM,GE_2OCPA = [],[],[],[],[]
        t_EM_LIN,t_EM_HAM,t_ML_HAM,t_ML_LIN,t_2OCPA = 0,0,0,0,0

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

            X1,X2 = x_synth_biv(secret_k,T,Mask,N=N)
            Y1 = np.dot(X1,secret_a1) + Noise1 #The synthetic leaks
            Y2 = np.dot(X2,secret_a2) + Noise2 #The synthetic leaks
            Y = np.array([Y1,Y2])

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

            if not convergence_2OCPA:
                YY = np.copy(Y)
                t1=time()
                k_g = CenteredProduct(T,YY,N=4)
                t2 = time()
                GE_2OCPA.append(np.argmax(k_g == secret_k))
                SR_2OCPA.append(k_g[0] == secret_k)
                t_2OCPA += t2 - t1

        print("For Q = ",Q," we have :")

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

        if not convergence_2OCPA:
            print("SR_2OCPA : ",np.mean(SR_2OCPA)," GE_2OCPA : ",np.mean(GE_2OCPA)," t = ",t_2OCPA/NbIter)
            convergence_2OCPA = (np.mean(SR_2OCPA) > 0.995)
            with open(File+"_2OCPA", "a") as f:
                f.write(str(NbIter) + " " + str(Q) + " " +  str(sigma) + " " + str(t_2OCPA/NbIter) + " " +  str(np.mean(SR_2OCPA)) + " " + str(np.std(SR_2OCPA)) + " " + str(np.mean(GE_2OCPA)) + " " + str(np.std(GE_2OCPA)) + "\n")
            f.close()

def read_metrics(file):

    NbIter,ListQ,Sigma,t_exec,SR,std_SR,GE,std_GE = [],[],[],[],[],[],[],[]

    P = [NbIter,ListQ,Sigma,t_exec,SR,std_SR,GE,std_GE]

    with open(file, 'r') as f:

        _ = f.readline()
        ligne = f.readline()

        while ligne != "":
            L = list(map(float,ligne.split(" ")))
            for i in range(8):
                P[i].append(L[i])

            ligne = f.readline()

    for i in range(len(P)):
        P[i]=np.array(P[i])

    P[6]+=1

    Order = np.argsort(P[1])

    for i in range(8):
        P[i]=P[i][Order]

    return P

def read_dpa(file):

    ListQ,SR,GE, = [],[],[]

    P = [ListQ,SR,GE]

    with open(file, 'r') as f:

        _ = f.readline()
        ligne = f.readline()

        while ligne != "":
            L = list(map(float,ligne.split(" ")))
            for i in range(3):
                P[i].append(L[i])

            ligne = f.readline()

    for i in range(len(P)):
        P[i]=np.array(P[i])

    P[1]+=1

    Order = np.argsort(P[0])

    for i in range(3):
        P[i]=P[i][Order]

    return P

Files_Synth0 =  os.getenv("HOME")+"/TRACK/16_XXX_EM/Resultat_up/"+"_sigmaA08_sigma02_cor_"
Title0 = r"$\sigma_a = 0.8$ and $\sigma = 0.2$"

Files_Synth1 =  os.getenv("HOME")+"/TRACK/16_XXX_EM/Resultat_up/"+"_sigmaA08_sigma03_cor_"
Title1 = r"$\sigma_a = 0.8$ and $\sigma = 0.3$"

Files_Synth2 =  os.getenv("HOME")+"/TRACK/16_XXX_EM/Resultat_up/"+"_sigmaA04_sigma03_cor_"
Title2 = r"$\sigma_a = 0.4$ and $\sigma = 0.3$"

def plot_sr(File,title="Title"):

    Alg = ["_EM_HAM","_EM_LIN","_ML_HAM","_ML_LIN","_2OCPA","_ridge"]
    Color = ["darkgreen","blue","lawngreen","cyan","black","red"]
    LineStyle =["-","-","--","--","-","-"]
    Label = ["EM-HAM","EM-LIN","ML-HAM","ML-LIN","2OCPA",r"EM-HYB-$(2 * \sigma_a^2)^{-1}$"]

    for i in range(5):
        P = read_metrics(File+Alg[i])
        plt.plot(P[1],P[4],label=Label[i],color=Color[i],linestyle =LineStyle[i])
        plt.fill_between(P[1],P[4] - P[5] / P[0] ** .5, P[4] + P[5] / P[0] ** .5, alpha=0.5,color=Color[i])
    #plt.title(title)
    plt.xlabel("Number of Traces Q",fontsize='x-large')
    plt.ylabel("Sucess Rate",fontsize='x-large')
    plt.legend(fontsize='large')
    plt.grid()

    plt.savefig(os.getenv('HOME')+'/'+title,format = 'pdf', bbox_inches='tight', dpi = 600)

    plt.show()

def plot_ge(File,title="Title"):

    Alg = ["_EM_HAM","_EM_LIN","_ML_HAM","_ML_LIN","_2OCPA"]
    Color = ["darkgreen","blue","lawngreen","cyan","black"]
    LineStyle =["-","-","--","--","-"]
    Label = ["EM-HAM","EM-LIN","ML-HAM","ML-LIN","2OCPA"]

    for i in range(5):
        P = read_metrics(File+Alg[i])
        plt.plot(P[1],P[6],label=Label[i],color=Color[i],linestyle =LineStyle[i])
        plt.fill_between(P[1],P[6] - P[7] / P[0] ** .5, P[6] + P[7] / P[0] ** .5, alpha=0.5,color=Color[i])

    #plt.title(title)
    plt.xlabel("Number of Traces Q",fontsize='x-large')
    plt.ylabel("Guessing Entropy",fontsize='x-large')
    plt.semilogy(basey=2)
    plt.legend(fontsize='large')
    plt.grid()

    plt.savefig(os.getenv('HOME')+'/'+title,format = 'pdf', bbox_inches='tight', dpi = 600)

    plt.show()

from scipy.stats import linregress

def plot_t(File,title="Title"):

    Alg = ["_EM_HAM","_EM_LIN","_ML_HAM","_ML_LIN","_2OCPA"]
    Color = ["red","blue","green","cyan","black"]
    LineStyle =["-","-","--","--","-"]

    for i in range(5):
        P = read_metrics(File+Alg[i])
        plt.plot(P[1],P[3],label=Alg[i][1:],color=Color[i])

        reg = linregress(P[1],P[3])
        print(Alg[i])
        print(reg.slope)
        print("\n")


    plt.title(title)
    plt.xlabel("Number of Traces Q")
    plt.ylabel("Execution Time")
    plt.legend()
    plt.grid()
    plt.show()

dir = os.getenv("HOME") + "/TRACK/16_XXX_EM/Resultat/"

def plot_ge_dpa(Files=[dir + "_DPA_k0",dir + "_DPA_k1",dir + "_DPA_k2",dir + "_DPA_k3"],title="Title"):

    Alg = ["_EM_HAM","_ML_HAM","_ML_LIN","_2OCPA"]
    Color = ["red","green","cyan","black"]
    LineStyle =["-","--","--","-"]

    for i in range(4):

        min = 80

        ge = np.zeros(min)

        for File in Files:
            [ListQ,GE,SR] = read_dpa(File+Alg[i])

            min = np.min([min,len(ListQ)])

            ge = ge[0:min] + GE[0:min]


        plt.plot(5 * np.arange(min),ge/len(Files),label=Alg[i][1:],color=Color[i],linestyle =LineStyle[i])

    min = 20
    ge = np.zeros(min)
    for File in Files:
        print(File)
        [ListQ,GE,SR] = read_dpa(File+"_EM_LIN")
        min = np.min([min,len(ListQ)])
        ge = ge[0:min] + GE[0:min]

    plt.plot(50 + 25 * np.arange(min),ge/len(Files),label="EM_LIN",color="blue",linestyle ="-")


    plt.title(title)
    plt.xlabel("Number of Traces Q")
    plt.ylabel("Guessing Entropy")
    plt.semilogy(basey=2)
    plt.legend()
    plt.grid()
    plt.show()

def plot_sr_dpa(Files=[dir + "_DPA_k0",dir + "_DPA_k1",dir + "_DPA_k2",dir + "_DPA_k3"],title="Title"):

    Alg = ["_EM_HAM","_ML_HAM","_ML_LIN","_2OCPA"]
    Color = ["red","green","cyan","black"]
    LineStyle =["-","--","--","-"]

    for i in range(4):

        min = 80

        sr = np.zeros(min)

        for File in Files:
            print(File)
            [ListQ,GE,SR] = read_dpa(File+Alg[i])

            min = np.min([min,len(ListQ)])

            sr = sr[0:min] + SR[0:min]


        plt.plot(5 * np.arange(min),sr/len(Files),label=Alg[i][1:],color=Color[i],linestyle =LineStyle[i])

    min = 20
    ge = np.zeros(min)
    for File in Files:
        [ListQ,GE,SR] = read_dpa(File+"_EM_LIN")
        min = np.min([min,len(ListQ)])
        sr = sr[0:min] + SR[0:min]

    plt.plot(50 + 25 * np.arange(min),sr/len(Files),label="EM_LIN",color="blue",linestyle ="-")

    plt.title(title)
    plt.xlabel("Number of Traces Q")
    plt.ylabel("Sucess Rate")
    plt.legend()
    plt.grid()
    plt.show()

Files_DPA  = [dir + "_DPA_k" + str(folder) for folder in range(16)]


def plot_ge_dpa_up(title="Guessing Entropy on the DPAv2 Contest"):

    folder_dir = os.getenv("HOME") + "/TRACK/16_XXX_EM/Resultat_up"

    Alg = ["_EM_HAM","_EM_LIN","_ML_HAM","_ML_LIN","_2OCPA","_EM_Hybride-cor_LAMBDA_01","_EM_Hybride-cor_LAMBDA_1","_EM_Hybride-cor_LAMBDA_5"]
    Color = ["darkgreen","blue","lawngreen","cyan","black","red","darkred","tomato"]
    LineStyle =["-","-","--","--","-","-","-","-"]
    Label = ["EM-HAM","EM-LIN","ML-HAM","ML-LIN","2OCPA",r"EM-HYB-$0.1$",r"EM-HYB-$1$",r"EM-HYB-$5$"]

    for j in range(8):
        print("On passe à : " +Label[j])
        [ListQ,GE,SR] = read_dpa(folder_dir+"/UP_DPA_k0"+Alg[j])
        for i in range(1,16):
            print(i)
            print(Alg[j]+ " : "+str(i))
            [ListQi,GEi,SRi] = read_dpa(folder_dir+"/UP_DPA_k"+str(i)+Alg[j])
            GE+=GEi
        plt.plot(ListQ,GE/16,label=Label[j],color=Color[j],linestyle =LineStyle[j])

    #plt.title(title)
    plt.xlabel("Number of Traces Q",fontsize='x-large')
    plt.ylabel("Guessing Entropy",fontsize='x-large')
    plt.semilogy(basey=2)
    plt.legend(fontsize='large')
    plt.grid()
    plt.savefig(os.getenv('HOME')+'/'+title,format = 'pdf', bbox_inches='tight', dpi = 600)
    plt.show()

def plot_sr_dpa_up(title="Succes Rate on the DPAv2 Contest"):

    folder_dir = os.getenv("HOME") + "/TRACK/16_XXX_EM/Resultat_up"

    Alg = ["_EM_HAM","_EM_LIN","_ML_HAM","_ML_LIN","_2OCPA","_EM_Hybride-cor_LAMBDA_01","_EM_Hybride-cor_LAMBDA_1","_EM_Hybride-cor_LAMBDA_5"]
    Color = ["darkgreen","blue","lawngreen","cyan","black","red","darkred","tomato"]
    LineStyle =["-","-","--","--","-","-","-","-"]
    Label = ["EM-HAM","EM-LIN","ML-HAM","ML-LIN","2OCPA",r"EM-HYB-$0.1$",r"EM-HYB-$1$",r"EM-HYB-$5$"]

    for j in range(8):
        [ListQ,GE,SR] = read_dpa(folder_dir+"/UP_DPA_k0"+Alg[j])
        print("On passe à : " +Label[j])
        for i in range(1,16):
            print(i)
            [ListQi,GEi,SRi] = read_dpa(folder_dir+"/UP_DPA_k"+str(i)+Alg[j])
            SR+=SRi
        plt.plot(ListQ,SR/16,label=Label[j],color=Color[j],linestyle =LineStyle[j])

    #plt.title(title)
    plt.xlabel("Number of Traces Q",fontsize='x-large')
    plt.ylabel("Succes Rate",fontsize='x-large')
    #plt.semilogy(basey=2)
    plt.legend(fontsize='large')
    plt.grid()
    plt.savefig(os.getenv('HOME')+'/'+title,format = 'pdf', bbox_inches='tight', dpi = 600)
    plt.show()

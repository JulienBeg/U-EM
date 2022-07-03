from ExpectationMaximizationAttack import *


def trace_for_profiling(secret_a1,secret_a2,sigma,QP=300):

    N = 4

    ProfilingTrace = np.zeros((2**N,2,QP))
    ProfilingText = np.zeros((2**N,QP))

    for k_profiling in range(2**N):

        pT = np.random.randint(0,2**N,QP ,dtype=np.uint8) #The plaintext
        Mask = np.random.randint(0,2**N,QP ,dtype=np.uint8) #The Mask

        Noise1 = sigma * np.random.randn(QP) #The Noise
        Noise2 = sigma * np.random.randn(QP) #The Noise

        X1,X2 = x_synth_biv(k_profiling,pT,Mask,N=N)
        Y1 = np.dot(X1,secret_a1) + Noise1 #The synthetic leaks
        Y2 = np.dot(X2,secret_a2) + Noise2 #The synthetic leaks

        ProfilingTrace[k_profiling] = np.array([Y1,Y2])
        ProfilingText[k_profiling] = pT

    return ProfilingTrace,ProfilingText

def P_EM_BIV_LIN2_profiling(ProfilingT,ProfilingTrace,sigma=np.reshape(np.array([1,1]),(2,1)),N=4,tolerance=10**-8):

    Q = len(ProfilingT[0])

    Profiled_a1 = np.zeros((2**N,N))
    Profiled_b1 = np.zeros((2**N,1))

    Profiled_a2 = np.zeros((2**N,N))
    Profiled_b2 = np.zeros((2**N,1))

    for k in range(2 ** N):

        Y = ProfilingTrace[k]
        pT = ProfilingT[k]

        #Y/=(2 ** .5 * sigma)
        y_bar = np.reshape(np.mean(Y,axis=1),(2,1)) #Precompute the average of the traces
        Y-=y_bar

        LL = np.zeros(2**N) #To store the log-likelihood of each key hypothesis

        Text,Mask = np.uint8(np.meshgrid(pT, M))

        X2 = np.reshape(np.unpackbits(Mask),(2**N,Q,8))[:,:,8-N:8]


        a_1,b_1 = -np.ones(N) * np.abs( (np.std(Y,axis=1)[0]/Q) ** 2 - sigma[0] **2) ** .5 * np.sqrt(N/4),0
        a_2,b_2 = -np.ones(N) * np.abs( (np.std(Y,axis=1)[1]/Q) ** 2 - sigma[1] **2) ** .5 * np.sqrt(N/4),0


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

        Profiled_a1[k] = a_1
        Profiled_b1[k]= b_1

        Profiled_a2[k] = a_2
        Profiled_b2[k]= b_2

    return Profiled_a1,Profiled_a2,Profiled_b1,Profiled_b2

def P_EM(T,Y,ProfilingTrace,ProfilingT,sigma,N=4,tolerance=10**-8):

    Profiled_a1,Profiled_a2,Profiled_b1,Profiled_b2 = P_EM_BIV_LIN2_profiling(ProfilingT,ProfilingTrace,sigma,N)

    Q = len(T)
    #Y/=(2 ** .5 * sigma)
    y_bar = np.reshape(np.mean(Y,axis=1),(2,1)) #Precompute the average of the traces
    Y-=y_bar

    y_bar = np.mean(Y,axis=1) #Precompute the average of the traces
    LL = np.zeros(2**N) #To store the log-likelihood of each key hypothesis

    Text,Mask = np.uint8(np.meshgrid(T, M))

    X2 = np.reshape(np.unpackbits(Mask),(2**N,Q,8))[:,:,8-N:8]

    for k in range(2**N):

        a1 = Profiled_a1[k]
        a2 = Profiled_a2[k]
        b1 = Profiled_b1[k]
        b2 = Profiled_b2[k]

        X1 = np.reshape(np.unpackbits(Sbox(k^Text)^Mask),(2**N,Q,8))[:,:,8-N:8]
        Norm = ((Y[0,:]-np.dot(X1,a1)-b1)**2/(sigma[0]**2)+(Y[1,:]-np.dot(X2,a2)-b2)**2/(sigma[1] ** 2))/2
        #Norm = (Y[0,:]-np.dot(X1,a1)-b1)**2+(Y[1,:]-np.dot(X2,a2)-b2)**2
        C = np.min(Norm,axis=0) #For numerical stabiblity of the exponnetial
        beta = np.exp(C-Norm)
        S  = np.sum(beta,axis=0) #Normalisation coefficient

        LL[k] = np.sum(np.log(S)) - np.sum(C) #Store the log-likelihood of the key


    k_guess = np.argsort(-LL)
    #print("P-EM-LIN",LL[k_guess[0]])
    return k_guess



def metrics_PEM_noisy(ListQ=[10,20,30],File="test.txt",sigma=1,NbIter=1000,QP=300):

    sigma_a = .8
    N=4
    Sigma = np.reshape(np.array([sigma,sigma]),(2,1))

    convergence_EM_LIN,convergence_EM_HAM,convergence_ML_HAM,convergence_ML_LIN,convergence_2OCPA,convergence_P_EM = False,False,False,False,True,False

    for Q in ListQ:

        SR_EM_LIN,SR_EM_HAM,SR_ML_LIN,SR_ML_HAM,SR_2OCPA,SR_P_EM = [],[],[],[],[],[]
        GE_EM_LIN,GE_EM_HAM,GE_ML_LIN,GE_ML_HAM,GE_2OCPA,GE_P_EM = [],[],[],[],[],[]
        t_EM_LIN,t_EM_HAM,t_ML_HAM,t_ML_LIN,t_2OCPA,t_P_EM = 0,0,0,0,0,0

        for i in range(NbIter):

            #The leakage Coefficient (we set b to 0)
            secret_a1 = np.ones(N) + sigma_a * np.random.randn(N)
            secret_a1 = np.sqrt(N) *  secret_a1 / norm(secret_a1)

            secret_a2 = np.ones(N) + sigma_a * np.random.randn(N)
            secret_a2 = np.sqrt(N) *  secret_a2 / norm(secret_a2)
            
            #Noisy Coeff 
            secret_a1_n = np.ones(N) + sigma_a * np.random.randn(N)
            secret_a1_n = np.sqrt(N) *  secret_a1_n / norm(secret_a1_n)

            secret_a2_n = np.ones(N) + sigma_a * np.random.randn(N)
            secret_a2_n = np.sqrt(N) *  secret_a2_n / norm(secret_a2_n)
            
            

            secret_k = np.random.randint(0,2**N) #The secret key
            T = np.random.randint(0,2**N,Q ,dtype=np.uint8) #The plaintext
            Mask = np.random.randint(0,2**N,Q ,dtype=np.uint8) #The Mask

            Noise1 = sigma * np.random.randn(Q) #The Noise
            Noise2 = sigma * np.random.randn(Q) #The Noise

            X1,X2 = x_synth_biv(secret_k,T,Mask,N=N)
            Y1 = np.dot(X1,secret_a1) + Noise1 #The synthetic leaks
            Y2 = np.dot(X2,secret_a2) + Noise2 #The synthetic leaks
            Y = np.array([Y1,Y2])

            ProfilingTrace,ProfilingText = trace_for_profiling(secret_a1_n,secret_a2_n,sigma,QP=QP)

            if not convergence_P_EM:
                YY = np.copy(Y)
                t1=time()
                k_g = EM_bivarie_2(T,YY,sigma=Sigma,N=4,tolerance=10**-8)
                t2=time()
                GE_EM_HAM.append(np.argmax(k_g == secret_k))
                SR_EM_HAM.append(k_g[0] == secret_k)
                t_EM_HAM += t2-t1

            if not convergence_EM_HAM:
                YY = np.copy(Y)
                t1=time()
                k_g = P_EM(T,YY,ProfilingTrace,ProfilingText,sigma=Sigma,N=4,tolerance=10**-8)
                t2=time()
                GE_P_EM.append(np.argmax(k_g == secret_k))
                SR_P_EM.append(k_g[0] == secret_k)
                t_P_EM += t2-t1

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
                k_g = template_ham(T,YY,sigma=Sigma,a_1=np.mean(secret_a1_n),b_1=0,a_2=np.mean(secret_a2_n),b_2=0,N=4)
                t2=time()
                GE_ML_HAM.append(np.argmax(k_g == secret_k))
                SR_ML_HAM.append(k_g[0] == secret_k)
                t_ML_HAM += t2-t1

            if not convergence_ML_LIN:
                YY = np.copy(Y)
                t1=time()
                k_g = template2(T,YY,sigma=Sigma,a_1=secret_a1_n,b_1=0,a_2=secret_a2_n,b_2=0,N=4)
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

        if not convergence_P_EM:
            print("SR_P_EM : ",np.mean(SR_P_EM)," GE_P_EM : ",np.mean(GE_P_EM)," t = ",t_P_EM/NbIter)
            convergence_P_EM = (np.mean(SR_P_EM) > 0.995)
            with open(File+"_P_EM_", "a") as f:
                f.write(str(NbIter) + " " + str(Q) + " " +  str(sigma) + " " + str(t_P_EM/NbIter) + " " +str(np.mean(SR_P_EM)) + " " + str(np.std(SR_P_EM)) + " " + str(np.mean(GE_P_EM)) + " " + str(np.std(GE_P_EM)) + "\n")
            f.close()

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

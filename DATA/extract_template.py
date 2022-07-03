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


def y(k,t,m): #Leakage model
    return hw(binary_repr_vec(Sbox(k^t)^m),'1')+hw(binary_repr_vec(m),'1')

index = scipy.io.loadmat('index.mat')
key = index['key']
offset = index['offset']
plain = index['plain']


def read_file(file=os.getenv("HOME") + "/PoI_k0",Q=5000):
    Q = 5000
    T = np.zeros(Q,dtype=np.uint8)
    Y = np.zeros((Q,2),dtype=np.float32)

    with open(file,'r') as f:
        ligne = f.readline()

        for i in range(Q):
            ligne = f.readline()
            [Plain,Y1,Y2] = ligne.split(" ")
            T[i] = np.uint8(Plain)
            Y[i] = np.array([np.float(Y1),np.float32(Y2)])

    Y = Y.T

    return Y,T

def regression(folder_index=0,Q=5000):

    Y,T = read_file(file=os.getenv("HOME")+"/PoI_k"+str(folder_index))

    db_offset = 5000 * folder_index

    Sortie_Mask_coeff = np.reshape(np.unpackbits(Mask(np.mod(offset[:,0]+1,16))[db_offset:Q+db_offset]),(Q,8))
    Sortie_SBox_coeff = np.reshape(np.unpackbits((Sbox(plain[:,0]^key[:,0]) ^ Mask(np.mod(offset[:,0]+1,16)))[db_offset:Q+db_offset]),(Q,8))

    reg1 = LinearRegression().fit(Sortie_SBox_coeff,Y[0,:])
    reg2 = LinearRegression().fit(Sortie_Mask_coeff,Y[1,:])

    sigma1 = (np.mean(np.dot(Sortie_Mask_coeff[0:Q],reg1.coef_) + reg1.intercept_ - Y[1,:]) ** 2) ** .5
    sigma2 = (np.mean(np.dot(Sortie_SBox_coeff[0:Q],reg2.coef_) + reg2.intercept_ - Y[0,:]) ** 2) ** .5

    sigma_reg_a.append([sigma2,sigma1])

    a1_lin.append(reg1.coef_)
    b1_lin.append(reg1.intercept_)

    a2_lin.append(reg2.coef_)
    b2_lin.append(reg2.intercept_)

    #Matrice des poids de Hamming en sortie de Sbox  et de manipulation du masque pour le premier byte
    Sortie_SBox_ = hw(binary_repr_vec(Sbox(plain[:,0]^key[:,0]) ^ Mask(np.mod(offset[:,0]+1,16))),'1')[db_offset:Q+db_offset]
    Sortie_Mask_ = hw(binary_repr_vec(Mask(np.mod(offset[:,0]+1,16))),'1')[db_offset:Q+db_offset]

    reg1 = linregress(Sortie_SBox_[0:Q],Y[0,:])
    reg2 = linregress(Sortie_Mask_[0:Q],Y[1,:])

    a1_ham.append(reg1.slope)
    b1_ham.append(reg1.intercept)

    a2_ham.append(reg2.slope)
    b2_ham.append(reg2.intercept)

    sigma1 = np.mean((reg2.slope * Sortie_Mask_[0:Q] + reg2.intercept - Y[1,:]) ** 2) ** .5
    sigma2 = np.mean((reg1.slope * Sortie_SBox_[0:Q] + reg1.intercept - Y[0,:]) ** 2) ** .5
    sigma_reg.append([sigma2,sigma1])

sigma_reg_a = []
sigma_reg = []
a1_ham = []
b1_ham = []
a2_ham = []
b2_ham = []
a1_lin = []
a2_lin = []
b1_lin = []
b2_lin = []
K_s = [130,239,121,239,106,192,56,47,243,212,195,128,5,81,83,179]

for i in range(16):
    regression(folder_index=i)

scipy.io.savemat('template_extraction.mat',{
    'sigma_reg':sigma_reg,
    'sigma_reg_a':sigma_reg_a,
    'a1_ham':a1_ham,
    'b1_ham':b1_ham,
    'a2_ham':a2_ham,
    'b2_ham':b2_ham,
    'a1_lin':a1_lin,
    'a2_lin':a2_lin,
    'b1_lin':b1_lin,
    'b2_lin':b2_lin,
    'K_s':K_s
})

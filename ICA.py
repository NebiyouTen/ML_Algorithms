import argparse
import sys
import os
import numpy as np
import glob
#%matplotlib inline
import matplotlib . pyplot as plt
import matplotlib.image as mpimg
import librosa
import librosa.display as disp
from scipy import linalg as LA

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def ICA(M):

    M_nor = M -  np.mean(M,1, keepdims = True)
    # return M_nor
    interm_1 = np.matmul(M_nor,M_nor.T)
    # print ("in1 ",interm_1.shape)
    w, V= LA.eigh(interm_1)
    w = np.diag(w)
    w_i = np.sqrt(np.linalg.inv(w))
    C = np.dot(w_i, V.T)
    X = np.matmul(C,M_nor)
    # print ("shape of x ", X.shape)
    X_norm = LA.norm(X, axis=0)
    # print ("X norm shape is ",X_norm)
    # print ("norm is ",norm.shape,X.shape)
    # print (X.shape)
    # print ("X ",X)
    d = np.cov(X_norm * X)
    # print ("conv2 in my ",d)
    w,V = LA.eigh(d)
    H = np.matmul(V.T,X)
    return H, V.T

def main(argv):

    sound_dir = "../../hw2materials/problem3/*"
    files = glob.glob(sound_dir)

    # X = np.zeros((len(notes_15_files),64,64))
    M = np.zeros((2,132203))
    for i,file in enumerate(files):

        sample1, sr1  = librosa.load(file,sr = None)
        M[i] = sample1
        print (file,"loaded")
        # print (sample1.shape,sr1)
    print ("input is ",M.shape)
    H,W = ICA(M)
    print ("output is ",H.shape)
    output = np.zeros((2,132203)).astype(float)
    output[0] = H[0]
    output[1] = H[1]
    librosa.output.write_wav('source1.wav',3*output[0,:],sr = sr1)
    librosa.output.write_wav('source2.wav',3*output[1,:],sr = sr1)

    A = M.dot(np.linalg.pinv(H))
    np.savetxt("mixing_matrix.csv",A)
    # print (A)
    # print (np.linalg.inv(W))


if __name__ == '__main__':
    main(sys.argv)

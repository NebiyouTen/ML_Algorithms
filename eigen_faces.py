import argparse
import sys
import os
import numpy as np
import glob
#%matplotlib inline
import matplotlib . pyplot as plt
import matplotlib.image as mpimg

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def main(argv):

    image_dir = "../../hw2materials/problem2/lfw1000/*"
    notes_15_files = glob.glob(image_dir)
    X = np.zeros((len(notes_15_files),64,64))
    for i,note_file in enumerate(notes_15_files):
        img=mpimg.imread(note_file)
        X[i] = img
    X = X.reshape(len(notes_15_files),-1).T
    print ("Final shape",X.shape)
    U,S,V = np.linalg.svd(X,False)
    print (U.shape,S.shape,V.shape)
    im = U[:,1].reshape(64,64) #* S[0]
    print (U[:,10].shape,np.diag(S[:10]).shape,V[:10,].shape)

    erros = []
    k = 100

    for k in range(100):
        # x_hat = np.dot(U[:,:k],np.dot(np.diag(S[:k]),))
        X_hat = np.dot(U[:,:k],np.dot(np.diag(S[:k]),V[:k,]))
        erros.append(np.sum( (X-X_hat) **2 ))

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.plot(erros,label='error')
    plt.title('Reconstruction Error.')
    plt.tight_layout()
    plt.ylabel('Error')
    plt.xlabel('K')
    plt.show()

    first_eigen_face  = U[:,0]
    print ("Shape of first eigen face ",first_eigen_face.shape)
    np.save("eigenface.csv",first_eigen_face)
    plt.imshow(U[:,0].reshape(64,64))
    plt.show()



if __name__ == '__main__':
    main(sys.argv)

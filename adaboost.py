import argparse
import sys
import os
import numpy as np
import glob
#%matplotlib inline
import matplotlib . pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import pickle


from sklearn.tree import DecisionTreeClassifier

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

class Weak_Classifer(object):

    def __init__(self, t1, t2,t3,alpha) :
        self.t1 = t1
        self.t2 = t2
        self.t3 = t3
        self.tresh = t2
        self.alpha = alpha

    def forward(self,X_tr):
        # print ("Weak class forward first",X_tr.shape)

        X_feat = X_tr[:,self.t3]

        # print ("Weak class forward second",X_feat.shape)
        # tresh = X_tr[self.t2,self.t3]
        if self.t1 == 0:
            Y_pred = (X_feat >= self.tresh).astype(int)
            # print ("when T is 0",Y_pred.shape)
            Y_pred[Y_pred==0] = -1
            # print ("weak class output",Y_pred.shape)
            return  self.alpha * Y_pred
        else:
            # print (X_feat < tresh)
            Y_pred = (X_feat < self.tresh).astype(int)
            # print ("when T is 1",Y_pred.shape)
            Y_pred[Y_pred==0] = -1
            # print ("weak class output",Y_pred.shape)
            return  self.alpha * Y_pred

    def get_attribs(self):
        return self.t1,self.t2,self.t3,self.alpha

class AdaboostModel(object):

    def __init__(self) :
        self.weak_classifers = []

    def __add_classifer__(self,weak_classifer):
        self.weak_classifers.append(weak_classifer)

    def forward (self,X_tr, sign = True):
        pred = np.zeros((X_tr.shape[0]))
        for i,weak_classifer in enumerate(self.weak_classifers):
            # print ("IN model ",pred.shape)
            pred += weak_classifer.forward(X_tr)
        if sign:
            return np.sign(pred)
        return pred


def adaboost_train(X_tr,Y_tr,X_ts,Y_ts,T):
    N , M = X_tr.shape
    N_test =  len(X_ts)
    w = np.ones(N) / N

    alphas = []
    ada_boost_errors = []
    ada_boost_error_tests = []
    adaboost_model = AdaboostModel()

    for i in range(T):
        print ("Weak Classifer ",i+1,"out of ",T)
        errors = np.zeros((2,N,M))
        for feat in range(M):
            X_feat = X_tr[:,feat]
            for k,tresh in enumerate(X_feat):
                Y_pred = (X_feat >= tresh).astype(int)
                Y_pred[Y_pred==0] = -1
                miss = Y_pred != Y_tr
                error = np.sum(w[miss])#,np.abs(Y_pred[miss]))
                errors[0,k,feat] = error
                Y_pred = (X_feat < tresh).astype(int)
                Y_pred[Y_pred==0] = -1
                # print (Y_pred.shape)
                miss = Y_pred!=Y_tr
                error = np.sum(w[miss])#,np.abs(Y_pred[miss]))
                errors[1,k,feat] = error

        t1,t2,t3 = np.unravel_index(errors.argmin(), (2,N,M))
        err_m = errors[t1,t2,t3]

        if err_m >= 1/2:
            print ("Stopping condition")
            break

        alpha_m = 0.5 * np.log( (1 - err_m) / float(err_m))
        weak_classifier = Weak_Classifer(t1,X_tr[t2,t3],t3,alpha_m)
        multiplier = np.exp( -1 * alpha_m * np.sign(weak_classifier.forward(X_tr)) * Y_tr )
        w = np.multiply (w , multiplier)
        w = w / np.sum(w)
        adaboost_model.__add_classifer__(weak_classifier)

        Y_pred = adaboost_model.forward(X_tr)
        ada_boost_error = 1 - np.sum(Y_pred==Y_tr)/ N

        Y_test = adaboost_model.forward(X_ts)
        ada_boost_error_test = 1 - np.sum(Y_test==Y_ts)/ N_test

        ada_boost_errors.append(ada_boost_error)
        ada_boost_error_tests.append(ada_boost_error_test)



    return adaboost_model, ada_boost_errors, ada_boost_error_tests

def adaboost_predict(model,X_te):
    return model.forward(X_te)

def get_images(image_dir):

    images = glob.glob(image_dir+"train/face/*")
    X_face = np.zeros((len(images),19,19))
    for i,note_file in enumerate(images):
        img=Image.open(note_file)
        X_face[i] = img

    images_test = glob.glob(image_dir+"test/face/*")
    X_face_test = np.zeros((len(images_test),19,19))
    for i,note_file in enumerate(images_test):
        X_face_test[i] = Image.open(note_file)

    images = glob.glob(image_dir+"train/non-face/*")
    X_non_face = np.zeros((len(images),19,19))
    for i,note_file in enumerate(images):
        img=Image.open(note_file)
        X_non_face[i] = img

    images_test = glob.glob(image_dir+"test/non-face/*")
    X_non_face_test = np.zeros((len(images_test),19,19))
    for i,note_file in enumerate(images_test):
        X_non_face_test[i] = Image.open(note_file)

    Y_tr = np.ones(len(X_face)+len(X_non_face))
    Y_ts = np.ones(len(X_face_test)+len(X_non_face_test))

    Y_tr[len(X_face):] = -1
    Y_ts[len(X_face_test):] = -1

    print ("Loaded ", len(Y_tr), " training samples. Loaded ", len(Y_ts) , " testing samples")

    return np.concatenate([X_face,X_non_face],0), Y_tr,\
            np.concatenate([X_face_test,X_non_face_test],0), Y_ts

def main(argv):

    image_dir = "../../hw2materials/problem2/lfw1000/*"
    notes_15_files = glob.glob(image_dir)
    X = np.zeros((len(notes_15_files),19,19))
    for i,note_file in enumerate(notes_15_files):
        img=Image.open(note_file)
        img.thumbnail((19, 19), Image.ANTIALIAS)
        X[i] = img
        # break
    X = X.reshape(len(notes_15_files),-1).T
    U,S,V = np.linalg.svd(X,False)

    for i in [10, 30, 50]:

        eigen_faces_10 = U[:,:i]

        X_tr,Y_tr,X_ts,Y_ts = get_images("../../hw2materials/problem2/")
        # break
        print ("Read images ",X_tr.shape,X_ts.shape,Y_tr.shape,Y_ts.shape)
        X_tr_projected = np.matmul(X_tr.reshape(X_tr.shape[0],-1),eigen_faces_10)
        X_ts_projected = np.matmul(X_ts.reshape(X_ts.shape[0],-1),eigen_faces_10)
        print ("Read images ",X_tr_projected.shape)
        adaboost_model,adaboost_errors,ada_boost_error_tests = adaboost_train(\
                        X_tr_projected,Y_tr,X_ts_projected,Y_ts,200)
        with open("Adaboost_model"+str(i), "wb") as f:
            pickle.dump(adaboost_model, f, pickle.HIGHEST_PROTOCOL)
        np.save("adaboost_errors_test"+str(i),ada_boost_error_tests)
        np.save("adaboost_errors"+str(i),adaboost_errors)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(adaboost_errors,label='Training error')
        plt.title('classification error ')
        plt.tight_layout()
        plt.ylabel('Error')
        plt.xlabel('T')
        plt.show()
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(ada_boost_error_tests,label='Testing error')
        plt.title('classification error ')
        plt.tight_layout()
        plt.ylabel('Error')
        plt.xlabel('T')
        plt.show()

def get_error_rate(pred, Y):
    return sum(pred != Y) / float(len(Y))

if __name__ == '__main__':
    main(sys.argv)

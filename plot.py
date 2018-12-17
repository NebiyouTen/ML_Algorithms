#%matplotlib inline
import matplotlib . pyplot as plt
import matplotlib.image as mpimg


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


def plot(errors,title):


    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.plot(errors,label='Training error')
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

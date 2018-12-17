import argparse
import sys
import os
import numpy as np
import glob
#%matplotlib inline
import matplotlib . pyplot as plt
import matplotlib.image as mpimg
from PIL import Image, ImageDraw, ImageColor
import adaboost as A
import pickle
import my_get_local_max as my
from skimage.feature import peak_local_max

from sklearn.tree import DecisionTreeClassifier

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

scales = [1/6]#\#,1/4,1/2,1]

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


def gray_and_re_scale(directory,scales):
    image_files = glob.glob(directory)
    X = []

    for i,image_file in enumerate(image_files):

        # if i == 0 :
        #
        #     continue
        print (image_file,i)
        x_scale = []#= np.zeros((len(image_files), scale, scale))
        for scale in scales:
            img= Image.open(image_file).convert('L')
            # img.show()
            w,h = img.size
            # print (scale,w,h)
            img.thumbnail((scale*w,scale*h), Image.ANTIALIAS)
            im_array = np.array(img)
            # print (im_array.shape)
            x_scale.append(im_array)
        X.append(x_scale)
    return X

def scan_image(image,model,eigen_faces):
    start_x = start_y = 0

    # window = np.zeros((19,19))
    scores = []#np.zeros((len(image)))#{}
    print ("Number of images to be scanned ",len(image))
    for i in range(len(image)):
        current_image = image[i]
        N,M = current_image.shape

        score = np.zeros((N,M))
        print ("Current image size ",N,M)
        while True:
            if (start_y + 19 > M):
                break
            start_x = 0
            while start_x + 19 <= N :

                window = current_image[start_x:start_x+19,start_y:start_y+19]
                image_projected = np.matmul(window.reshape(1,-1),eigen_faces)
                # print ("In scan iamge ")
                s = model.forward(image_projected,False)
                # print (s)
                # print ()
                # if start_x==0:
                #     print (start_x,start_y,s)
                score[start_x,start_y] = s
                # print (score)

                start_x += 1
            if start_y % 15 == 0:
                print ("Looping through object ",start_y ,'/',  M)
            start_y += 1
        print ("after scainning score is ",score[0,:])
        scores.append(score)

    return scores

def get_local_max(scores,threshold):
    local_max_cords = []
    local_max_scores = []
    print ("Len of scores is ",len(scores))
    for i in range(len(scores)):

        score = scores[i]
        # print ("here score is ",score)
        max , x, y= my.my_get_localmax(score,threshold)
        # print (x,y)
        cord = np.zeros((len(max),3))
        cord[:,0] = i
        cord[:,1] = x
        cord[:,2] = y

        local_max_scores.append(max)
        local_max_cords.append(cord)
    return local_max_scores, local_max_cords

def fuse_boxes (local_max_cords):
    boxes = np.zeros((1,4))
    print ("Total max scores are ",len(local_max_cords))
    for i in range(len(local_max_cords)):
        score = local_max_cords[i]

        s = score[:,0].tolist()
        s_temp = np.zeros((len(s))).astype(int)
        s_temp[:] = s

        # s = np.array(score[:,0].tolist()).astype(int)
        # print (type(s))
        # print ("S is ",s)
        s = 1/np.array(scales)[s_temp]
        # print ("s is ",s.shape)
        x,y = score[:,1],score[:,2]
        x1,y1 =  s*x, s*y

        x2,y2 = s*(x+19), s*(y+19)
        box = np.zeros((len(x1),4))
        box[:,0],box[:,1],box[:,2],box[:,3] = x1,y1,x2,y2
        # box = np.array()
        # print (box.shape)
        boxes = np.concatenate([boxes,box],0)
        # print (score.shape)
    print ("final value is ",boxes[1:,:].shape)
    return boxes[1:,:]

def get_over_lap_boxes(box,fused_boxes,local_max_scores):
    overlap_boxes = []
    mask = np.ones_like(fused_boxes).astype(bool)
    mask_for_scores = np.ones_like(local_max_scores).astype(bool)

    # print ("Inside overlapping boxes")
    for i in range(len(fused_boxes)):
        cur_box = fused_boxes[i]
        x1c,y1c,x2c,y2c = cur_box[0],cur_box[1],cur_box[2],cur_box[3]
        x3c,y3c = x1c,y2c
        x4c,y4c = x2c,y1c
        x1,y1,x2,y2 = box[0],box[1],box[2],box[3]
        # print ("cur bock ",x1c,y1c,x2c,y2c)
        condition_1 = (x1 <= x1c <= x2) & (y1 <= y1c <= y2)
        condition_2 = (x1 <= x2c <= x2) & (y1 <= y2c <= y2)
        condition_3 = (x1 <= x3c <= x2) & (y1 <= y3c <= y2)
        condition_4 = (x1 <= x4c <= x2) & (y1 <= y4c <= y2)

        if condition_1 or condition_2 or condition_3 or condition_4:
             mask[i] = False
             mask_for_scores[i] = False
        # print (box[0],box[1],fused_boxes[i][0],fused_boxes[i][1])
    return fused_boxes[mask].reshape(-1,4),local_max_scores[mask_for_scores]

def nms(local_max_scores,fused_boxes):
    final_boxes = []
    print ("socres for \n",local_max_scores.shape," boxes are ",fused_boxes.shape)
    while len(fused_boxes) > 0:
        highest_scoring_block_index = np.argmax(local_max_scores,0)
        highest_scoring_block = fused_boxes[highest_scoring_block_index]
        final_boxes.append(highest_scoring_block)

        mask = np.ones_like(fused_boxes).astype(bool)
        mask_for_boxes = np.ones_like(local_max_scores).astype(bool)

        mask_for_boxes[highest_scoring_block_index] = False
        mask[highest_scoring_block_index] = False

        fused_boxes = fused_boxes[mask].reshape(-1,4)
        local_max_scores = local_max_scores[mask_for_boxes]

        fused_boxes,local_max_scores = get_over_lap_boxes(highest_scoring_block,fused_boxes,local_max_scores)
    return final_boxes

def write_boxes(boxes,im):
    for box in boxes:
        x1,y1,x2,y2 = box[0],box[1], box[2], box[3]
        # print ("rectangle is ",x1,y1,x2,y2)
        draw = ImageDraw.Draw(im)
        draw.rectangle(((x1,y1),(x2, y2)) , outline= "#ff0000")
    im.show()

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

    eigen_faces_50 = U[:,:50]

    image_dir = "../../hw2materials/problem2/photos/*"
    image_names = glob.glob(image_dir)
    images = gray_and_re_scale(image_dir,scales)#,730,800,600])

    with open("Adaboost_model50", "rb") as f:
        model = pickle.load(f)

    THRESHOLD = 1#0.5

    for i in range(len(images)):
        if i <= 0 :
            continue
        print ("Detecting image ",image_names[i])
        scores = scan_image(images[i], model, eigen_faces_50)
        print ("after scanning ",scores)
        local_max_scores, local_max_cords = get_local_max(scores,THRESHOLD)

        local_max_scores = np.concatenate([s for s in local_max_scores ],0)
        # print ("Loca mas scores \n",local_max_cords.shape,local_max_scores.shape)
        fused_boxes = fuse_boxes(local_max_cords)
        # print (fused_boxes)
        final_boxes = nms(local_max_scores,fused_boxes)

        print ("Length of final boxes ",len(final_boxes))

        print (final_boxes)

        im= Image.open(image_names[i])#.convert('RGB')
        write_boxes(final_boxes,im)

        # print (final_boxes)
        break

if __name__ == '__main__':
    main(sys.argv)

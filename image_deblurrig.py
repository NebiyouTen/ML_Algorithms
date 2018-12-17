import argparse
import sys
import os
import numpy as np
import glob
import matplotlib . pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import pickle


from sklearn.tree import DecisionTreeClassifier

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def P_x1_x2_y_x (x1, x2, y1, x, p_x1_y1, p_x2, p_x_y):
    val = 1 if x == x1 + x2 else 0
    num = val * p_x1_y1[y1,x1] * p_x2[x2]

    den = p_x_y[y1,x]
    if den == 0:
        return den
    return num/den


def EM(H, n_iter):

    height, width = H.shape
    shake = 15

    P_x2 = np.ones((shake)).astype(float)
    P_x2 = P_x2/ np.sum(P_x2)

    P_x1_y1 = np.ones((height,width-shake+1)).astype(float)
    P_x1_y1 = P_x1_y1/ np.sum(P_x1_y1)

    P_x_y = H / np.sum(H) # np.zeros_like(H).astype(float)

    for i in range(n_iter):

        '''
            compute P_x_y(X,Y)
        '''
        for x in range(width):
            for y in range(height):
                sum = 0
                for k in range(shake):
                    if (x - k) < 0 or (x - k) > (width - shake): continue
                    sum += P_x1_y1[y,x-k] * P_x2[k]
                # if sum == 0:
                #     print ("Sum is zero ",x,y)
                P_x_y[y,x] = sum
        P_x_y /= np.sum(P_x_y)
        print ("P_x_y computed num zero", np.sum(P_x_y==0))

        '''
        compute P_x1_y1
        '''
        for x1 in range(width-shake+1):
            for y1 in range(height):
                count = 0
                for x in range(width):
                    if x - x1 < 0 or x-x1 >= shake: continue
                    count += H[y1,x] * P_x1_x2_y_x (x1= x1,x2= x-x1,y1= y1,
                                        x= x, p_x1_y1 = P_x1_y1, p_x2= P_x2,
                                         p_x_y= P_x_y)
                P_x1_y1[y1,x1] = count
        P_x1_y1 =  P_x1_y1/np.sum(P_x1_y1)
        print ('P x1 y1 computed num zero', np.sum(P_x1_y1==0))

        '''
        compute P_x2_y2

        '''

        for x2 in range(shake):
            count = 0
            for y1 in range(height):
                for x in range(width):
                    if x - x2 < 0 or x-x2 > width - shake: continue
                    count += H[y1,x] * P_x1_x2_y_x (x1= x - x2,x2= x2,y1= y1,
                                        x= x, p_x1_y1 = P_x1_y1, p_x2= P_x2,
                                         p_x_y= P_x_y)
            P_x2[x2] = count

        P_x2 = P_x2 / np.sum(P_x2)
        print ('P x2 y2 computed ', np.sum(P_x2==0))
        print ("Done ------------------------------------------------------",i)

    img_min = np.min(P_x1_y1)
    P_x1_y1 -= img_min
    img_max = np.max(P_x1_y1)
    img = (P_x1_y1 - img_min ) / img_max

    plt.plot(P_x2)

    plt.xlabel('Px2')
    plt.ylabel('Probabilities')
    plt.title('Distribution of PX2')
    plt.grid(True)
    plt.show()

    plt.xlabel('Px1y1')
    plt.title('Heatmap ')
    plt.show()
    plt.imshow(P_x1_y1)
    plt.show()

    img = (img * 255).astype(np.uint8)
    print (img)
    im = Image.fromarray(img)
    im.show()


def main(argv):
    img= np.array(Image.open("carblurred.png"))
    im = Image.fromarray(img)
    im.show()

    print (img.shape,np.sum(img==0))

    EM(img,5)

if __name__ == '__main__':
    main(sys.argv)

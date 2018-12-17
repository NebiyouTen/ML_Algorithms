import numpy as np

def my_get_localmax(score_map=None, thresh=0):
    print ("max sore is",np.min(score_map))
    shifts = np.zeros((score_map.shape[0], score_map.shape[1], 9))
    shifts[:, :, 0] = score_map
    shifts[1:, 1:, 1] = score_map[:-1,:-1]
    shifts[1:, :, 2] = score_map[:-1,:]
    shifts[1:, :-1, 3] = score_map[:-1,1:]
    shifts[:, 1:, 4]  = score_map[:, :-1]
    shifts[:, :-1, 5] = score_map[:, 1:]

    shifts[:-1, 1:, 6] = score_map[: -1,: -1]
    shifts[:-1, :, 7] = score_map[: -1,:]
    shifts[:-1, :-1, 8] = score_map[: -1,1:]

    max, ind = np.max(shifts, 2), np.argmax(shifts, 2)
    # print ("max",max)
    # print ("Max is ",max.shape)
    max[ind!=0] = 0
    max = max.reshape(score_map.shape[0], score_map.shape[1])

    x,y = np.where(max>thresh)
    print ("max",max)
    # print ("inside max is ",x,y)
    return max[x,y],x,y

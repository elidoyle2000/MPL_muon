from typing import List
import numpy as np

"""
Ch0 (A) is left middle PMT (trigger)<br>
Ch1 (B) is right middle PMT<br>
Ch2 (C) Top Paddle<br>
Ch3 (D) Bottom Paddle<br>
"""
class channel():
    def __init__(self) -> None:
        self.left = 0
        self.right = 1
        self.top = 2
        self.bottom = 3
        self.leftAndRight = 0
        self.topAfterCombined = 1
        self.bottomAfterCombined = 2

        self.A = 0
        self.B = 1
        self.C = 2
        self.D = 3


Channel = channel()


def combineLeftAndRight(arr):
    new_arr = np.zeros((arr.shape[0], arr.shape[1] - 1))
    new_arr[:,0] = arr[:,0] + arr[:,1]
    new_arr[:,1] = arr[:,2]
    new_arr[:,2] = arr[:,3]

    new_arr = new_arr[new_arr[:,0]!=0, :]

    return new_arr


def purify(p1_h, p2_h, p1_t, p2_t) -> List:

    topFiresCondition:bool = (p1_h[:, Channel.top] != 0)
    topDoesntFireCondition:bool = (p1_h[:, Channel.top] == 0)
    leftFiresCondition:bool = (p1_h[:, Channel.left] != 0)
    leftDoesntFireCondition:bool = (p1_h[:, Channel.left] == 0)
    rightFiresCondition:bool = (p1_h[:, Channel.right] != 0)
    rightDoesntFireCondition:bool = (p1_h[:, Channel.right] == 0)
    bottomFiresCondition:bool = (p1_h[:, Channel.bottom] != 0)
    bottomDoesntFireCondition:bool = (p1_h[:, Channel.bottom] == 0)

    conditionArray = np.logical_and(topFiresCondition, np.logical_or(leftFiresCondition, rightFiresCondition))

    new_p1_h = p1_h[conditionArray]
    new_p2_h = p2_h[conditionArray]
    new_p1_t = p1_t[conditionArray]
    new_p2_t = p2_t[conditionArray]

    return new_p1_h, new_p2_h, new_p1_t, new_p2_t



def get_data(path, combineLandR=False, doPurify=False):
    # Use npz file
    muon = np.load(path)
    p1_h = muon['peak1_heights']
    p2_h = muon['peak2_heights']
    p1_t = muon['peak1_times']
    p2_t = muon['peak2_times']
    if doPurify:
        p1_h, p2_h, p1_t, p2_t = purify(p1_h, p2_h, p1_t, p2_t)
    if combineLandR:
        p1_h = combineLeftAndRight(p1_h)
        p2_h = combineLeftAndRight(p2_h)
        p1_t = combineLeftAndRight(p1_t)
        p2_t = combineLeftAndRight(p2_t)
    


    return p1_h, p2_h, p1_t, p2_t

import numpy as np
import cv2 as cv
from utilityFunctions.chaosFunctions import *

def calc_UACI(enc_img, orig_img):
    temp = cv.absdiff(orig_img, enc_img)/255
    return (np.sum(temp) * 100) / (temp.size)

def calc_NPCR(enc_img, orig_img):
    temp = cv.absdiff(orig_img, enc_img)
    temp = temp.astype(np.uint8)
    return (np.count_nonzero(temp) * 100) / temp.size

def createChaos(params, height, width):
    keys = []
    log_map_seed = params[0]
    log_map_r = params[1]
    log_map_on = params[2]

    lfsr_seed = np.binary_repr(int(str(int(params[3])), 2), width=8)
    lfsr_on = params[4]

    rossler_c = [params[5]]
    rossler_on = params[6]

    tent_map_seed = params[7]
    tent_map_r = params[8]
    tent_on = params[9]

    henon_map_x_seed = params[10]
    henon_map_y_seed = params[11]
    henon_map_a = params[12]
    henon_on = params[13]

    if log_map_on == 1:
        K1 = logistic_map(height, width, log_map_seed, log_map_r)
        keys.append(K1)

    # generating pseudorandom numbers - Linear feedback shift register
    if lfsr_on == 1:
        K2 = linear_shift_register(lfsr_seed, height, width)
        keys.append(K2)

    # generating pseudorandom numbers - Rossler map
    # params for Rossler map
    if rossler_on == 1:
        rossler_a = 0.1
        rossler_b = 0.1
        rossler_seed = [1, 1, 0]
        K3, y, z = rosslerMap(height, width, rossler_a, rossler_b, rossler_c, rossler_seed)
        keys.append(K3)

    # Tent map
    if tent_on == 1:
        K4 = tentMap(height, width, tent_map_seed, tent_map_r)
        keys.append(K4)

    # Henon map
    b = 0.3
    if henon_on == 1:
        K5 = henonMap(height, width, henon_map_x_seed, henon_map_y_seed, henon_map_a, b)
        keys.append(K5)

    K = np.array([np.binary_repr(i, width=8) for i in np.zeros((height * width), dtype=int)])
    for k in keys:
        K = np.array([np.binary_repr(int(i, 2) ^ int(j, 2), width=8) for i, j in zip(K.flatten(), k.flatten())])
    return K

    # K1 = logistic_map(height, width, log_map_seed, log_map_r)
    # keys.append(K1)
    #
    # # generating pseudorandom numbers - Linear feedback shift register
    # K2 = linear_shift_register(lfsr_seed, height, width)
    # keys.append(K2)
    #
    # # generating pseudorandom numbers - Rossler map
    # # params for Rossler map
    # rossler_a = 0.1
    # rossler_b = 0.1
    # rossler_seed = [1, 1, 0]
    # K3, y, z = rosslerMap(height, width, rossler_a, rossler_b, rossler_c, rossler_seed)
    # keys.append(K3)
    #
    # # Tent map
    # K4 = tentMap(height, width, tent_map_seed, tent_map_r)
    # keys.append(K4)
    #
    # # Henon map
    # b = 0.3
    # K5 = henonMap(height, width, henon_map_x_seed, henon_map_y_seed, henon_map_a, b)
    # keys.append(K5)
    #
    # K = np.array([np.binary_repr(i, width=8) for i in np.zeros((height * width), dtype=int)])
    #
    # for k in keys:
    #     K = np.array([np.binary_repr(int(i, 2) ^ int(j, 2), width=8) for i, j in zip(K.flatten(), k.flatten())])
    # return K

def convert2sequences(arr, SEQ_1, SEQ_2, L):
    NUM_PLANES = 4
    seq1 = np.zeros((L, NUM_PLANES), dtype="int")
    seq2 = np.zeros((L, NUM_PLANES), dtype="int")
    for count, i in enumerate(arr.flatten()):
        for j in range(NUM_PLANES):
            seq1[count, j] = np.binary_repr(int(i[SEQ_1[j]], 2), width=1)
            seq2[count, j] = np.binary_repr(int(i[SEQ_2[j]], 2), width=1)
    return seq1.flatten("F"), seq2.flatten("F")

def adaptPix(img, SEQ_1, SEQ_2, chaosParams):
    height, width = img.shape
    L = height * width
    bin_img = np.array([np.binary_repr(i, width=8) for i in img.flatten()])
    A1, A2 = convert2sequences(bin_img, SEQ_1, SEQ_2, L)
    K = createChaos(chaosParams, height, width)
    a1, a2 = convert2sequences(K, SEQ_1, SEQ_2, L)
    A11 = np.roll(A1, np.sum(A2))
    A21 = np.roll(A2, np.sum(A1))
    C1 = np.zeros((A11.shape), dtype=int)
    C2 = np.zeros((A21.shape), dtype=int)

    for i in range(C1.shape[0]):
        if i == 0:
            C1[i] = A11[i] ^ A11[C1.shape[0] - 1] ^ a1[i]
            C2[i] = A21[i] ^ A21[C1.shape[0] - 1] ^ a2[i]
        else:
            C1[i] = A11[i] ^ A11[i - 1] ^ a1[i]
            C2[i] = A21[i] ^ A21[i - 1] ^ a2[i]

    C1 = C1.reshape((height * width, 4))
    C2 = C2.reshape((height * width, 4))

    C = np.zeros((height * width, 8), dtype=int)
    for i in range(C.shape[0]):
        for count, (s1, s2) in enumerate(zip(SEQ_1, SEQ_2)):
            C[i, s1] = C1[i, count]
            C[i, s2] = C2[i, count]

    C_ = np.zeros((height * width), dtype=int)
    for i in range(C.shape[0]):
        temp = str(C[i, 0]) + str(C[i, 1]) + str(C[i, 2]) + str(C[i, 3]) \
               + str(C[i, 4]) + str(C[i, 5]) + str(C[i, 6]) + str(C[i, 7])
        C_[i] = int(temp, 2)

    C_ = C_.reshape((height, width)).astype("uint8")
    return C_, a1, a2
import os

from utilityFunctions.chaosFunctions import *
from utilityFunctions.utils import *
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import copy as cp
import random
import pandas as pd
import time
from skimage.measure import shannon_entropy

NUM = 15

np.random.seed(NUM)
random.seed(NUM)

def convert2sequences(arr, SEQ_1, SEQ_2, L):
    NUM_PLANES = 4
    seq1 = np.zeros((L, NUM_PLANES), dtype="int")
    seq2 = np.zeros((L, NUM_PLANES), dtype="int")
    for count, i in enumerate(arr.flatten()):
        for j in range(NUM_PLANES):
            seq1[count, j] = np.binary_repr(int(i[SEQ_1[j]], 2), width=1)
            seq2[count, j] = np.binary_repr(int(i[SEQ_2[j]], 2), width=1)
    return seq1.flatten("F"), seq2.flatten("F")

def createChaos(params, height, width):
    keys = []
    log_map_seed = params[0]
    log_map_r = params[1]

    lfsr_seed = np.binary_repr(int(str(int(params[3])), 2), width=8)

    tent_map_seed = params[7]
    tent_map_r = params[8]

    rossler_c = [params[5]]

    henon_map_x_seed = params[10]
    henon_map_y_seed = params[11]
    henon_map_a = params[12]

    K1 = logistic_map(height, width, log_map_seed, log_map_r)
    keys.append(K1)

    # generating pseudorandom numbers - Linear feedback shift register
    K2 = linear_shift_register(lfsr_seed, height, width)
    keys.append(K2)

    # generating pseudorandom numbers - Rossler map
    # params for Rossler map
    rossler_a = 0.1
    rossler_b = 0.1
    rossler_seed = [1, 1, 0]
    K3, y, z = rosslerMap(height, width, rossler_a, rossler_b, rossler_c, rossler_seed)
    keys.append(K3)

    # Tent map
    K4 = tentMap(height, width, tent_map_seed, tent_map_r)
    keys.append(K4)

    # Henon map
    b = 0.3
    K5 = henonMap(height, width, henon_map_x_seed, henon_map_y_seed, henon_map_a, b)
    keys.append(K5)

    K = np.array([np.binary_repr(i, width=8) for i in np.zeros((height * width), dtype=int)])

    for k in keys:
        K = np.array([np.binary_repr(int(i, 2) ^ int(j, 2), width=8) for i, j in zip(K.flatten(), k.flatten())])
    return K

def encryptImage(img, SEQ_1, SEQ_2, chaosParams):
    height, width = img.shape
    L = height * width
    bin_img = np.array([np.binary_repr(i, width=8) for i in img.flatten()])
    A1, A2 = convert2sequences(bin_img, SEQ_1, SEQ_2, L)
    K = createChaos(chaosParams, height, width)
    a1, a2 = convert2sequences(K, SEQ_1, SEQ_2, L)
    A11 = np.roll(A1, np.sum(a1))
    A21 = np.roll(A2, np.sum(a2))
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

def decryptImage(cipher, a1, a2, SEQ_1, SEQ_2):
    height, width = cipher.shape
    C_de = cipher.flatten()
    C_dec = np.zeros((height * width, 8), dtype=int)
    for i in range(C_de.shape[0]):
        temp = np.binary_repr(C_de[i], width=8)
        for j in range(len(temp)):
            C_dec[i, j] = temp[j]

    C1_dec = np.zeros((height * width, 4), dtype=int)
    C2_dec = np.zeros((height * width, 4), dtype=int)

    for i in range(C_dec.shape[0]):
        for count, (s1, s2) in enumerate(zip(SEQ_1, SEQ_2)):
            C1_dec[i, count] = C_dec[i, s1]
            C2_dec[i, count] = C_dec[i, s2]

    C1_dec = C1_dec.flatten()
    C2_dec = C2_dec.flatten()

    C11_dec = np.zeros((C1_dec.shape[0]), dtype=int)
    C21_dec = np.zeros((C2_dec.shape[0]), dtype=int)

    for i in range(C1_dec.shape[0]):
        if i == 0:
            C11_dec[i] = C1_dec[i] ^ a1[i]
            C21_dec[i] = C2_dec[i] ^ a2[i]
        else:
            C11_dec[i] = C1_dec[i] ^ a1[i]
            C21_dec[i] = C2_dec[i] ^ a2[i]

    for i in range(C1_dec.shape[0]):
        if i == 0:
            C11_dec[i] = C11_dec[i] ^ C11_dec[C1_dec.shape[0] - 1]
            C21_dec[i] = C21_dec[i] ^ C21_dec[C2_dec.shape[0] - 1]
        else:
            C11_dec[i] = C11_dec[i] ^ C11_dec[i - 1]
            C21_dec[i] = C21_dec[i] ^ C21_dec[i - 1]

    A1_dec = np.roll(C11_dec, -np.sum(a1))
    A2_dec = np.roll(C21_dec, -np.sum(a2))

    A1_dec = np.reshape(A1_dec, ((height * width), 4), order="F")
    A2_dec = np.reshape(A2_dec, ((height * width), 4), order="F")

    A = np.hstack((A1_dec, A2_dec))
    togu = []
    for i in A:
        togu.append("".join(map(str, i)))
    togu = np.array(togu)

    restored_img = np.zeros((height * width), dtype=int)
    for i in range(restored_img.shape[0]):
        restored_img[i] = int(togu[i], 2)
    restored_img = restored_img.reshape((height, width)).astype("uint8")
    return restored_img

os.chdir("/Users/rt/PycharmProjects/PixAdapt/")

img = cv.imread("FINAL DATABASE/images/image_5.jpeg", 0)
height, width = img.shape
# plt.imshow(img)
# plt.show()

# img = np.arange(25).reshape((5, 5))

SEQ_1 = [0, 1, 2, 3]
SEQ_2 = [4, 5, 6, 7]

params = [0.01, 3.99, 1,
          101111.0, 1,
          9.0, 1,
          0.01, 1.99, 1.0,
          0.01, 0.01, 1.5, 1.0]

change_x = np.random.randint(0, width)
change_y = np.random.randint(0, height)
val = np.random.randint(0, 255)

new_img = cp.deepcopy(img)
new_img[change_x, change_y] = val


temp = cp.deepcopy(img)
for i in range(1000):
    temp, a1, a2 = encryptImage(temp, SEQ_1, SEQ_2, params)
cipher_image = temp

temp = cp.deepcopy(new_img)

for i in range(1000):
    temp, a1_1, a2_1 = encryptImage(temp, SEQ_1, SEQ_2, params)
cipher_image1 = temp

print(calc_UACI(cipher_image, cipher_image1))










# restored_image = decryptImage(cipher_image, a1, a2, SEQ_1, SEQ_2)
# plt.imshow(img)
# plt.show()
# plt.imshow(cipher_image)
# plt.show()
# plt.imshow(restored_image)
# plt.show()

# print(f"NPCR : {calc_NPCR(cipher_image, img)}\nUACI : {calc_UACI(cipher_image, img)}\nEntropy : {shannon_entropy(cipher_image)}")

# print(np.array_equal(img, restored_image))
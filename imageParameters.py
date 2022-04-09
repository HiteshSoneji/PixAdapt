from utilityFunctions.chaosFunctions import *
from utilityFunctions.utils import *
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import copy as cp
import random
import pandas as pd

import os
from skimage.measure import shannon_entropy

os.chdir("/Users/rt/PycharmProjects/PixAdapt/")

NUM = 5

filepath = f'FINAL DATABASE/images/image_{NUM}.jpeg'

img = cv.imread(filepath, 0)

height, width = img.shape

P = convert_to_binary(img)

keys = []


params = [0.2529991714972711,
 3.89601789200077,
 0.0,
 101111.0,
 0.0,
 9.0,
 0.0,
 0.986362476096752,
 1.6348911246127822,
 1.0,
 0.5274343651124074,
 0.3429152007644213,
 1.2294119671353356,
 1.0]

# generating pseudorandom numbers - Logistic map

log_map_seed = params[0]
log_map_r = params[1]

lfsr_seed = np.binary_repr(int(str(int(params[3])), 2), width = 8)

tent_map_seed = params[7]
tent_map_r = params[8]

rossler_c = [params[5]]

henon_map_x_seed = params[10]
henon_map_y_seed = params[11]
henon_map_a = params[12]

SEQ_1 = [0, 1, 2, 3]
SEQ_2 = [4, 5, 6, 7]


K1 = logistic_map(height, width, log_map_seed, log_map_r)
keys.append(K1)

# generating pseudorandom numbers - Linear feedback shift register
K2 = linear_shift_register(lfsr_seed, height, width)
keys.append(K2)

# # generating pseudorandom numbers - Rossler map
# # params for Rossler map
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

# generating the encrypted image
P_PRIME = np.array([np.binary_repr(int(i, 2) ^ int(j, 2), width=8) for i, j in zip(K, P.flatten())])
chaos_encrypted_image = np.array([int(i, 2) for i in P_PRIME]).reshape((height, width)).astype('uint8')

# print(calc_UACI(chaos_encrypted_image, img))

print(f"Original image entropy : {round(shannon_entropy(img), 6)}")
print(f"Encrypted image entropy : {round(shannon_entropy(chaos_encrypted_image), 6)}")
print(f"NPCR : {round(calc_NPCR(chaos_encrypted_image, img), 6)}")

# plt.hist(img.astype(np.uint8).ravel(), bins=256, color = "lightblue")
# plt.title(f"Image {NUM} histogram before encryption")
# plt.xlabel("Intensity value")
# plt.ylabel("Count")
# plt.savefig(f"analysis/Histograms/Before/image_{NUM}_before.png")
# plt.show()
#
# plt.hist(chaos_encrypted_image.astype(np.uint8).ravel(), bins=256, color = "darkblue")
# plt.title(f"Image {NUM} histogram after encryption")
# plt.xlabel("Intensity value")
# plt.ylabel("Count")
# plt.savefig(f"analysis/Histograms/After/image_{NUM}_before.png")
# plt.show()
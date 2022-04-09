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

NUM = 6

filepath = f'FINAL DATABASE/images/image_{NUM}.jpeg'

img = cv.imread(filepath, 0)

height, width = img.shape

P = convert_to_binary(img)

keys = []


params = [0.01, 3.99, 1,
          100, 1,
          9, 1,
          0.01, 1.999, 1,
          0.01, 0.01, 1.99, 1]

# generating pseudorandom numbers - Logistic map

log_map_seed = params[0]
log_map_r = params[1]
log_map_on = params[2]

lfsr_seed = np.binary_repr(int(str(int(params[3])), 2), width = 8)
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

SEQ_1 = [0, 1, 2, 3]
SEQ_2 = [4, 5, 6, 7]

PARAMETERS = [log_map_seed, log_map_r, log_map_on,
              lfsr_seed, lfsr_on,
              rossler_c, rossler_on,
              tent_map_seed, tent_map_r, tent_on,
              henon_map_x_seed, henon_map_y_seed, henon_map_a, henon_on]

change_y = np.random.randint(0, width)
change_x = np.random.randint(0, height)
val = np.random.randint(0, 255)

new_img = cp.deepcopy(img)
new_img[change_x, change_y] = val

temp = cp.deepcopy(img)
for i in range(2):
    temp, a1, a2 = adaptPix(temp, SEQ_1, SEQ_2, PARAMETERS)
cipher_image = temp

temp = cp.deepcopy(new_img)

for i in range(2):
    temp, a1_1, a2_1 = adaptPix(temp, SEQ_1, SEQ_2, PARAMETERS)
cipher_image1 = temp

cv.imwrite(f"analysis/encryptedImages/NonAdaptive/image_{NUM}_enc.png", cipher_image)
print(f"ORIGINAL IMAGE ENTROPY : {round(shannon_entropy(img), 6)}")
print(f"Entropy : {round(shannon_entropy(cipher_image), 6)}")
print(f"NPCR : {round(calc_NPCR(cipher_image, cipher_image1), 6)}")
print(f"UACI : {round(calc_UACI(cipher_image, cipher_image1), 6)}")

plt.hist(img.astype(np.uint8).ravel(), bins=256, color = "pink")
plt.title(f"Image {NUM} histogram before encryption")
plt.xlabel("Intensity value")
plt.ylabel("Count")
plt.savefig(f"analysis/Histograms/Before/image_{NUM}_before.png")
plt.show()

plt.hist(cipher_image.astype(np.uint8).ravel(), bins=256, color = "pink")
plt.title(f"Image {NUM} histogram after encryption")
plt.xlabel("Intensity value")
plt.ylabel("Count")
plt.savefig(f"analysis/Histograms/After/image_{NUM}_after.png")
plt.show()
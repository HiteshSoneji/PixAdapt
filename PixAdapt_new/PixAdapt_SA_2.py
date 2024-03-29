import copy
from utilityFunctions.chaosFunctions import *
from utilityFunctions.utils import *
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd
import copy as cp
import time
import math
import os

# IMAGE = 1
# NUM = 14

# IMAGE = 2
# NUM = 23

# IMAGE = 3
# NUM = 69
# #
# IMAGE = 4
# NUM = 70
#
# IMAGE = 5
# NUM = 83
# #
# IMAGE = 6
# NUM = 95
#
# IMAGE = 7
# NUM = 96
# #
# IMAGE = 8
# NUM = 102
# #
# IMAGE = 9
# NUM = 103
#
IMAGE = 10
NUM = 104

np.random.seed(NUM)
random.seed(NUM)

def run_sim(filepath, params):
    start = time.time()
    img = cv.imread(filepath, 0)
    params_ = cp.deepcopy(params)

    log_map_seed = params_[0][0]
    log_map_r = params_[1][0]
    log_map_on = params_[2]

    lfsr_seed = np.binary_repr(params_[3][0], width=8)
    lfsr_on = params_[4]

    rossler_c = params_[5][0]
    rossler_on = params_[6]

    tent_map_seed = params_[7][0]
    tent_map_r = params_[8][0]
    tent_map_on = params_[9]

    henon_map_x_seed = params_[10][0]
    henon_map_y_seed = params_[11][0]
    henon_map_a = params_[12][0]
    henon_map_on = params_[13]

    params = [log_map_seed, log_map_r, log_map_on,
              lfsr_seed, lfsr_on,
              rossler_c, rossler_on,
              tent_map_seed, tent_map_r, tent_map_on,
              henon_map_x_seed, henon_map_y_seed, henon_map_a, henon_map_on]


    initial_temp = 10
    final_temp = 0.01
    alpha = 0.1
    wrap = lambda num, end: num % end
    epochs = 1

    current_temp = initial_temp
    best = params

    enc_uaci = encryptImage(img, best)
    best_eval = abs(33.46 - enc_uaci)
    print(f"Epochs : {epochs} | Cost : {round(best_eval, 4)}\n")

    curr, curr_eval = best, best_eval

    temp_data = cp.deepcopy(params)
    temp_data.append(enc_uaci)
    data = [temp_data]

    while current_temp > final_temp:
        temp_data = []
        epochs+=1
        log_map_seed = wrap(log_map_seed + random.gauss(0.1, 0.2), params_[0][1])
        log_map_r = wrap(log_map_r + random.gauss(0.1, 0.01), params_[1][1])
        log_map_on = random.choice([0, 1])

        lfsr_seed = np.binary_repr(wrap(int(lfsr_seed, 2) + random.choice(range(1, 10)), params_[3][1]), width=8)
        lfsr_on = random.choice([0, 1])

        rossler_c = wrap(rossler_c + random.gauss(0.1, 0.1), params_[5][1])
        rossler_on = random.choice([0, 1])

        tent_map_seed = wrap(tent_map_seed + random.gauss(0.1, 0.2), params_[7][1])
        tent_map_r = wrap(tent_map_r + random.gauss(0.1, 0.01), params_[8][1])
        tent_map_on = random.choice([0, 1])

        henon_map_x_seed = wrap(henon_map_x_seed + random.gauss(0.1, 0.2), params_[10][1])
        henon_map_y_seed = wrap(henon_map_y_seed + random.gauss(0.1, 0.2), params_[11][1])
        henon_map_a = wrap(henon_map_a + random.gauss(0.1, 0.01), params_[12][1])
        henon_map_on = random.choice([0, 1])

        candidate = [log_map_seed, log_map_r, log_map_on,
                     lfsr_seed, lfsr_on,
                     rossler_c, rossler_on,
                     tent_map_seed, tent_map_r, tent_map_on,
                     henon_map_x_seed, henon_map_y_seed, henon_map_a, henon_map_on]

        enc_uaci = encryptImage(img, candidate)

        candidate_eval = abs(33.46 - enc_uaci)

        temp_data = copy.deepcopy(candidate)
        temp_data.append(enc_uaci)

        data.append(temp_data)

        if candidate_eval < best_eval:
            best, best_eval = candidate, candidate_eval
            print(f"Epochs : {epochs} | Cost : {round(candidate_eval, 4)}\n")

        if best_eval < 0.01:
            break

        diff = candidate_eval - curr_eval
        metropolis = math.exp(-diff/current_temp)

        if diff < 0 or random.uniform(0, 1) < metropolis:
            curr, curr_eval = candidate, candidate_eval
        current_temp -= alpha
    end = time.time()
    time_taken = end - start
    print(f"Time taken : {round(time_taken, 6)} | time per pass = {time_taken/epochs}")
    return best, best_eval, data

def encryptImage(img, params):
    height, width = img.shape

    log_map_seed = params[0]
    log_map_r = params[1]
    log_map_on = params[2]

    lfsr_seed = params[3]
    lfsr_on = params[4]

    rossler_c = params[5]
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

    cv.imwrite(f"analysis/encryptedImages/SA/image_{IMAGE}_enc.png", cipher_image)
    print(f"ORIGINAL IMAGE ENTROPY : {round(shannon_entropy(img), 6)}")
    print(f"Entropy : {round(shannon_entropy(cipher_image), 6)}")
    print(f"NPCR : {round(calc_NPCR(cipher_image, cipher_image1), 6)}")
    print(f"UACI : {round(calc_UACI(cipher_image, cipher_image1), 6)}")
    return calc_UACI(cipher_image, cipher_image1)


"""
Params which can be changed
    1. Logistic map
        1. Seed pixel value - (0.01, 1)
        2. R - (3.6, 4)

    2. Linear feedback shift register
        1. Seed pixel

    3. Rossler map
        1. a = 0.2
        2. b = 0.2
        3. c
        4. x0 - seed value

    4. Tent map
        1. tent map seed pixel
        2. R

    5. Henon map
        1. henon map x seed
        2. henon map y seed
        3. a
        4. b = 0.2

Fitness parameters
    1. NPCR - nearly equal to 99.60
    2. UACI - 33.4635

"""

log_map_seed = [0.01, 1]
log_map_r = [3.7, 4]
log_map_on = [0, 1]

lfsr_seed = [1, 255]
lfsr_on = [0, 1]

rossler_c = [9, 18]
rossler_on = [0, 1]

tent_map_seed = [0.01, 1]
tent_map_r = [1, 2]
tent_on = [0, 1]

henon_map_x_seed = [0.01, 1]
henon_map_y_seed = [0.01, 1]
henon_map_a = [1, 2]
henon_on = [0, 1]

os.chdir("//Users/hitesh/Desktop/PixAdapt/")

params = [log_map_seed, log_map_r, random.choice([0, 1]),
          lfsr_seed, random.choice([0, 1]),
          rossler_c, random.choice([0, 1]),
          tent_map_seed, tent_map_r, random.choice([0, 1]),
          henon_map_x_seed, henon_map_y_seed, henon_map_a, random.choice([0, 1])]

filepath = f'FINAL DATABASE/images/image_{IMAGE}.jpeg'
enc_parameters, enc_eval, data = run_sim(filepath, params)

cols = ["log_map_seed", "log_map_r", "log_map_on",
        "lfsr_seed", "lfsr_on",
        "rossler_c", "rossler_on",
        "tent_map_seed", "tent_map_r", "tent_on",
        "henon_map_x_seed", "henon_map_y_seed", "henon_map_a", "henon_on",
        "fitness"]

df = pd.DataFrame(data, columns=cols)
df.to_csv(f"csv files/simulated annealing/IMG_{IMAGE}.csv")

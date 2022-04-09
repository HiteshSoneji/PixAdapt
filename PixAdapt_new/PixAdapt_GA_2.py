from utilityFunctions.chaosFunctions import *
from utilityFunctions.utils import *
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import copy as cp
import random
import pandas as pd
import time
import os

# IMAGE = 1
# NUM = 14

# IMAGE = 2
# NUM = 23

# IMAGE = 3
# NUM = 69
#
# IMAGE = 4
# NUM = 70
#
# IMAGE = 5
# NUM = 83
#
# IMAGE = 6
# NUM = 95
#
# IMAGE = 7
# NUM = 96
#
# IMAGE = 8
# NUM = 102
#
# IMAGE = 9
# NUM = 103
#
IMAGE = 10
NUM = 104

np.random.seed(NUM)
random.seed(NUM)

def run_sim(filepath, genotype1):
    start = time.time()
    img_params = []
    img = cv.imread(filepath, 0)
    cost, PARAMETERS = encryptImage(img, genotype1)
    PARAMETERS.append(cost)
    img_params.append(PARAMETERS)
    print(f"After 1 generation, cost : {cost} | {print_genotype_vals(genotype1)}")
    print()
    epochs = 2

    while True:
        best = 33.46
        if abs(best - cost) < 0.01:
            break
        genotype2 = cp.deepcopy(genotype1)
        mutate_genotype(genotype2)
        cost2, PARAMETERS = encryptImage(img, genotype2)
        PARAMETERS.append(cost2)
        img_params.append(PARAMETERS)
        if abs(best - cost) > abs(best - cost2):
            cost = cost2
            genotype1 = cp.deepcopy(genotype2)
            print(f"After {epochs} generation, cost : {cost} | {print_genotype_vals(genotype1)}")
        epochs += 1
    end = time.time()
    print(f"OVERALL TIME TAKEN FOR {epochs} epochs : {round(end-start, 6)}")
    print(f"Genotype after simulation : {print_genotype_vals(genotype1)}\nFitness : {round(cost, 2)}\n\n")
    return img_params


def encryptImage(img, genotype):
    height, width = img.shape

    log_map_seed = genotype[0]["values"][genotype[0]["ind"]]
    log_map_r = genotype[1]["values"][genotype[1]["ind"]]
    log_map_on = genotype[2]["values"][genotype[2]["ind"]]

    lfsr_seed = genotype[3]["values"][genotype[3]["ind"]]
    lfsr_on = genotype[4]["values"][genotype[4]["ind"]]

    rossler_c = genotype[5]["values"][genotype[5]["ind"]]
    rossler_on = genotype[6]["values"][genotype[6]["ind"]]

    tent_map_seed = genotype[7]["values"][genotype[7]["ind"]]
    tent_map_r = genotype[8]["values"][genotype[8]["ind"]]
    tent_on = genotype[9]["values"][genotype[9]["ind"]]

    henon_map_x_seed = genotype[10]["values"][genotype[10]["ind"]]
    henon_map_y_seed = genotype[11]["values"][genotype[11]["ind"]]
    henon_map_a = genotype[12]["values"][genotype[12]["ind"]]
    henon_on = genotype[13]["values"][genotype[13]["ind"]]

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

    cv.imwrite(f"analysis/encryptedImages/GA/image_{IMAGE}_enc.png", cipher_image)
    print(f"ORIGINAL IMAGE ENTROPY : {shannon_entropy(img)}")
    print(f"NPCR : {calc_NPCR(cipher_image, cipher_image1)}")
    print(f"Entropy : {shannon_entropy(cipher_image)} \n")
    return calc_UACI(cipher_image, cipher_image1), PARAMETERS


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


def mutate_gene(thing):
    ind = thing["ind"] + np.random.choice([-1, 1])
    if ind < 0:
        ind = thing["size"] - 1
    if ind == thing["size"]:
        ind = 0
    thing["ind"] = ind


def make_gene(values):
    return {"values": values, "size": len(values), "ind": np.random.randint(low=0, high=len(values))}


def mutate_genotype(genotype):
    ind = np.random.choice(len(genotype))
    mutate_gene(genotype[ind])


def evaluate_genotype(orig_img, enc_img):
    return calc_UACI(enc_img, orig_img)


def create_discrete_genes(start, end):
    return random.sample(range(start, end), 10)


def create_continous_genes(start, end):
    return np.random.uniform(start, end, size=10)


def print_genotype_vals(genotype):
    s = ''
    for i, gene in enumerate(genotype):
        vals = gene["values"]
        if isinstance(vals[gene["ind"]], float):
            s += str(f"{round(vals[gene['ind']], 5)}")
        else:
            s += str(f"{vals[gene['ind']]}")
        if i != len(genotype):
            s += ", "
    return s


log_map_seed = create_continous_genes(0.01, 1)
log_map_r = create_continous_genes(3.7, 4) # actual : 3.6-4
log_map_on = [0, 1]

lfsr_seed = [np.binary_repr(i, width=8) for i in create_discrete_genes(0, 255)]
lfsr_on = [0, 1]

rossler_c = [9, 10, 13, 18]
rossler_on = [0, 1]

tent_map_seed = create_continous_genes(0.01, 1)
tent_map_r = create_continous_genes(1, 2) # actual 1.5 - 2
tent_on = [0, 1]

henon_map_x_seed = create_continous_genes(0.1, 1)
henon_map_y_seed = create_continous_genes(0.1, 1)
henon_map_a = create_continous_genes(1, 2) # actual 1.5 - 2
henon_on = [0, 1]

log_map_seed_gene = make_gene(log_map_seed)
log_map_r_gene = make_gene(log_map_r)
log_map_on_gene = make_gene(log_map_on)

lfsr_seed_gene = make_gene(lfsr_seed)
lfsr_on_gene = make_gene(lfsr_on)

rossler_c_gene = make_gene(rossler_c)
rossler_on_gene = make_gene(rossler_on)

tent_map_seed_gene = make_gene(tent_map_seed)
tent_map_r_gene = make_gene(tent_map_r)
tent_on_gene = make_gene(tent_on)

henon_map_x_seed_gene = make_gene(henon_map_x_seed)
henon_map_y_seed_gene = make_gene(henon_map_y_seed)
henon_map_a_gene = make_gene(henon_map_a)
henon_on_gene = make_gene(henon_on)

genotype1 = [log_map_seed_gene, log_map_r_gene, log_map_on_gene,
             lfsr_seed_gene, lfsr_on_gene,
             rossler_c_gene, rossler_on_gene,
             tent_map_seed_gene, tent_map_r_gene, tent_on_gene,
             henon_map_x_seed_gene, henon_map_y_seed_gene, henon_map_a_gene, henon_on_gene]

os.chdir("/Users/rt/PycharmProjects/PixAdapt/")

filepath = f'FINAL DATABASE/images/Image_{IMAGE}.jpeg'


img_params = run_sim(filepath, genotype1)


cols = ["log_map_seed", "log_map_r", "log_map_on",
        "lfsr_seed", "lfsr_on",
        "rossler_c", "rossler_on",
        "tent_map_seed", "tent_map_r", "tent_on",
        "henon_map_x_seed", "henon_map_y_seed", "henon_map_a", "henon_on",
        "fitness"]

df = pd.DataFrame(img_params, columns=cols)
df.to_csv(f"csv files/hill climb/IMG_{IMAGE}.csv")
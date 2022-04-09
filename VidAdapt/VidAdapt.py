import cv2
import numpy as np
import matplotlib.pyplot as plt
from utilityFunctions.chaosFunctions import *
from utilityFunctions.utils import *
import cv2 as cv
import copy as cp
import random
import pandas as pd
from skimage.measure import shannon_entropy
import os
import time

NUM = 9

np.random.seed(NUM)
random.seed(NUM)

def getFrame(sec):
    video.set(cv2.CAP_PROP_POS_MSEC, sec*1000)
    hasFrames, image = video.read()
    return hasFrames, image

def run_sim(img, genotype1, seq):
    start = time.time()
    img_params = []
    seed_enc = seq
    cost, seq, PARAMETERS = encryptImage(img, genotype1, seed_enc)
    PARAMETERS.append(cost)
    img_params.append(PARAMETERS)
    print(f"After 1 generation, cost : {cost} | {print_genotype_vals(genotype1)}")
    epochs = 2

    while True:
        best = 33.46
        if abs(best - cost) < 1:
            break
        genotype2 = cp.deepcopy(genotype1)
        mutate_genotype(genotype2)
        cost2, seq, PARAMETERS = encryptImage(img, genotype2, seed_enc)
        PARAMETERS.append(cost)
        img_params.append(PARAMETERS)
        if abs(best - cost) > abs(best - cost2):
            cost = cost2
            genotype1 = cp.deepcopy(genotype2)
            print(f"After {epochs} generation, cost : {cost} | {print_genotype_vals(genotype1)}")
        epochs += 1
    end = time.time()
    print(f"OVERALL TIME TAKEN FOR {epochs} epochs : {round(end - start, 6)}")
    print(f"Genotype after simulation : {print_genotype_vals(genotype1)}\nFitness : {cost}\n\n")
    return img_params


def encryptImage(img, genotype, seq):
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

    PARAMETERS = [log_map_seed, log_map_r, log_map_on,
                  lfsr_seed, lfsr_on,
                  rossler_c, rossler_on,
                  tent_map_seed, tent_map_r, tent_on,
                  henon_map_x_seed, henon_map_y_seed, henon_map_a, henon_on]

    P = convert_to_binary(img)

    keys = [convert_to_binary(seq.astype("uint8"))]

    # generating pseudorandom numbers - Logistic map
    if log_map_on == 1:
        K1 = logistic_map(height, width, log_map_seed, log_map_r)
        keys.append(K1)

    # generating pseudorandom numbers - Linear feedback shift register
    if lfsr_on == 1:
        K2 = linear_shift_register(lfsr_seed, height, width)
        keys.append(K2)

    # # generating pseudorandom numbers - Rossler map
    # # params for Rossler map
    if rossler_on == 1:
        rossler_a = 0.1
        rossler_b = 0.1
        rossler_seed = [1, 1, 0]
        K3, y, z = rosslerMap(height, width, rossler_a, rossler_b, rossler_c, rossler_seed)
        # XOR
        # 1. K4 = rossler params - x and y
        # 2. K5 = rossler params - K4 and z
        # K_XY = np.array([np.binary_repr(int(i, 2) ^ int(j, 2), width=8) for i, j in zip(x.flatten(), y.flatten())])
        # K3 = np.array([np.binary_repr(int(i, 2) ^ int(j, 2), width=8) for i, j in zip(K_XY.flatten(), z.flatten())])
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

    # generating the encrypted image
    P_PRIME = np.array([np.binary_repr(int(i, 2) ^ int(j, 2), width=8) for i, j in zip(K, P.flatten())])
    chaos_encrypted_image = np.array([int(i, 2) for i in P_PRIME]).reshape((height, width)).astype('uint8')

    return calc_UACI(chaos_encrypted_image, img), chaos_encrypted_image, PARAMETERS


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
    return random.sample(range(start, end), 100)


def create_continous_genes(start, end):
    return np.random.uniform(start, end, size=100)


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
log_map_r = create_continous_genes(3.6, 4)
log_map_on = [0, 1]

lfsr_seed = [np.binary_repr(i, width=8) for i in create_discrete_genes(0, 255)]
lfsr_on = [0, 1]

# rossler_c = create_continous_genes(5, 30)
rossler_c = [9, 10, 13, 18]
rossler_on = [0, 1]

tent_map_seed = create_continous_genes(0.01, 1)
tent_map_r = create_continous_genes(1, 2)
tent_on = [0, 1]

henon_map_x_seed = create_continous_genes(0.1, 1)
henon_map_y_seed = create_continous_genes(0.1, 1)
henon_map_a = create_continous_genes(1, 1.4)
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
video = cv2.VideoCapture("FINAL DATABASE/videos/sample.mp4")

sec = 0
frameRate = 4
count = 1
success = getFrame(sec)
frames = []
while success:
    count += 1
    sec += frameRate
    sec = round(sec, 2)
    success, image = getFrame(sec)
    if success:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (500, 500))
        frames.append(image)
frames = np.array(frames)
seq = np.zeros(frames[0].shape)

cols = ["log_map_seed", "log_map_r", "log_map_on",
        "lfsr_seed", "lfsr_on",
        "rossler_c", "rossler_on",
        "tent_map_seed", "tent_map_r", "tent_on",
        "henon_map_x_seed", "henon_map_y_seed", "henon_map_a", "henon_on",
        "fitness"]

for img_num, img in enumerate(frames):
    print(f"IMAGE NUMBER {img_num+1}")
    data = run_sim(img, genotype1, seq)
    df = pd.DataFrame(data, columns=cols)
    df.to_csv(f"csv files/PA_2/IMG_{img_num+1}.csv")
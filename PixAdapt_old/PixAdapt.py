from utilityFunctions.chaosFunctions import *
from utilityFunctions.utils import *
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import copy as cp
import random
import pandas as pd
NUM=7

np.random.seed(NUM)
random.seed(NUM)

def run_sim(filepath, genotype1):
    costs, genes = [], []
    img = cv.imread(filepath, 0)
    cost = encryptImage(img, genotype1)
    costs.append(cost)
    genes.append(genotype1)
    print(f"After 1 generation, cost : {cost} | {print_genotype_vals(genotype1)}")
    epochs = 2

    while True:
        best = 33.46
        if abs(best - cost) < 0.23:
            break
        genotype2 = cp.deepcopy(genotype1)
        mutate_genotype(genotype2)
        cost2 = encryptImage(img, genotype2)
        costs.append(cost2)
        genes.append(genotype2)
        if abs(best - cost) > abs(best - cost2):
            cost = cost2
            genotype1 = cp.deepcopy(genotype2)
            print(f"After {epochs} generation, cost : {cost} | {print_genotype_vals(genotype1)}")
        epochs += 1

    print(f"Genotype after simulation : {print_genotype_vals(genotype1)}\nFitness : {cost}\n\n")
    return genotype1

def encryptImage(img, genotype):
    height, width = img.shape

    log_map_seed = genotype[0]["values"][genotype[0]["ind"]]
    log_map_r = genotype[1]["values"][genotype[1]["ind"]]
    log_map_on = genotype[2]["values"][genotype[2]["ind"]]

    lfsr_seed = genotype[3]["values"][genotype[3]["ind"]]
    lfsr_on = genotype[4]["values"][genotype[4]["ind"]]

    tent_map_seed = genotype[5]["values"][genotype[5]["ind"]]
    tent_map_r = genotype[6]["values"][genotype[6]["ind"]]
    tent_on = genotype[7]["values"][genotype[7]["ind"]]

    henon_map_x_seed = genotype[8]["values"][genotype[8]["ind"]]
    henon_map_y_seed = genotype[9]["values"][genotype[9]["ind"]]
    henon_map_a = genotype[10]["values"][genotype[10]["ind"]]
    henon_on = genotype[11]["values"][genotype[11]["ind"]]

    # rossler_c = genotype[5]["values"][genotype[5]["ind"]]
    # rossler_on = genotype[6]["values"][genotype[6]["ind"]]

    # tent_map_seed = genotype[7]["values"][genotype[7]["ind"]]
    # tent_map_r = genotype[8]["values"][genotype[8]["ind"]]
    # tent_on = genotype[9]["values"][genotype[9]["ind"]]
    #
    # henon_map_x_seed = genotype[10]["values"][genotype[10]["ind"]]
    # henon_map_y_seed = genotype[11]["values"][genotype[11]["ind"]]
    # henon_map_a = genotype[12]["values"][genotype[12]["ind"]]
    # henon_on = genotype[13]["values"][genotype[13]["ind"]]

    # temp = [log_map_seed, log_map_r, log_map_on,
    #         lfsr_seed, lfsr_on,
    #         rossler_c, rossler_on,
    #         tent_map_seed, tent_map_r, tent_on,
    #         henon_map_x_seed, henon_map_y_seed, henon_map_a, henon_on]
    #
    # cols = ["LM_SEED", "LM_R", "LM_ON",
    #         "LFSR_SEED", "LFSR_ON",
    #         "ROSSLER_C", "ROSSLER_ON",
    #         "TENT_SEED", "TENT_R", "TENT_ON",
    #         "HENON_X_SEED", "HENON_Y_SEED", "HENON_A", "HENON_ON"]
    #
    # data = {"LM_SEED": log_map_seed, "LM_R": log_map_r, "LM_ON": log_map_on,
    #         "LFSR_SEED": lfsr_seed, "LFSR_ON": lfsr_on,
    #         "ROSSLER_C": rossler_c, "ROSSLER_ON": rossler_on,
    #         "TENT_SEED": tent_map_seed, "TENT_R": tent_map_r, "TENT_ON": tent_on,
    #         "HENON_X_SEED": henon_map_x_seed, "HENON_Y_SEED": henon_map_y_seed, "HENON_A":henon_map_a, "HENON_ON": henon_on}
    #
    # df = pd.DataFrame(data, index = [0])
    # with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
    #     print(df)


    # print(log_map_seed, log_map_r, log_map_on)
    # print(lfsr_seed, lfsr_on)
    # print(rossler_c, rossler_on)
    # print(tent_map_seed, tent_map_r, tent_on)
    # print(henon_map_x_seed, henon_map_y_seed, henon_map_a, henon_on)

    P = convert_to_binary(img)

    keys = []

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

    K = np.array([np.binary_repr(i, width = 8) for i in np.zeros((height*width), dtype = int)])

    for k in keys:
        K = np.array([np.binary_repr(int(i, 2) ^ int(j, 2), width=8) for i, j in zip(K.flatten(), k.flatten())])

    # generating the encrypted image
    P_PRIME = np.array([np.binary_repr(int(i, 2) ^ int(j, 2), width=8) for i, j in zip(K, P.flatten())])
    chaos_encrypted_image = np.array([int(i, 2) for i in P_PRIME]).reshape((height, width)).astype('uint8')

    return calc_UACI(chaos_encrypted_image, img)

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
    return {"values" : values, "size" : len(values), "ind" : np.random.randint(low=0, high=len(values))}

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

lfsr_seed = [np.binary_repr(i, width = 8) for i in create_discrete_genes(0, 255)]
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

# rossler_c_gene = make_gene(rossler_c)
# rossler_on_gene = make_gene(rossler_on)

tent_map_seed_gene = make_gene(tent_map_seed)
tent_map_r_gene = make_gene(tent_map_r)
tent_on_gene = make_gene(tent_on)

henon_map_x_seed_gene = make_gene(henon_map_x_seed)
henon_map_y_seed_gene = make_gene(henon_map_y_seed)
henon_map_a_gene = make_gene(henon_map_a)
henon_on_gene = make_gene(henon_on)


print(lfsr_seed_gene)

# print(log_map_seed_gene, log_map_r_gene, log_map_on_gene)
# print(lfsr_seed_gene, lfsr_on_gene)
# print(tent_map_seed_gene, tent_map_r_gene, tent_on_gene)
# print(henon_map_x_seed_gene, henon_map_y_seed_gene, henon_map_a_gene, henon_on_gene)

# genotype1 = [log_map_seed_gene, log_map_r_gene, log_map_on_gene,
#              lfsr_seed_gene, lfsr_on_gene,
#              rossler_c_gene, rossler_on_gene,
#              tent_map_seed_gene, tent_map_r_gene, tent_on_gene,
#              henon_map_x_seed_gene, henon_map_y_seed_gene, henon_map_a_gene, henon_on_gene]

genotype1 = [log_map_seed_gene, log_map_r_gene, log_map_on_gene,
             lfsr_seed_gene, lfsr_on_gene,
             tent_map_seed_gene, tent_map_r_gene, tent_on_gene,
             henon_map_x_seed_gene, henon_map_y_seed_gene, henon_map_a_gene, henon_on_gene]


print("SYMMETRIC SMALL - FIRST IMAGE")
filepath = 'FINAL DATABASE/SYM SMALL/image_5.jpeg'
genotype1 = run_sim(filepath, genotype1)

# print("SYMMETRIC SMALL - SECOND IMAGE")
# filepath = 'FINAL DATABASE/SYM SMALL/IMG_SYM_SMALL_2.jpeg'
# genotype1 = run_sim(filepath, genotype1)
#
# print("SYMMETRIC SMALL - THIRD IMAGE")
# filepath = 'FINAL DATABASE/SYM SMALL/IMG_SYM_SMALL_3.jpeg'
# genotype1 = run_sim(filepath, genotype1)
#
# print("SYMMETRIC SMALL - FOURTH IMAGE")
# filepath = 'FINAL DATABASE/SYM SMALL/IMG_SYM_SMALL_4.jpeg'
# genotype1 = run_sim(filepath, genotype1)
#
# print("SYMMETRIC SMALL - FIFTH IMAGE")
# filepath = 'FINAL DATABASE/SYM SMALL/image_2.jpeg'
# genotype1 = run_sim(filepath, genotype1)
#
# print("\n\n---------------------------------------------\n\n")
#
# print("SYMMETRIC BIG - FIRST IMAGE")
# filepath = 'FINAL DATABASE/SYM BIG/image_4.jpeg'
# genotype1 = run_sim(filepath, genotype1)
#
# print("SYMMETRIC BIG - SECOND IMAGE")
# filepath = 'FINAL DATABASE/SYM BIG/IMG_SYM_BIG_2.jpeg'
# genotype1 = run_sim(filepath, genotype1)
#
# print("SYMMETRIC BIG - THIRD IMAGE")
# filepath = 'FINAL DATABASE/SYM BIG/IMG_SYM_BIG_3.jpeg'
# genotype1 = run_sim(filepath, genotype1)
#
# print("SYMMETRIC BIG - FOURTH IMAGE")
# filepath = 'FINAL DATABASE/SYM BIG/IMG_SYM_BIG_4.jpeg'
# genotype1 = run_sim(filepath, genotype1)
#
# print("SYMMETRIC BIG - FIFTH IMAGE")
# filepath = 'FINAL DATABASE/SYM BIG/image_3.jpeg'
# genotype1 = run_sim(filepath, genotype1)
#
# print("\n\n---------------------------------------------\n\n")
#
# print("ASYMMETRIC SMALL - FIRST IMAGE")
# filepath = 'FINAL DATABASE/ASYM SMALL/IMG_ASYM_SMALL_1.jpeg'
# genotype1 = run_sim(filepath, genotype1)
#
# print("ASYMMETRIC SMALL - SECOND IMAGE")
# filepath = 'FINAL DATABASE/ASYM SMALL/IMG_ASYM_SMALL_2.jpeg'
# genotype1 = run_sim(filepath, genotype1)
#
# print("ASYMMETRIC SMALL - THIRD IMAGE")
# filepath = 'FINAL DATABASE/ASYM SMALL/IMG_ASYM_SMALL_3.jpeg'
# genotype1 = run_sim(filepath, genotype1)
#
# print("ASYMMETRIC SMALL - FOURTH IMAGE")
# filepath = 'FINAL DATABASE/ASYM SMALL/IMG_ASYM_SMALL_4.jpeg'
# genotype1 = run_sim(filepath, genotype1)
#
# print("ASYMMETRIC SMALL - FIFTH IMAGE")
# filepath = 'FINAL DATABASE/ASYM SMALL/image_1.jpeg'
# genotype1 = run_sim(filepath, genotype1)
#
# print("\n\n---------------------------------------------\n\n")
#
# print("ASYMMETRIC BIG - FIRST IMAGE")
# filepath = 'FINAL DATABASE/ASYM BIG/IMG_ASYM_BIG_1.jpeg'
# genotype1 = run_sim(filepath, genotype1)
#
# print("ASYMMETRIC BIG - SECOND IMAGE")
# filepath = 'FINAL DATABASE/ASYM BIG/IMG_ASYM_BIG_2.jpeg'
# genotype1 = run_sim(filepath, genotype1)
#
# print("ASYMMETRIC BIG - THIRD IMAGE")
# filepath = 'FINAL DATABASE/ASYM BIG/IMG_ASYM_BIG_3.jpeg'
# genotype1 = run_sim(filepath, genotype1)
#
# print("ASYMMETRIC BIG - FOURTH IMAGE")
# filepath = 'FINAL DATABASE/ASYM BIG/IMG_ASYM_BIG_4.jpeg'
# genotype1 = run_sim(filepath, genotype1)
#
# print("ASYMMETRIC BIG - FIFTH IMAGE")
# filepath = 'FINAL DATABASE/ASYM BIG/IMG_ASYM_BIG_5.jpeg'
# genotype1 = run_sim(filepath, genotype1)
#
# print("\n\n---------------------------------------------\n\n")
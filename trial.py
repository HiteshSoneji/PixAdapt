# from utilityFunctions.chaosFunctions import *
# from utilityFunctions.utils import *
# import cv2 as cv
# import numpy as np
# import os
# import cv2
#
# def getFrame(sec):
#     video.set(cv2.CAP_PROP_POS_MSEC, sec*1000)
#     hasFrames, image = video.read()
#     return hasFrames, image
#
# os.chdir("/Users/rt/PycharmProjects/PixAdapt/")
# video = cv2.VideoCapture("FINAL DATABASE/videos/sample.mp4")
#
# sec = 0
# frameRate = 4
# count = 1
# success = getFrame(sec)
# frames = []
# while success:
#     count += 1
#     sec += frameRate
#     sec = round(sec, 2)
#     success, image = getFrame(sec)
#     if success:
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#         image = cv2.resize(image, (500, 500))
#         frames.append(image)
# frames = np.array(frames)
#
# for img in frames:
#     height, width = img.shape
#
#     P = convert_to_binary(img)
#
#     keys = []
#
#     # generating pseudorandom numbers - Logistic map
#
#     log_map_seed = 0.01
#     log_map_r = 3.99
#
#     lfsr_seed = '10001001'
#
#     tent_map_seed = 0.01
#     tent_map_r = 1.99
#
#     rossler_c = [9]
#
#     henon_map_x_seed = 0.01
#     henon_map_y_seed = 0.01
#     henon_map_a = 1.4
#
#     K1 = logistic_map(height, width, log_map_seed, log_map_r)
#     keys.append(K1)
#
#     # generating pseudorandom numbers - Linear feedback shift register
#     K2 = linear_shift_register(lfsr_seed, height, width)
#     keys.append(K2)
#
#     # # generating pseudorandom numbers - Rossler map
#     # # params for Rossler map
#     rossler_a = 0.1
#     rossler_b = 0.1
#     rossler_seed = [1, 1, 0]
#     K3, y, z = rosslerMap(height, width, rossler_a, rossler_b, rossler_c, rossler_seed)
#     keys.append(K3)
#
#     # Tent map
#     K4 = tentMap(height, width, tent_map_seed, tent_map_r)
#     keys.append(K4)
#
#     # Henon map
#     b = 0.3
#     K5 = henonMap(height, width, henon_map_x_seed, henon_map_y_seed, henon_map_a, b)
#     keys.append(K5)
#
#     K = np.array([np.binary_repr(i, width=8) for i in np.zeros((height * width), dtype=int)])
#
#     for k in keys:
#         K = np.array([np.binary_repr(int(i, 2) ^ int(j, 2), width=8) for i, j in zip(K.flatten(), k.flatten())])
#
#     # generating the encrypted image
#     P_PRIME = np.array([np.binary_repr(int(i, 2) ^ int(j, 2), width=8) for i, j in zip(K, P.flatten())])
#     chaos_encrypted_image = np.array([int(i, 2) for i in P_PRIME]).reshape((height, width)).astype('uint8')
#
#     print(f"Encrypted image entropy : {round(shannon_entropy(chaos_encrypted_image), 6)}")
#     print(f"NPCR : {round(calc_NPCR(chaos_encrypted_image, img), 6)}")
#     print(f"UACI : {round(calc_UACI(chaos_encrypted_image, img), 6)}")
#     print("\n\n\n")
#
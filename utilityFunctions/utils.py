import numpy as np
import cv2 as cv

def calc_UACI(enc_img, orig_img):
    temp = cv.absdiff(orig_img, enc_img)/255
    return (np.sum(temp) * 100) / temp.size

def calc_NPCR(enc_img, orig_img):
    temp = cv.absdiff(orig_img, enc_img)
    temp = temp.astype(np.uint8)
    return (np.count_nonzero(temp) * 100) / temp.size